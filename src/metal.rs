// SPDX-License-Identifier: Apache-2.0
//! Apple/macOS detection via `system_profiler SPDisplaysDataType` and `sysctl
//! hw.memsize`. No Metal framework linkage required. Apple Silicon uses a
//! unified memory architecture (no dedicated VRAM), so `total_bytes` falls
//! back to total physical memory.

/// Map a vendor id as printed by `system_profiler` (e.g. `0x106b`) to a
/// [`Vendor`](crate::Vendor).
#[allow(dead_code)] // used on macOS + in tests; unused on other targets
fn vendor_from_id(id: &str) -> crate::Vendor {
    use crate::Vendor;
    match id.trim().to_ascii_lowercase().as_str() {
        "0x106b" => Vendor::Apple,
        "0x1002" => Vendor::Amd,
        "0x10de" => Vendor::Nvidia,
        "0x8086" => Vendor::Intel,
        _ => Vendor::Unknown,
    }
}

/// Parse a `system_profiler` VRAM value like `"8 GB"` or `"1536 MB"` into bytes
/// (binary units).
#[allow(dead_code)] // used on macOS + in tests; unused on other targets
fn parse_vram(value: &str) -> Option<u64> {
    let (num, unit) = value.trim().split_once(' ')?;
    let amount: u64 = num.trim().parse().ok()?;
    let mult: u64 = match unit.trim().to_ascii_uppercase().as_str() {
        "GB" => 1024 * 1024 * 1024,
        "MB" => 1024 * 1024,
        "KB" => 1024,
        _ => return None,
    };
    Some(amount * mult)
}

/// Parse `sysctl -n hw.memsize` output (a decimal byte count).
#[allow(dead_code)] // used on macOS + in tests; unused on other targets
fn parse_memsize(content: &str) -> Option<u64> {
    content.trim().parse().ok()
}

/// Parse plain-text `system_profiler SPDisplaysDataType` output into one
/// [`GpuInfo`](crate::GpuInfo) per "Chipset Model:" block. `total_bytes` is `0`
/// when no VRAM line is present (Apple Silicon unified memory); callers
/// backfill it from physical memory.
#[allow(dead_code)] // used on macOS + in tests; unused on other targets
fn parse_system_profiler(text: &str) -> Vec<crate::GpuInfo> {
    use crate::{GpuInfo, Vendor};

    let mut gpus: Vec<GpuInfo> = Vec::new();
    for raw in text.lines() {
        let line = raw.trim();
        if let Some(name) = line.strip_prefix("Chipset Model:") {
            gpus.push(GpuInfo {
                name: name.trim().to_string(),
                vendor: Vendor::Apple,
                total_bytes: 0,
                free_bytes: None,
                used_bytes: None,
            });
        } else if let Some(gpu) = gpus.last_mut() {
            if let Some(v) = line.strip_prefix("Vendor:") {
                // e.g. "Apple (0x106b)" — pull out the parenthesized id.
                if let Some(id) = v.split('(').nth(1).and_then(|s| s.split(')').next()) {
                    gpu.vendor = vendor_from_id(id);
                }
            } else if let Some(v) = line
                .strip_prefix("VRAM (Total):")
                .or_else(|| line.strip_prefix("VRAM (Dynamic, Max):"))
                && let Some(bytes) = parse_vram(v)
            {
                gpu.total_bytes = bytes;
            }
        }
    }
    gpus
}

#[cfg(target_os = "macos")]
pub(crate) fn detect() -> Vec<crate::GpuInfo> {
    use crate::{GpuInfo, Vendor};

    let mut gpus = std::process::Command::new("system_profiler")
        .arg("SPDisplaysDataType")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| parse_system_profiler(&String::from_utf8_lossy(&o.stdout)))
        .unwrap_or_default();

    // Apple Silicon reports no VRAM line — backfill from physical memory.
    let memsize = sysctl_memsize();
    for gpu in &mut gpus {
        if gpu.total_bytes == 0
            && let Some(mem) = memsize
        {
            gpu.total_bytes = mem;
        }
    }
    if gpus.is_empty()
        && let Some(mem) = memsize
    {
        gpus.push(GpuInfo {
            name: "Apple GPU".to_string(),
            vendor: Vendor::Apple,
            total_bytes: mem,
            free_bytes: None,
            used_bytes: None,
        });
    }
    gpus
}

#[cfg(target_os = "macos")]
fn sysctl_memsize() -> Option<u64> {
    let output = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| parse_memsize(&String::from_utf8_lossy(&output.stdout)))
        .flatten()
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn detect() -> Vec<crate::GpuInfo> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vendor;

    #[test]
    fn apple_silicon_block_has_no_vram() {
        let text = "Graphics/Displays:\n\n    Apple M2 Pro:\n\n      Chipset Model: Apple M2 Pro\n      Type: GPU\n      Vendor: Apple (0x106b)\n      Metal Support: Metal 3\n";
        let gpus = parse_system_profiler(text);
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].name, "Apple M2 Pro");
        assert_eq!(gpus[0].vendor, Vendor::Apple);
        assert_eq!(
            gpus[0].total_bytes, 0,
            "unified memory; backfilled in detect()"
        );
    }

    #[test]
    fn intel_mac_discrete_gpu_reports_vram() {
        let text = "      Chipset Model: AMD Radeon Pro 5500M\n      Type: GPU\n      Bus: PCIe\n      VRAM (Total): 8 GB\n      Vendor: AMD (0x1002)\n";
        let gpus = parse_system_profiler(text);
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].vendor, Vendor::Amd);
        assert_eq!(gpus[0].total_bytes, 8 * 1024 * 1024 * 1024);
    }

    #[test]
    fn dynamic_vram_line_is_parsed() {
        let text = "      Chipset Model: Intel Iris Pro\n      VRAM (Dynamic, Max): 1536 MB\n      Vendor: Intel (0x8086)\n";
        let gpus = parse_system_profiler(text);
        assert_eq!(gpus[0].vendor, Vendor::Intel);
        assert_eq!(gpus[0].total_bytes, 1536 * 1024 * 1024);
    }

    #[test]
    fn vram_and_memsize_parsers() {
        assert_eq!(parse_vram("8 GB"), Some(8 * 1024 * 1024 * 1024));
        assert_eq!(parse_vram("1536 MB"), Some(1536 * 1024 * 1024));
        assert_eq!(parse_vram("weird"), None);
        assert_eq!(parse_memsize("17179869184\n"), Some(17_179_869_184));
    }

    #[test]
    fn vendor_id_mapping() {
        assert_eq!(vendor_from_id("0x106b"), Vendor::Apple);
        assert_eq!(vendor_from_id("0x10DE"), Vendor::Nvidia);
        assert_eq!(vendor_from_id("0x1002"), Vendor::Amd);
        assert_eq!(vendor_from_id("0xbeef"), Vendor::Unknown);
    }

    #[test]
    fn parses_multiple_gpu_blocks() {
        let text = "      Chipset Model: AMD Radeon Pro 5500M\n      VRAM (Total): 8 GB\n      Vendor: AMD (0x1002)\n      Chipset Model: Intel UHD Graphics 630\n      VRAM (Dynamic, Max): 1536 MB\n      Vendor: Intel (0x8086)\n";
        let gpus = parse_system_profiler(text);
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].vendor, Vendor::Amd);
        assert_eq!(gpus[0].total_bytes, 8 * 1024 * 1024 * 1024);
        assert_eq!(gpus[1].vendor, Vendor::Intel);
        assert_eq!(gpus[1].total_bytes, 1536 * 1024 * 1024);
    }

    #[test]
    fn empty_output_yields_no_gpus() {
        assert!(parse_system_profiler("").is_empty());
        // Lines before any "Chipset Model:" have no GPU to attach to.
        assert!(parse_system_profiler("Graphics/Displays:\n      VRAM (Total): 8 GB\n").is_empty());
    }

    #[test]
    fn vendor_line_without_parens_keeps_default() {
        let text = "      Chipset Model: Mystery GPU\n      Vendor: sieve\n";
        let gpus = parse_system_profiler(text);
        assert_eq!(gpus.len(), 1);
        // No "(id)" to parse, so the Apple default placed at block start stands.
        assert_eq!(gpus[0].vendor, Vendor::Apple);
    }

    #[test]
    fn malformed_vram_leaves_total_at_zero() {
        let text =
            "      Chipset Model: Broken\n      VRAM (Total): lots\n      Vendor: Apple (0x106b)\n";
        let gpus = parse_system_profiler(text);
        assert_eq!(gpus[0].total_bytes, 0);
    }

    #[test]
    fn parse_vram_rejects_unknown_units_and_bad_numbers() {
        assert_eq!(parse_vram("8 TB"), None);
        assert_eq!(parse_vram("8"), None);
        assert_eq!(parse_vram("eight GB"), None);
        assert_eq!(
            parse_vram("512 kb"),
            Some(512 * 1024),
            "unit is case-insensitive"
        );
    }

    #[test]
    fn parse_memsize_rejects_non_numeric() {
        assert_eq!(parse_memsize("nope"), None);
        assert_eq!(parse_memsize(""), None);
        assert_eq!(parse_memsize("0"), Some(0));
    }
}
