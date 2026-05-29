// SPDX-License-Identifier: Apache-2.0
//! AMD and Intel detection via Linux DRM sysfs (`/sys/class/drm/card*/device`).
//! No `ROCm`, Level Zero, or vendor libraries required.
//!
//! GPUs that expose `mem_info_vram_total` (amdgpu, plus Intel drivers that
//! implement it) report dedicated VRAM with used/free. Integrated GPUs (AMD
//! APUs, Intel iGPUs) have no dedicated VRAM, so `total_bytes` is the shared
//! system-memory ceiling and used/free are `None`. NVIDIA cards are skipped
//! here — they're handled by NVML in the `nvidia` module.

/// True for a primary DRM card node (`card0`, `card1`, …) — not a connector
/// (`card0-eDP-1`) or a render node (`renderD128`).
#[allow(dead_code)] // used on Linux + in tests; unused on other targets
fn is_card_dir(name: &str) -> bool {
    name.strip_prefix("card")
        .is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
}

/// Parse a sysfs integer file (e.g. `mem_info_vram_total`) holding a decimal
/// byte count.
#[allow(dead_code)] // used on Linux + in tests; unused on other targets
fn parse_bytes(content: &str) -> Option<u64> {
    content.trim().parse().ok()
}

/// Map a PCI vendor id (`device/vendor`, e.g. `0x1002`) to the [`Vendor`] this
/// scanner owns. NVIDIA (`0x10de`) returns `None` — NVML handles it — as do
/// unknown vendors.
#[allow(dead_code)] // used on Linux + in tests; unused on other targets
fn sysfs_vendor(id: &str) -> Option<crate::Vendor> {
    use crate::Vendor;
    match id.trim().to_ascii_lowercase().as_str() {
        "0x1002" => Some(Vendor::Amd),
        "0x8086" => Some(Vendor::Intel),
        _ => None,
    }
}

/// Parse `MemTotal:` (in kB) from `/proc/meminfo` contents into bytes.
#[allow(dead_code)] // used on Linux + in tests; unused on other targets
fn parse_meminfo_total(content: &str) -> Option<u64> {
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn system_memory() -> Option<u64> {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| parse_meminfo_total(&s))
}

#[cfg(target_os = "linux")]
pub(crate) fn detect() -> Vec<crate::GpuInfo> {
    use crate::GpuInfo;

    let mut gpus = Vec::new();
    let Ok(entries) = std::fs::read_dir("/sys/class/drm") else {
        return gpus;
    };
    // Fetched lazily: only an integrated GPU needs it, and most hosts have none.
    let mut sysmem: Option<u64> = None;

    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let card = file_name.to_string_lossy();
        if !is_card_dir(&card) {
            continue;
        }
        let device = entry.path().join("device");
        let Some(vendor) = std::fs::read_to_string(device.join("vendor"))
            .ok()
            .and_then(|v| sysfs_vendor(&v))
        else {
            continue;
        };

        if let Some(total) = std::fs::read_to_string(device.join("mem_info_vram_total"))
            .ok()
            .and_then(|s| parse_bytes(&s))
        {
            // Discrete GPU with dedicated VRAM.
            let used = std::fs::read_to_string(device.join("mem_info_vram_used"))
                .ok()
                .and_then(|s| parse_bytes(&s));
            gpus.push(GpuInfo {
                name: format!("{vendor} GPU ({card})"),
                vendor,
                total_bytes: total,
                free_bytes: used.map(|u| total.saturating_sub(u)),
                used_bytes: used,
            });
        } else {
            // Integrated GPU (APU / iGPU): no dedicated VRAM — report the
            // shared system-memory ceiling, with no precise used/free.
            if sysmem.is_none() {
                sysmem = system_memory();
            }
            if let Some(total) = sysmem {
                gpus.push(GpuInfo {
                    name: format!("{vendor} GPU ({card})"),
                    vendor,
                    total_bytes: total,
                    free_bytes: None,
                    used_bytes: None,
                });
            }
        }
    }
    gpus
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn detect() -> Vec<crate::GpuInfo> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vendor;

    #[test]
    fn card_dir_matches_only_primary_nodes() {
        assert!(is_card_dir("card0"));
        assert!(is_card_dir("card12"));
        assert!(!is_card_dir("card0-eDP-1"));
        assert!(!is_card_dir("renderD128"));
        assert!(!is_card_dir("controlD64"));
        assert!(!is_card_dir("card"));
        assert!(!is_card_dir("cardX"));
    }

    #[test]
    fn parses_sysfs_byte_count() {
        assert_eq!(parse_bytes("17163091968\n"), Some(17_163_091_968));
        assert_eq!(parse_bytes("  8589934592 "), Some(8_589_934_592));
        assert_eq!(parse_bytes("nope"), None);
        assert_eq!(parse_bytes(""), None);
    }

    #[test]
    fn vendor_ids_amd_and_intel_only() {
        assert_eq!(sysfs_vendor("0x1002"), Some(Vendor::Amd));
        assert_eq!(sysfs_vendor("0x8086\n"), Some(Vendor::Intel));
        assert_eq!(sysfs_vendor("0x10DE"), None, "NVIDIA is handled by NVML");
        assert_eq!(sysfs_vendor("0xffff"), None);
    }

    #[test]
    fn parses_meminfo_memtotal() {
        let meminfo = "MemTotal:       32789868 kB\nMemFree:         1234 kB\n";
        assert_eq!(parse_meminfo_total(meminfo), Some(32_789_868 * 1024));
        assert_eq!(parse_meminfo_total("MemFree: 100 kB"), None);
    }

    #[test]
    fn meminfo_handles_empty_and_malformed() {
        assert_eq!(parse_meminfo_total(""), None);
        assert_eq!(parse_meminfo_total("MemTotal:"), None);
        assert_eq!(parse_meminfo_total("MemTotal:        kB"), None);
        assert_eq!(parse_meminfo_total("MemTotal: notanumber kB"), None);
    }

    #[test]
    fn parse_bytes_handles_zero_and_large_values() {
        assert_eq!(parse_bytes("0"), Some(0));
        assert_eq!(parse_bytes(&u64::MAX.to_string()), Some(u64::MAX));
        // Overflowing u64 must fail rather than wrap.
        assert_eq!(parse_bytes("99999999999999999999999"), None);
        assert_eq!(parse_bytes("-1"), None);
        assert_eq!(parse_bytes("12 34"), None);
    }

    #[test]
    fn card_dir_rejects_render_and_control_nodes() {
        assert!(!is_card_dir("renderD129"));
        assert!(!is_card_dir("by-path"));
        assert!(!is_card_dir(""));
        // Leading zeros are still all-digits, so they count as a card node.
        assert!(is_card_dir("card007"));
    }

    #[test]
    fn sysfs_vendor_trims_and_lowercases() {
        assert_eq!(sysfs_vendor("  0x1002\n"), Some(Vendor::Amd));
        assert_eq!(sysfs_vendor("0X8086"), Some(Vendor::Intel));
        assert_eq!(sysfs_vendor(""), None);
        assert_eq!(sysfs_vendor("1002"), None, "missing 0x prefix is unknown");
    }
}
