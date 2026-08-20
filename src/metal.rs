// SPDX-License-Identifier: Apache-2.0
//! Apple/macOS detection via `system_profiler SPDisplaysDataType`, `sysctl
//! hw.memsize` and `vm_stat`. No Metal framework linkage required. Apple
//! Silicon uses a unified memory architecture (no dedicated VRAM), so
//! `total_bytes` falls back to total physical memory and the used/free split
//! comes from system-wide paging statistics.

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

/// Pull the page size out of the `vm_stat` header, which reads
/// `Mach Virtual Memory Statistics: (page size of 16384 bytes)`.
///
/// Taken from the report itself rather than `hw.pagesize` so the counts and
/// their multiplier always come from the same source: Apple Silicon pages are
/// 16 KiB where Intel Macs use 4 KiB, and mixing the two would be off by 4x.
#[allow(dead_code)] // used on macOS + in tests; unused on other targets
fn parse_page_size(text: &str) -> Option<u64> {
    let rest = text.split_once("page size of ")?.1;
    rest.split_once(" bytes")?.0.trim().parse().ok()
}

/// Look up one `Pages ...: <count>.` row in `vm_stat` output. The trailing
/// period is part of the format, not the number.
#[allow(dead_code)] // used on macOS + in tests; unused on other targets
fn page_count(text: &str, key: &str) -> Option<u64> {
    for line in text.lines() {
        if let Some(value) = line.trim().strip_prefix(key) {
            return value.trim().trim_end_matches('.').trim().parse().ok();
        }
    }
    None
}

/// Bytes in use, as macOS itself accounts for them: resident anonymous and
/// kernel pages (`active` + `wired down`) plus the compressor's footprint.
/// This is the figure Activity Monitor labels "Memory Used".
///
/// Apple Silicon shares one pool between CPU and GPU, so system-wide usage *is*
/// the GPU's usage; there is no separate VRAM to account for. Callers apply
/// this only to unified-memory GPUs — a discrete card on an Intel Mac reports
/// its own VRAM and must not be described by these numbers.
///
/// Reads *occupied by* rather than *stored in* the compressor: the former is
/// the compressor's real physical footprint, while the latter counts pages as
/// they were before compression and routinely exceeds installed memory.
#[allow(dead_code)] // used on macOS + in tests; unused on other targets
fn parse_vm_stat_used(text: &str) -> Option<u64> {
    let page_size = parse_page_size(text)?;
    let active = page_count(text, "Pages active:")?;
    let wired = page_count(text, "Pages wired down:")?;
    let compressor = page_count(text, "Pages occupied by compressor:")?;
    active
        .checked_add(wired)?
        .checked_add(compressor)?
        .checked_mul(page_size)
}

/// Parse plain-text `system_profiler SPDisplaysDataType` output into one
/// Metal GPU family for an Apple Silicon chip name — `Apple M2 Pro` is
/// `apple8`.
///
/// Derived from the name rather than queried, because the real source is
/// `MTLDevice.supportsFamily()`, which means linking Metal. Chips this table
/// does not know report `None`: a wrong family would be acted on, an absent one
/// would not. Non-Apple chipsets (`AMD Radeon Pro 5500M` on an Intel Mac) fall
/// out naturally, since the prefix will not match.
///
/// The rows come from Apple's published Metal Feature Set Tables (May 21,
/// 2026), which list M1 as `Apple7`, M2 as `Apple8`, M3 and M4 as `Apple9`, and
/// M5 as `Apple10`. Apple documents these per *series*, so one row covers a
/// generation's Pro, Max and Ultra variants — which is what falls out of
/// reading only the leading digits. Detection was exercised end to end on an
/// `Apple M2` (macOS 26.5).
#[allow(dead_code)] // used on macOS + in tests; unused on other targets
fn apple_family(name: &str) -> Option<crate::AppleFamily> {
    let rest = name.strip_prefix("Apple M")?;
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    let chip: u32 = rest[..end].parse().ok()?;
    let generation = match chip {
        1 => 7,
        2 => 8,
        // M3 introduced apple9 and M4 stayed on it; M5 moved to apple10.
        3 | 4 => 9,
        5 => 10,
        _ => return None,
    };
    Some(crate::AppleFamily::new(generation))
}

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
            let name = name.trim();
            gpus.push(GpuInfo {
                name: name.to_string(),
                vendor: Vendor::Apple,
                total_bytes: 0,
                free_bytes: None,
                used_bytes: None,
                arch_target: apple_family(name).map(crate::ArchTarget::Apple),
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

    // Apple Silicon reports no VRAM line — backfill from physical memory, and
    // with it the unified pool's usage. A zero total is what marks a GPU as
    // unified here, so a discrete card that already reported its own VRAM keeps
    // the `None` split rather than being handed system-wide figures.
    let memsize = sysctl_memsize();
    // Read lazily, the way the DRM probe defers its system-memory lookup: the
    // split costs a subprocess, and an Intel Mac with only a discrete card
    // never needs one.
    let mut split: Option<(Option<u64>, Option<u64>)> = None;
    for gpu in &mut gpus {
        if gpu.total_bytes == 0
            && let Some(mem) = memsize
        {
            gpu.total_bytes = mem;
            let (used, free) = *split.get_or_insert_with(|| unified_split(mem));
            gpu.used_bytes = used;
            gpu.free_bytes = free;
        }
    }
    // Nothing was parsed, so the loop above never ran and never read the split.
    if gpus.is_empty()
        && let Some(mem) = memsize
    {
        let (used, free) = unified_split(mem);
        gpus.push(GpuInfo {
            name: "Apple GPU".to_string(),
            vendor: Vendor::Apple,
            total_bytes: mem,
            free_bytes: free,
            used_bytes: used,
            arch_target: None,
        });
    }
    gpus
}

/// The `(used, free)` split of a unified pool of `total` bytes, or `(None,
/// None)` when `vm_stat` is unavailable or reports more than is installed —
/// a total that small would make `free` underflow, and a partial answer is
/// worse than admitting the split is unknown.
#[cfg(target_os = "macos")]
fn unified_split(total: u64) -> (Option<u64>, Option<u64>) {
    let Some(used) = vm_stat_used().filter(|&used| used <= total) else {
        return (None, None);
    };
    (Some(used), Some(total - used))
}

#[cfg(target_os = "macos")]
fn vm_stat_used() -> Option<u64> {
    let output = std::process::Command::new("vm_stat").output().ok()?;
    output
        .status
        .success()
        .then(|| parse_vm_stat_used(&String::from_utf8_lossy(&output.stdout)))
        .flatten()
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
    #[test]
    fn maps_apple_silicon_chips_to_metal_families() {
        use crate::AppleFamily;
        assert_eq!(apple_family("Apple M1"), Some(AppleFamily::new(7)));
        assert_eq!(apple_family("Apple M1 Max"), Some(AppleFamily::new(7)));
        assert_eq!(apple_family("Apple M2 Pro"), Some(AppleFamily::new(8)));
        assert_eq!(apple_family("Apple M3 Ultra"), Some(AppleFamily::new(9)));
        assert_eq!(apple_family("Apple M4"), Some(AppleFamily::new(9)));
        // M5 is the first generation on apple10.
        assert_eq!(apple_family("Apple M5"), Some(AppleFamily::new(10)));
        assert_eq!(apple_family("Apple M5 Pro"), Some(AppleFamily::new(10)));
        assert_eq!(apple_family("Apple M5 Max"), Some(AppleFamily::new(10)));
    }

    #[test]
    fn two_digit_families_render_unpacked() {
        // `apple10` must not collapse to `apple1`.
        assert_eq!(
            apple_family("Apple M5").map(|family| family.to_string()),
            Some("apple10".to_string())
        );
    }

    #[test]
    fn unknown_chips_report_no_family() {
        // A generation this table predates must not be guessed at.
        assert_eq!(apple_family("Apple M9"), None);
        // Multi-digit chips must not be read as their first digit.
        assert_eq!(apple_family("Apple M10"), None, "not M1");
        // Intel Macs and the sysctl fallback name.
        assert_eq!(apple_family("AMD Radeon Pro 5500M"), None);
        assert_eq!(apple_family("Intel Iris Plus Graphics"), None);
        assert_eq!(apple_family("Apple GPU"), None);
        assert_eq!(apple_family(""), None);
    }

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

    /// Trimmed from a real M2 running macOS 26.5, keeping the rows the parser
    /// reads plus the compressor pair it must tell apart.
    const VM_STAT: &str = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
                           Pages free:                                     3480.\n\
                           Pages active:                                  77803.\n\
                           Pages inactive:                                74322.\n\
                           Pages speculative:                              2300.\n\
                           Pages wired down:                             137287.\n\
                           Pages purgeable:                                   0.\n\
                           Pages stored in compressor:                  1263882.\n\
                           Pages occupied by compressor:                 192142.\n";

    #[test]
    fn reads_vm_stat_page_size_and_counts() {
        assert_eq!(parse_page_size(VM_STAT), Some(16384));
        assert_eq!(page_count(VM_STAT, "Pages active:"), Some(77803));
        assert_eq!(page_count(VM_STAT, "Pages wired down:"), Some(137_287));
        assert_eq!(page_count(VM_STAT, "Pages free:"), Some(3480));
        assert_eq!(page_count(VM_STAT, "Pages absent:"), None);
    }

    #[test]
    fn sums_used_pages_the_way_activity_monitor_does() {
        // active + wired + compressor, at 16 KiB per page.
        let expected = (77803 + 137_287 + 192_142) * 16384;
        assert_eq!(parse_vm_stat_used(VM_STAT), Some(expected));
        // ~6.2 GiB of the 8 GiB this fixture was taken from.
        assert!(expected < 8 * 1024 * 1024 * 1024);
    }

    #[test]
    fn used_counts_the_compressor_footprint_not_its_contents() {
        // "stored in" is the pre-compression count and dwarfs installed memory;
        // reading it instead of "occupied by" would report ~19 GiB on 8 GiB.
        let stored = 1_263_882u64 * 16384;
        assert!(
            stored > 8 * 1024 * 1024 * 1024,
            "fixture must expose the trap"
        );
        assert!(parse_vm_stat_used(VM_STAT).is_some_and(|used| used < stored));
    }

    #[test]
    fn vm_stat_parsing_needs_every_field() {
        // A report missing any row the sum needs yields nothing, rather than
        // silently undercounting.
        assert_eq!(parse_vm_stat_used(""), None);
        assert_eq!(parse_page_size("Pages free: 1.\n"), None);
        let no_wired = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
                        Pages active:                                  77803.\n\
                        Pages occupied by compressor:                 192142.\n";
        assert_eq!(parse_vm_stat_used(no_wired), None);
    }

    #[test]
    fn page_size_survives_a_four_kib_intel_header() {
        let intel = "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n";
        assert_eq!(parse_page_size(intel), Some(4096));
        assert_eq!(parse_page_size("(page size of many bytes)"), None);
        assert_eq!(parse_page_size("(page size of 4096)"), None);
    }
}
