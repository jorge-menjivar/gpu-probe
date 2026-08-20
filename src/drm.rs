// SPDX-License-Identifier: Apache-2.0
//! AMD and Intel detection via Linux DRM sysfs (`/sys/class/drm/card*/device`).
//! No `ROCm`, Level Zero, or vendor libraries required.
//!
//! GPUs that expose `mem_info_vram_total` report dedicated VRAM with used/free.
//! That attribute is provided by `amdgpu`; whether Intel's `i915`/`xe` drivers
//! expose it on discrete Arc cards is unverified (untested on real hardware).
//! Any card without it — most Intel iGPUs, and Intel discrete cards that don't
//! implement the file — falls back to the shared system-memory ceiling with
//! used/free `None`. NVIDIA cards are skipped here — they're handled by NVML in
//! the `nvidia` module.
//!
//! AMD APUs do expose `mem_info_vram_total`, but only as a small BIOS carveout
//! (as little as 512 MiB, for instance) — the memory such a part really
//! allocates from is the GTT pool in `mem_info_gtt_total`, sized by the kernel's
//! `ttm.pages_limit`. [`fold_gtt`] adds the two together for cards that look
//! integrated, so a unified-memory part reports what it can actually hand out.

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

/// Largest `mem_info_vram_total` still treated as an APU carveout rather than
/// real dedicated VRAM.
///
/// There is no sysfs flag for "this is an APU" — `amdgpu` only exposes that
/// through a debugfs file (`amdgpu_gpu_info`) that unprivileged callers can't
/// read, and through an ioctl this crate doesn't make. Size is the practical
/// stand-in: APU carveouts are typically 512 MiB to 2 GiB, while no current
/// discrete AMD card ships under 4 GiB. The threshold errs toward leaving a
/// card alone — an APU configured with a carveout above it is under-reported
/// (the pre-existing behaviour) rather than a discrete card being inflated by
/// system memory it shouldn't count.
#[allow(dead_code)] // used on Linux + in tests; unused on other targets
const INTEGRATED_VRAM_MAX: u64 = 2 * 1024 * 1024 * 1024;

/// Total and used memory for a card reporting `vram_total`, folding in the GTT
/// pool when the VRAM looks like an APU carveout.
///
/// `used` is `None` unless both pools report it — summing only the half that
/// answered would understate usage and so overstate free memory.
///
/// The carveout is counted, not dropped, which is worth knowing because the two
/// runtimes on such a part disagree. On an APU with a 512 MiB carveout over a
/// 14 GiB GTT pool:
///
/// ```text
/// vram_total + gtt_total   15_569_256_448   what this reports
/// Vulkan (RADV) heaps      15_569_256_448   identical, to the byte
/// KFD memory bank          15_032_385_536   GTT alone, carveout excluded
/// ```
///
/// `ROCm` sees the smaller figure because KFD publishes only the GTT-backed
/// bank. Counting the carveout matches Vulkan exactly, and costs nothing in
/// accuracy: its usage is folded into `used` as well, so a carveout consumed by
/// the framebuffer is subtracted straight back out of free memory.
#[allow(dead_code)] // used on Linux + in tests; unused on other targets
fn fold_gtt(
    vram_total: u64,
    vram_used: Option<u64>,
    gtt_total: Option<u64>,
    gtt_used: Option<u64>,
) -> (u64, Option<u64>) {
    match gtt_total {
        Some(gtt_total) if vram_total <= INTEGRATED_VRAM_MAX => (
            vram_total.saturating_add(gtt_total),
            vram_used.zip(gtt_used).map(|(v, g)| v.saturating_add(g)),
        ),
        _ => (vram_total, vram_used),
    }
}

/// Minor number of a card's render node (`renderD128` → `128`), which is how a
/// DRM card lines up with its KFD compute node.
#[cfg(target_os = "linux")]
fn render_minor(device: &std::path::Path) -> Option<u32> {
    std::fs::read_dir(device.join("drm"))
        .ok()?
        .flatten()
        .find_map(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .strip_prefix("renderD")?
                .parse()
                .ok()
        })
}

/// Read one sysfs attribute under a card's `device/` directory as a byte count.
#[cfg(target_os = "linux")]
fn read_bytes(device: &std::path::Path, attr: &str) -> Option<u64> {
    std::fs::read_to_string(device.join(attr))
        .ok()
        .and_then(|s| parse_bytes(&s))
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
    // Likewise: the KFD tree is only consulted once an AMD card shows up.
    let mut compute: Option<Vec<crate::kfd::ComputeNode>> = None;

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

        // AMD cards carry a `gfx` target and an ASIC codename in KFD sysfs; the
        // codename is a better name than the `card1` node path.
        let node = if vendor == crate::Vendor::Amd {
            let nodes = compute.get_or_insert_with(crate::kfd::nodes);
            render_minor(&device)
                .and_then(|minor| nodes.iter().find(|n| n.render_minor == minor))
                .cloned()
        } else {
            None
        };
        let name = node.as_ref().and_then(|n| n.name.clone()).map_or_else(
            || format!("{vendor} GPU ({card})"),
            |asic| format!("{vendor} {asic}"),
        );
        let arch_target = match vendor {
            crate::Vendor::Amd => node.map(|n| crate::ArchTarget::Gfx(n.gfx_target)),
            // Intel publishes no architecture anywhere readable, so it comes
            // from the PCI device id this same directory already exposes.
            crate::Vendor::Intel => std::fs::read_to_string(device.join("device"))
                .ok()
                .and_then(|id| crate::intel::parse_device_id(&id))
                .and_then(crate::intel::arch_for_device_id)
                .map(crate::ArchTarget::Xe),
            _ => None,
        };

        if let Some(vram_total) = read_bytes(&device, "mem_info_vram_total") {
            // Dedicated VRAM — plus the GTT pool when this is an APU carveout.
            let (total, used) = fold_gtt(
                vram_total,
                read_bytes(&device, "mem_info_vram_used"),
                read_bytes(&device, "mem_info_gtt_total"),
                read_bytes(&device, "mem_info_gtt_used"),
            );
            gpus.push(GpuInfo {
                name,
                vendor,
                total_bytes: total,
                free_bytes: used.map(|u| total.saturating_sub(u)),
                used_bytes: used,
                arch_target,
            });
        } else {
            // Integrated GPU with no VRAM pool at all (typical Intel iGPU):
            // report the shared system-memory ceiling, with no used/free.
            if sysmem.is_none() {
                sysmem = system_memory();
            }
            if let Some(total) = sysmem {
                gpus.push(GpuInfo {
                    name,
                    vendor,
                    total_bytes: total,
                    free_bytes: None,
                    used_bytes: None,
                    arch_target,
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
    fn apu_carveout_folds_in_gtt_pool() {
        // A 512 MiB carveout over a 14 GiB GTT pool, as an APU reports it.
        let (total, used) = fold_gtt(
            536_870_912,
            Some(522_469_376),
            Some(15_032_385_536),
            Some(705_626_112),
        );
        assert_eq!(total, 15_569_256_448);
        assert_eq!(used, Some(1_228_095_488));
    }

    #[test]
    fn discrete_vram_ignores_gtt_pool() {
        // A 24 GiB card keeps its own total even though GTT is large.
        let (total, used) = fold_gtt(
            25_757_220_864,
            Some(1_073_741_824),
            Some(33_483_649_024),
            Some(268_435_456),
        );
        assert_eq!(total, 25_757_220_864);
        assert_eq!(used, Some(1_073_741_824));
    }

    #[test]
    fn threshold_is_inclusive_at_two_gib() {
        let (at, _) = fold_gtt(INTEGRATED_VRAM_MAX, None, Some(1024), None);
        assert_eq!(
            at,
            INTEGRATED_VRAM_MAX + 1024,
            "2 GiB still counts as a carveout"
        );
        let (over, _) = fold_gtt(INTEGRATED_VRAM_MAX + 1, None, Some(1024), None);
        assert_eq!(over, INTEGRATED_VRAM_MAX + 1, "just over is dedicated VRAM");
    }

    #[test]
    fn missing_gtt_total_leaves_card_untouched() {
        let (total, used) = fold_gtt(536_870_912, Some(1024), None, Some(2048));
        assert_eq!(total, 536_870_912);
        assert_eq!(used, Some(1024));
    }

    #[test]
    fn used_needs_both_pools_to_report() {
        // Summing one pool alone would understate usage, so report neither.
        assert_eq!(fold_gtt(1024, Some(512), Some(4096), None).1, None);
        assert_eq!(fold_gtt(1024, None, Some(4096), Some(512)).1, None);
    }

    #[test]
    fn folded_totals_saturate_instead_of_wrapping() {
        let (total, used) = fold_gtt(1024, Some(u64::MAX), Some(u64::MAX), Some(u64::MAX));
        assert_eq!(total, u64::MAX);
        assert_eq!(used, Some(u64::MAX));
    }

    #[test]
    fn sysfs_vendor_trims_and_lowercases() {
        assert_eq!(sysfs_vendor("  0x1002\n"), Some(Vendor::Amd));
        assert_eq!(sysfs_vendor("0X8086"), Some(Vendor::Intel));
        assert_eq!(sysfs_vendor(""), None);
        assert_eq!(sysfs_vendor("1002"), None, "missing 0x prefix is unknown");
    }
}
