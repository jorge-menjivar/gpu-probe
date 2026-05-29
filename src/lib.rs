// SPDX-License-Identifier: Apache-2.0
//! Cross-platform GPU memory (VRAM) detection with **no vendor SDKs**.
//!
//! `gpu_probe` reports the GPUs visible on the host and how much memory each
//! has, using only facilities the OS or driver already ship:
//!
//! - **NVIDIA** (Linux, Windows): NVML (`libnvidia-ml`) via `nvml-wrapper`,
//!   loaded at runtime. The CUDA toolkit is not required and nothing links at
//!   build time. Behind the default `nvidia` feature.
//! - **AMD & Intel** (Linux): DRM sysfs under `/sys/class/drm`. Discrete cards
//!   report dedicated VRAM; integrated GPUs report the shared system-memory
//!   ceiling (see [`GpuInfo::total_bytes`]).
//! - **Apple/macOS**: `system_profiler` + `sysctl` (Apple Silicon reports
//!   unified memory).
//!
//! Detection is best-effort: [`detect`] returns an empty `Vec` when no GPU is
//! found or the platform is unsupported — never an error.
//!
//! ```no_run
//! for gpu in gpu_probe::detect() {
//!     println!("{gpu}");
//! }
//! ```

mod drm;
mod metal;
mod nvidia;

/// GPU hardware vendor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Vendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Unknown,
}

impl std::fmt::Display for Vendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Vendor::Nvidia => "NVIDIA",
            Vendor::Amd => "AMD",
            Vendor::Intel => "Intel",
            Vendor::Apple => "Apple",
            Vendor::Unknown => "Unknown",
        })
    }
}

/// A single detected GPU and its memory.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct GpuInfo {
    /// Human-readable name (e.g. `"NVIDIA GeForce RTX 4090"`).
    pub name: String,
    /// Hardware vendor.
    pub vendor: Vendor,
    /// Total memory in bytes. For discrete GPUs this is dedicated VRAM; for
    /// integrated/unified GPUs (Intel iGPUs, AMD APUs, Apple Silicon) it is the
    /// shared system-memory ceiling available to the GPU, not a dedicated pool.
    pub total_bytes: u64,
    /// Free device memory in bytes, when known.
    pub free_bytes: Option<u64>,
    /// Used device memory in bytes, when known.
    pub used_bytes: Option<u64>,
}

impl std::fmt::Display for GpuInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({}): {:.1} GiB total",
            self.name,
            self.vendor,
            gib(self.total_bytes)
        )?;
        if let Some(free) = self.free_bytes {
            write!(f, ", {:.1} GiB free", gib(free))?;
        }
        Ok(())
    }
}

#[allow(clippy::cast_precision_loss)] // display-only; the imprecision is cosmetic
fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// Detect all GPUs visible on the host.
///
/// Best-effort: spawns only read-only platform queries (NVML, `system_profiler`,
/// `sysctl`) and reads sysfs. Returns an empty `Vec` on unsupported platforms
/// or when no GPU is found.
#[must_use]
pub fn detect() -> Vec<GpuInfo> {
    let mut gpus = Vec::new();
    gpus.extend(nvidia::detect());
    gpus.extend(drm::detect());
    gpus.extend(metal::detect());
    gpus
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_never_panics() {
        // Environment-dependent (may be empty on headless CI); exercise the
        // full path plus the Display impl without asserting a GPU exists.
        for gpu in detect() {
            assert!(!gpu.name.is_empty());
            let _ = gpu.to_string();
        }
    }

    #[test]
    fn display_includes_free_when_present() {
        let gpu = GpuInfo {
            name: "Test GPU".to_string(),
            vendor: Vendor::Nvidia,
            total_bytes: 24 * 1024 * 1024 * 1024,
            free_bytes: Some(12 * 1024 * 1024 * 1024),
            used_bytes: Some(12 * 1024 * 1024 * 1024),
        };
        let shown = gpu.to_string();
        assert!(shown.contains("NVIDIA"));
        assert!(shown.contains("24.0 GiB total"));
        assert!(shown.contains("12.0 GiB free"));
    }

    #[test]
    fn display_omits_free_when_absent() {
        let gpu = GpuInfo {
            name: "AMD GPU (card0)".to_string(),
            vendor: Vendor::Amd,
            total_bytes: 8 * 1024 * 1024 * 1024,
            free_bytes: None,
            used_bytes: None,
        };
        let shown = gpu.to_string();
        assert!(shown.contains("8.0 GiB total"));
        assert!(!shown.contains("free"));
    }

    #[test]
    fn vendor_display_covers_every_variant() {
        assert_eq!(Vendor::Nvidia.to_string(), "NVIDIA");
        assert_eq!(Vendor::Amd.to_string(), "AMD");
        assert_eq!(Vendor::Intel.to_string(), "Intel");
        assert_eq!(Vendor::Apple.to_string(), "Apple");
        assert_eq!(Vendor::Unknown.to_string(), "Unknown");
    }

    #[test]
    fn gib_converts_using_binary_units() {
        assert!((gib(0) - 0.0).abs() < f64::EPSILON);
        assert!((gib(1024 * 1024 * 1024) - 1.0).abs() < f64::EPSILON);
        // 1.5 GiB exercises the fractional path the Display rounds to one place.
        assert!((gib(3 * 1024 * 1024 * 1024 / 2) - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn display_rounds_to_one_decimal_place() {
        // 25 GiB + 256 MiB -> 25.25 GiB, which "{:.1}" renders as "25.2".
        let gpu = GpuInfo {
            name: "Rounding".to_string(),
            vendor: Vendor::Nvidia,
            total_bytes: 25 * 1024 * 1024 * 1024 + 256 * 1024 * 1024,
            free_bytes: None,
            used_bytes: None,
        };
        assert!(gpu.to_string().contains("25.2 GiB total"));
    }

    #[test]
    fn detect_results_have_consistent_memory_fields() {
        // Environment-dependent; asserts invariants only for whatever is present.
        for gpu in detect() {
            assert!(!gpu.name.is_empty());
            if let Some(free) = gpu.free_bytes {
                assert!(free <= gpu.total_bytes, "free must not exceed total");
            }
            if let (Some(free), Some(used)) = (gpu.free_bytes, gpu.used_bytes) {
                assert!(
                    free.saturating_add(used) <= gpu.total_bytes.saturating_add(used),
                    "free/used must be coherent",
                );
            }
        }
    }

    #[test]
    fn gpu_info_equality_compares_all_fields() {
        let base = GpuInfo {
            name: "G".to_string(),
            vendor: Vendor::Intel,
            total_bytes: 16 * 1024 * 1024 * 1024,
            free_bytes: None,
            used_bytes: None,
        };
        assert_eq!(base.clone(), base);
        let mut other = base.clone();
        other.vendor = Vendor::Amd;
        assert_ne!(base, other);
    }
}
