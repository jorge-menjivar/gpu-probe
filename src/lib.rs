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
//!   ceiling, and AMD APUs their VRAM carveout plus GTT pool (see
//!   [`GpuInfo::total_bytes`]). AMD cards additionally report their `gfx`
//!   target from KFD sysfs — no `ROCm` install needed.
//! - **Apple/macOS**: `system_profiler` + `sysctl` for the chip and its memory
//!   ceiling, plus `vm_stat` for the used/free split (Apple Silicon reports
//!   unified memory, so that split is system-wide).
//!
//! Host toolchain properties are reported separately from any one GPU:
//! [`cuda_host`] for the CUDA driver, [`rocm_host`] for the `ROCm` install,
//! [`oneapi_host`] for the Intel `oneAPI` install, and [`vulkan_host`] for the
//! Vulkan runtime.
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
mod intel;
mod kfd;
mod metal;
mod nvidia;
mod oneapi;
mod rocm;
mod vulkan;

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
    ///
    /// An AMD APU reports its BIOS VRAM carveout plus the GTT pool the driver
    /// allocates from — the latter sized by the kernel's `ttm.pages_limit` —
    /// since the carveout alone is far below what the part can actually hand
    /// out (512 MiB of 14.5 GiB on one such part).
    pub total_bytes: u64,
    /// Free device memory in bytes, when known.
    pub free_bytes: Option<u64>,
    /// Used device memory in bytes, when known.
    pub used_bytes: Option<u64>,
    /// The architecture a prebuilt artifact must target to run on this GPU:
    /// [`ArchTarget::Gfx`] on AMD, from KFD sysfs, and [`ArchTarget::Sm`] on
    /// NVIDIA, from NVML.
    ///
    /// `None` when neither driver reports one — an Apple or Intel GPU, an AMD
    /// card on a kernel without KFD, or the `nvidia` feature disabled. The
    /// NVIDIA value also appears on [`CudaHost`], which reports device 0's
    /// alongside the host driver version; this field is per-GPU.
    pub arch_target: Option<ArchTarget>,
}

impl std::fmt::Display for GpuInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({}", self.name, self.vendor)?;
        if let Some(arch) = self.arch_target {
            write!(f, ", {arch}")?;
        }
        write!(f, "): {:.1} GiB total", gib(self.total_bytes))?;
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

/// CUDA compute capability, e.g. `8.6` for `sm_86`.
///
/// Ordered `major` first, so a host can be checked against a minimum:
///
/// ```
/// use gpu_probe::ComputeCapability;
/// assert!(ComputeCapability::new(8, 6) >= ComputeCapability::new(8, 0));
/// assert!(ComputeCapability::new(9, 0) >= ComputeCapability::new(8, 9));
/// ```
///
/// Constructible so callers can express such a requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeCapability {
    /// Major version — the `8` in `8.6`.
    pub major: u32,
    /// Minor version — the `6` in `8.6`.
    pub minor: u32,
}

impl ComputeCapability {
    /// Create a compute capability from its major and minor parts.
    #[must_use]
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }
}

impl std::fmt::Display for ComputeCapability {
    /// Renders as `8.6`, matching `nvidia-smi`'s `compute_cap`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// AMD GPU architecture target, e.g. `gfx1013` — the identifier a `ROCm`/HIP
/// code object is built for (`--offload-arch=gfx1013`).
///
/// The AMD counterpart of [`ComputeCapability`], and read the same way: to pick
/// a prebuilt artifact the host can actually run. Ordered `major` first, so a
/// host can be checked against a minimum:
///
/// ```
/// use gpu_probe::GfxTarget;
/// assert!(GfxTarget::new(10, 3, 0) >= GfxTarget::new(10, 1, 3));
/// assert!(GfxTarget::new(11, 0, 0) >= GfxTarget::new(10, 3, 0));
/// ```
///
/// Constructible so callers can express such a requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GfxTarget {
    /// Major version — the `10` in `gfx1013`.
    pub major: u32,
    /// Minor version — the `1` in `gfx1013`.
    pub minor: u32,
    /// Stepping — the `3` in `gfx1013`, and the `a` in `gfx90a`.
    pub step: u32,
}

impl GfxTarget {
    /// Create a target from its major, minor, and stepping parts.
    #[must_use]
    pub const fn new(major: u32, minor: u32, step: u32) -> Self {
        Self { major, minor, step }
    }
}

impl std::fmt::Display for GfxTarget {
    /// Renders as `gfx1013`, matching `--offload-arch`. Minor and stepping are
    /// single hex digits there, so `9.0.10` renders as `gfx90a`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gfx{}{:x}{:x}", self.major, self.minor, self.step)
    }
}

/// Intel GPU architecture family, e.g. [`IntelArch::XeHpg`] for an Arc A-series
/// card.
///
/// Coarser than its AMD and NVIDIA counterparts by necessity. Neither `i915`
/// nor `xe` publishes an architecture anywhere readable, so this is derived
/// from the PCI device id, which identifies the family reliably but not the
/// exact product. The `ocloc -device` value for an ahead-of-time build (`dg2`,
/// `acm-g10`, …) is more specific than this; treat it as "which generation is
/// this" rather than a literal compiler argument.
///
/// Deliberately not ordered: "newer" across integrated and discrete lines is
/// not a total order worth implying.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum IntelArch {
    /// Xe-LP — Tiger Lake, Rocket Lake, Alder Lake, Raptor Lake, DG1.
    XeLp,
    /// Xe-HPG — DG2, sold as Arc A-series (Alchemist).
    XeHpg,
    /// Xe-HPC — Ponte Vecchio, sold as Data Center GPU Max.
    XeHpc,
    /// Xe-LPG — Meteor Lake and Arrow Lake integrated graphics.
    XeLpg,
    /// Xe2 — Lunar Lake integrated graphics, and Arc B-series (Battlemage).
    Xe2,
}

impl std::fmt::Display for IntelArch {
    /// Renders as the lowercase family name — `xe-hpg`, `xe2`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            IntelArch::XeLp => "xe-lp",
            IntelArch::XeHpg => "xe-hpg",
            IntelArch::XeHpc => "xe-hpc",
            IntelArch::XeLpg => "xe-lpg",
            IntelArch::Xe2 => "xe2",
        })
    }
}

/// Apple GPU family, e.g. `apple8` for an M2.
///
/// The Metal feature tier a shader can be compiled against
/// (`MTLGPUFamily.apple8`). Ordered, so a minimum can be expressed:
///
/// ```
/// use gpu_probe::AppleFamily;
/// assert!(AppleFamily::new(9) >= AppleFamily::new(8));
/// ```
///
/// Unlike a `gfx` target or a compute capability, this does not select a build
/// artifact — a `.metallib` is not per-family — so it reads as a capability
/// tier. It is derived from the chip name `system_profiler` reports, since
/// querying it properly means linking Metal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AppleFamily {
    /// Family generation — the `8` in `apple8`.
    pub generation: u32,
}

impl AppleFamily {
    /// Create a family from its generation number.
    #[must_use]
    pub const fn new(generation: u32) -> Self {
        Self { generation }
    }
}

impl std::fmt::Display for AppleFamily {
    /// Renders as `apple8`, matching the `MTLGPUFamily` name.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "apple{}", self.generation)
    }
}

/// The architecture a prebuilt GPU artifact must target.
///
/// Each vendor names this differently but uses it the same way — to select a
/// build the device can actually run — so one field carries whichever form
/// applies. A GPU has at most one, which the type enforces.
///
/// ```
/// use gpu_probe::{ArchTarget, GfxTarget};
///
/// let target = ArchTarget::Gfx(GfxTarget::new(10, 1, 3));
/// assert_eq!(target.to_string(), "gfx1013");
/// assert_eq!(target.gfx(), Some(GfxTarget::new(10, 1, 3)));
/// assert_eq!(target.sm(), None);
/// ```
///
/// Deliberately not `Ord`: comparing an AMD target against an NVIDIA one has no
/// meaning. Order within a vendor by matching out [`GfxTarget`] or
/// [`ComputeCapability`], both of which are ordered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ArchTarget {
    /// AMD: the `--offload-arch` value a `ROCm`/HIP code object is built for.
    Gfx(GfxTarget),
    /// NVIDIA: the compute capability a CUDA artifact is built for.
    Sm(ComputeCapability),
    /// Intel: the GPU architecture family, from the PCI device id.
    Xe(IntelArch),
    /// Apple: the Metal GPU family. A capability tier rather than a build
    /// target — see [`AppleFamily`].
    Apple(AppleFamily),
}

impl ArchTarget {
    /// The AMD target, or `None` when this names another vendor's.
    #[must_use]
    pub const fn gfx(self) -> Option<GfxTarget> {
        match self {
            Self::Gfx(target) => Some(target),
            _ => None,
        }
    }

    /// The NVIDIA compute capability, or `None` when this names another
    /// vendor's.
    #[must_use]
    pub const fn sm(self) -> Option<ComputeCapability> {
        match self {
            Self::Sm(capability) => Some(capability),
            _ => None,
        }
    }

    /// The Intel architecture family, or `None` when this names another
    /// vendor's.
    #[must_use]
    pub const fn xe(self) -> Option<IntelArch> {
        match self {
            Self::Xe(arch) => Some(arch),
            _ => None,
        }
    }

    /// The Apple GPU family, or `None` when this names another vendor's.
    #[must_use]
    pub const fn apple(self) -> Option<AppleFamily> {
        match self {
            Self::Apple(family) => Some(family),
            _ => None,
        }
    }
}

impl std::fmt::Display for ArchTarget {
    /// Renders in the form each vendor's toolchain expects: `gfx1013` for
    /// `--offload-arch`, `sm_89` for CUDA.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gfx(target) => write!(f, "{target}"),
            Self::Sm(capability) => {
                write!(f, "sm_{}{}", capability.major, capability.minor)
            }
            Self::Xe(arch) => write!(f, "{arch}"),
            Self::Apple(family) => write!(f, "{family}"),
        }
    }
}

/// A CUDA version, e.g. `12.9`.
///
/// Ordered `major` first, so a host can be checked against a minimum:
///
/// ```
/// use gpu_probe::CudaVersion;
/// assert!(CudaVersion::new(12, 9) >= CudaVersion::new(12, 0));
/// ```
///
/// Constructible so callers can express such a requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CudaVersion {
    /// Major version — the `12` in `12.9`.
    pub major: u32,
    /// Minor version — the `9` in `12.9`.
    pub minor: u32,
}

impl CudaVersion {
    /// Create a CUDA version from its major and minor parts.
    #[must_use]
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }
}

impl std::fmt::Display for CudaVersion {
    /// Renders as `12.9`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// Host-wide CUDA properties reported by the NVIDIA driver.
///
/// These describe the host and its driver rather than any one GPU, which is why
/// they are separate from the per-GPU [`GpuInfo`]. Consumers typically use them
/// to select a prebuilt artifact compatible with the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct CudaHost {
    /// Compute capability of device 0.
    pub compute_capability: ComputeCapability,
    /// Version of the installed CUDA driver.
    pub driver_version: CudaVersion,
}

/// A `ROCm` release version, e.g. `6.2.4`.
///
/// Ordered `major` first, so a host can be checked against a minimum:
///
/// ```
/// use gpu_probe::RocmVersion;
/// assert!(RocmVersion::new(6, 2, 4) >= RocmVersion::new(6, 0, 0));
/// ```
///
/// Constructible so callers can express such a requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RocmVersion {
    /// Major version — the `6` in `6.2.4`.
    pub major: u32,
    /// Minor version — the `2` in `6.2.4`.
    pub minor: u32,
    /// Patch version — the `4` in `6.2.4`.
    pub patch: u32,
}

impl RocmVersion {
    /// Create a version from its major, minor, and patch parts.
    #[must_use]
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
}

impl std::fmt::Display for RocmVersion {
    /// Renders as `6.2.4`, matching the `.info/version` file it comes from.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// The host's `ROCm` installation.
///
/// The AMD counterpart of [`CudaHost`], but a narrower one: there is no
/// driver-side version to report, so this describes the userspace install only.
/// See [`rocm_host`] for what its absence does and does not imply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct RocmHost {
    /// Installed `ROCm` release.
    pub version: RocmVersion,
}

/// Parse a dotted version — `6.2.4`, `2024.2` — into major, minor, and patch.
///
/// A trailing build suffix (`6.2.4-123`) is dropped: it identifies a package
/// build, not the release. Patch defaults to `0`, since some releases ship only
/// `major.minor`. Shared by the `ROCm` and `oneAPI` probes, which read the same
/// shape of version out of different places.
fn parse_dotted_version(text: &str) -> Option<(u32, u32, u32)> {
    let version = text.trim().split(['-', '+']).next()?;
    let mut parts = version.split('.');
    let major = parts.next()?.trim().parse().ok()?;
    let minor = parts.next()?.trim().parse().ok()?;
    let patch = match parts.next() {
        Some(patch) => patch.trim().parse().ok()?,
        None => 0,
    };
    Some((major, minor, patch))
}

/// An Intel `oneAPI` toolkit version, e.g. `2024.2.1`.
///
/// Ordered `major` first — which for `oneAPI` is the release year — so a host
/// can be checked against a minimum:
///
/// ```
/// use gpu_probe::OneApiVersion;
/// assert!(OneApiVersion::new(2025, 0, 0) >= OneApiVersion::new(2024, 2, 0));
/// ```
///
/// Constructible so callers can express such a requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OneApiVersion {
    /// Major version — the release year, the `2024` in `2024.2.1`.
    pub major: u32,
    /// Minor version — the `2` in `2024.2.1`.
    pub minor: u32,
    /// Patch version — the `1` in `2024.2.1`.
    pub patch: u32,
}

impl OneApiVersion {
    /// Create a version from its major, minor, and patch parts.
    #[must_use]
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
}

impl std::fmt::Display for OneApiVersion {
    /// Renders as `2024.2.1`, matching the install directory it comes from.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// The host's Intel `oneAPI` installation.
///
/// The Intel counterpart of [`RocmHost`], and equally narrow: a userspace
/// install, with no driver version behind it. See [`oneapi_host`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct OneApiHost {
    /// Installed `oneAPI` toolkit release.
    pub version: OneApiVersion,
}

/// A Vulkan API version, e.g. `1.3.280`.
///
/// Ordered `major` first, so a host can be checked against a minimum:
///
/// ```
/// use gpu_probe::VulkanVersion;
/// assert!(VulkanVersion::new(1, 3, 280) >= VulkanVersion::new(1, 2, 0));
/// ```
///
/// Constructible so callers can express such a requirement.
///
/// Gate on `major`/`minor`. Vulkan's patch number is the spec header revision
/// and carries no feature guarantee — a 1.4.354 driver and a 1.4.357 loader
/// are both Vulkan 1.4 — so a patch-sensitive comparison rejects builds that
/// would have run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VulkanVersion {
    /// Major version — the `1` in `1.3.280`.
    pub major: u32,
    /// Minor version — the `3` in `1.3.280`.
    pub minor: u32,
    /// Patch version — the `280` in `1.3.280`. The spec header revision, not
    /// a feature level.
    pub patch: u32,
}

impl VulkanVersion {
    /// Create a version from its major, minor, and patch parts.
    #[must_use]
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
}

impl std::fmt::Display for VulkanVersion {
    /// Renders as `1.3.280`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// The host's Vulkan runtime.
///
/// Reported from the loader and the installed ICD manifests, so — unlike
/// [`RocmHost`] and [`OneApiHost`] — this describes a runtime that is actually
/// present rather than a toolkit that may be. There is no architecture field:
/// SPIR-V is portable and driver-compiled, so a Vulkan build has no per-GPU
/// target to match.
///
/// Host-level, not per-device: see [`api_version`](Self::api_version) for what
/// the number does and does not promise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct VulkanHost {
    /// Highest API version any installed driver advertises in its ICD
    /// manifest.
    ///
    /// A driver's own static declaration on disk, which makes it two things it
    /// is easy to mistake it for:
    ///
    /// - **Not per-device.** With two drivers installed — an AMD iGPU beside
    ///   an NVIDIA dGPU, say — this is the higher of the two and may describe
    ///   neither card. A specific device's version comes from
    ///   `vkGetPhysicalDeviceProperties`, which means linking the loader and
    ///   creating an instance; this crate deliberately does neither.
    /// - **Not the loader's instance version.** That is what `vulkaninfo` and
    ///   `vkEnumerateInstanceVersion` report, and it is usually the newer of
    ///   the two, so the numbers routinely disagree. The driver's is the one
    ///   that binds in practice: loaders track the current headers while
    ///   drivers implement features on their own schedule.
    ///
    /// Compare on `major`/`minor` only — see [`VulkanVersion`].
    pub api_version: VulkanVersion,
}

/// Detect all GPUs visible on the host.
///
/// Best-effort: spawns only read-only platform queries (NVML, `system_profiler`,
/// `sysctl`, `vm_stat`) and reads sysfs. Returns an empty `Vec` on unsupported
/// platforms or when no GPU is found.
#[must_use]
pub fn detect() -> Vec<GpuInfo> {
    let mut gpus = Vec::new();
    gpus.extend(nvidia::detect());
    gpus.extend(drm::detect());
    gpus.extend(metal::detect());
    gpus
}

/// Host-wide CUDA properties, or `None` when NVML is unavailable — no NVIDIA
/// driver, the `nvidia` feature disabled, no device, or a driver reporting
/// values that aren't usable.
///
/// Shares the one process-wide NVML handle with [`detect`], so calling this on
/// a timer does not accumulate resources.
///
/// ```no_run
/// use gpu_probe::ComputeCapability;
///
/// if let Some(cuda) = gpu_probe::cuda_host() {
///     println!("sm_{}{} on CUDA {}",
///         cuda.compute_capability.major,
///         cuda.compute_capability.minor,
///         cuda.driver_version);
///
///     if cuda.compute_capability >= ComputeCapability::new(8, 0) {
///         // pick an Ampere-or-newer build
///     }
/// }
/// ```
#[must_use]
pub fn cuda_host() -> Option<CudaHost> {
    nvidia::cuda_host()
}

/// The host's `ROCm` installation, or `None` when `ROCm` is not installed.
///
/// Read from `$ROCM_PATH/.info/version`, falling back to `/opt/rocm` — the
/// plain text file the `rocm-core` package writes. Nothing is linked or
/// executed, so this costs one file read.
///
/// `Some` is the signal that `ROCm` is installed and the host can run its
/// builds. `None` is weaker: the install was not found at the prefixes above,
/// which a distro shipping `ROCm` into `/usr` — or a container carrying only
/// the runtime libraries — will trigger despite working. Treat `Some` as proof
/// and `None` as "probably not, worth confirming".
///
/// `None` does not mean the GPU is unusable for compute: the kernel side is a
/// separate component, and what a build has to target is
/// [`GpuInfo::arch_target`], reported with no `ROCm` installed at all.
///
/// ```no_run
/// use gpu_probe::RocmVersion;
///
/// if let Some(rocm) = gpu_probe::rocm_host()
///     && rocm.version >= RocmVersion::new(6, 0, 0)
/// {
///     // pick a `ROCm` 6 build
/// }
/// ```
#[must_use]
pub fn rocm_host() -> Option<RocmHost> {
    rocm::host()
}

/// The host's Intel `oneAPI` installation, or `None` when it is not installed.
///
/// Read from the component layout under `$ONEAPI_ROOT`, falling back to
/// `/opt/intel/oneapi`. Nothing is linked or executed.
///
/// Narrower than it looks: this reports the **toolkit**, not the GPU runtime.
/// A host running compute through a distro-packaged Level Zero driver with no
/// toolkit installed reports `None`, because reading that runtime's version
/// requires linking it rather than reading a file. So `Some` proves the
/// toolkit is present, while `None` is the weakest negative of the three
/// probes — it does not rule out a usable Level Zero runtime.
///
/// ```no_run
/// use gpu_probe::OneApiVersion;
///
/// if let Some(oneapi) = gpu_probe::oneapi_host()
///     && oneapi.version >= OneApiVersion::new(2024, 0, 0)
/// {
///     // pick a oneAPI 2024-or-newer build
/// }
/// ```
#[must_use]
pub fn oneapi_host() -> Option<OneApiHost> {
    oneapi::host()
}

/// The host's Vulkan runtime, or `None` when no loader is installed.
///
/// Read from `libvulkan.so.1` plus the ICD manifests under
/// `/usr/share/vulkan/icd.d`. Nothing is linked or executed, so this costs a
/// handful of file reads.
///
/// The version is the highest any installed *driver* advertises — neither the
/// loader's instance version nor any one GPU's, both of which would require
/// calling into the loader. See [`VulkanHost::api_version`].
///
/// ```no_run
/// use gpu_probe::VulkanVersion;
///
/// if let Some(vulkan) = gpu_probe::vulkan_host()
///     && vulkan.api_version >= VulkanVersion::new(1, 2, 0)
/// {
///     // pick a Vulkan 1.2-or-newer build
/// }
/// ```
#[must_use]
pub fn vulkan_host() -> Option<VulkanHost> {
    vulkan::host()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_dotted_versions_in_both_shapes() {
        assert_eq!(parse_dotted_version("6.2.4-123"), Some((6, 2, 4)));
        assert_eq!(parse_dotted_version("2024.2"), Some((2024, 2, 0)));
        assert_eq!(parse_dotted_version("  5.7.1  "), Some((5, 7, 1)));
        assert_eq!(parse_dotted_version("6"), None, "a bare major is not one");
        assert_eq!(parse_dotted_version("latest"), None);
        assert_eq!(parse_dotted_version(""), None);
    }

    #[test]
    fn oneapi_version_renders_with_patch() {
        assert_eq!(OneApiVersion::new(2024, 2, 1).to_string(), "2024.2.1");
        assert_eq!(OneApiVersion::new(2025, 0, 0).to_string(), "2025.0.0");
    }

    #[test]
    fn oneapi_host_is_stable_across_calls() {
        // Environment-dependent: most hosts have no oneAPI, which is a valid,
        // passing environment. A filesystem read must not vary between calls.
        assert_eq!(oneapi_host(), oneapi_host());
    }

    #[test]
    fn rocm_version_renders_with_patch() {
        assert_eq!(RocmVersion::new(6, 2, 4).to_string(), "6.2.4");
        assert_eq!(RocmVersion::new(6, 2, 0).to_string(), "6.2.0");
    }

    #[test]
    fn rocm_host_is_stable_across_calls() {
        // Environment-dependent: most hosts have no ROCm, which is a valid,
        // passing environment. A filesystem read must not vary between calls.
        assert_eq!(rocm_host(), rocm_host());
    }

    #[test]
    fn vulkan_version_renders_with_patch() {
        assert_eq!(VulkanVersion::new(1, 3, 280).to_string(), "1.3.280");
        assert_eq!(VulkanVersion::new(1, 0, 0).to_string(), "1.0.0");
        // A three-digit patch is the norm for Vulkan headers, and must not be
        // packed or truncated the way `1.3` alone would be.
        assert_eq!(VulkanVersion::new(1, 10, 5).to_string(), "1.10.5");
    }

    #[test]
    fn vulkan_host_is_stable_across_calls() {
        // Environment-dependent: a host with no loader is a valid, passing
        // environment. A filesystem read must not vary between calls.
        assert_eq!(vulkan_host(), vulkan_host());
    }

    #[test]
    fn vulkan_host_carries_only_an_api_version() {
        // No `arch_target` counterpart, deliberately: SPIR-V is portable and
        // driver-compiled, so there is no per-GPU target to match. This pins
        // that shape, and the `Copy`/`Eq` derives callers rely on.
        let host = VulkanHost {
            api_version: VulkanVersion::new(1, 3, 280),
        };
        let copied = host;
        assert_eq!(copied, host);
        assert_eq!(copied.api_version, VulkanVersion::new(1, 3, 280));
        assert_ne!(
            host,
            VulkanHost {
                api_version: VulkanVersion::new(1, 2, 0)
            }
        );
    }

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
            arch_target: None,
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
            arch_target: None,
        };
        let shown = gpu.to_string();
        assert!(shown.contains("8.0 GiB total"));
        assert!(!shown.contains("free"));
    }

    #[test]
    fn display_includes_gfx_target_when_present() {
        let gpu = GpuInfo {
            name: "AMD cyan_skillfish".to_string(),
            vendor: Vendor::Amd,
            total_bytes: 15 * 1024 * 1024 * 1024,
            free_bytes: None,
            used_bytes: None,
            arch_target: Some(ArchTarget::Gfx(GfxTarget::new(10, 1, 3))),
        };
        assert!(gpu.to_string().contains("(AMD, gfx1013)"));
    }

    #[test]
    fn display_includes_compute_capability_when_present() {
        let gpu = GpuInfo {
            name: "NVIDIA GeForce RTX 4090".to_string(),
            vendor: Vendor::Nvidia,
            total_bytes: 24 * 1024 * 1024 * 1024,
            free_bytes: None,
            used_bytes: None,
            arch_target: Some(ArchTarget::Sm(ComputeCapability::new(8, 9))),
        };
        assert!(gpu.to_string().contains("(NVIDIA, sm_89)"));
    }

    #[test]
    fn arch_target_unwraps_only_its_own_vendor() {
        let amd = ArchTarget::Gfx(GfxTarget::new(10, 1, 3));
        assert_eq!(amd.gfx(), Some(GfxTarget::new(10, 1, 3)));
        assert_eq!(amd.sm(), None);

        let nvidia = ArchTarget::Sm(ComputeCapability::new(8, 9));
        assert_eq!(nvidia.sm(), Some(ComputeCapability::new(8, 9)));
        assert_eq!(nvidia.gfx(), None);
    }

    #[test]
    fn arch_target_accessors_are_exclusive_across_all_vendors() {
        let targets = [
            ArchTarget::Gfx(GfxTarget::new(10, 1, 3)),
            ArchTarget::Sm(ComputeCapability::new(8, 9)),
            ArchTarget::Xe(IntelArch::XeHpg),
            ArchTarget::Apple(AppleFamily::new(8)),
        ];
        for target in targets {
            let hits = [
                target.gfx().is_some(),
                target.sm().is_some(),
                target.xe().is_some(),
                target.apple().is_some(),
            ];
            assert_eq!(
                hits.iter().filter(|hit| **hit).count(),
                1,
                "{target} must answer exactly one accessor",
            );
        }
    }

    #[test]
    fn intel_and_apple_targets_render_by_family() {
        assert_eq!(ArchTarget::Xe(IntelArch::XeHpg).to_string(), "xe-hpg");
        assert_eq!(ArchTarget::Xe(IntelArch::Xe2).to_string(), "xe2");
        assert_eq!(ArchTarget::Apple(AppleFamily::new(8)).to_string(), "apple8");
    }

    #[test]
    fn apple_families_are_ordered() {
        assert!(AppleFamily::new(9) > AppleFamily::new(8));
        assert!(AppleFamily::new(8) > AppleFamily::new(7));
    }

    #[test]
    fn arch_target_renders_per_vendor_toolchain() {
        // `sm_89`, not the bare `8.9` `ComputeCapability` renders on its own,
        // which would read as a version number in this position.
        assert_eq!(
            ArchTarget::Sm(ComputeCapability::new(8, 9)).to_string(),
            "sm_89"
        );
        assert_eq!(
            ArchTarget::Gfx(GfxTarget::new(10, 1, 3)).to_string(),
            "gfx1013"
        );
    }

    #[test]
    fn gfx_target_renders_as_offload_arch() {
        assert_eq!(GfxTarget::new(10, 1, 3).to_string(), "gfx1013");
        assert_eq!(GfxTarget::new(10, 3, 0).to_string(), "gfx1030");
        assert_eq!(GfxTarget::new(11, 0, 0).to_string(), "gfx1100");
        // Stepping 10 is the `a` in `gfx90a`, not a literal "10".
        assert_eq!(GfxTarget::new(9, 0, 10).to_string(), "gfx90a");
        assert_eq!(GfxTarget::new(9, 4, 2).to_string(), "gfx942");
    }

    #[test]
    fn gfx_targets_order_major_first() {
        assert!(GfxTarget::new(11, 0, 0) > GfxTarget::new(10, 3, 0));
        assert!(GfxTarget::new(10, 3, 0) > GfxTarget::new(10, 1, 3));
        assert!(GfxTarget::new(10, 1, 3) > GfxTarget::new(10, 1, 0));
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
            arch_target: None,
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
    fn versions_display_as_major_dot_minor() {
        assert_eq!(ComputeCapability::new(8, 6).to_string(), "8.6");
        assert_eq!(CudaVersion::new(12, 9).to_string(), "12.9");
        // A two-digit minor stays unambiguous — the reason these aren't packed
        // into a single integer.
        assert_eq!(ComputeCapability::new(8, 10).to_string(), "8.10");
    }

    #[test]
    fn versions_order_by_major_then_minor() {
        assert!(ComputeCapability::new(8, 6) > ComputeCapability::new(8, 0));
        assert!(ComputeCapability::new(9, 0) > ComputeCapability::new(8, 9));
        assert_eq!(ComputeCapability::new(8, 6), ComputeCapability::new(8, 6));
        assert!(CudaVersion::new(12, 9) > CudaVersion::new(12, 0));
        assert!(CudaVersion::new(13, 0) > CudaVersion::new(12, 9));
        // Packing as `major * 10 + minor` would collide here: 8.10 and 9.0
        // both pack to 90, which is why the parts are kept separate.
        assert!(ComputeCapability::new(9, 0) > ComputeCapability::new(8, 10));
    }

    #[test]
    fn cuda_host_is_environment_dependent_but_coherent() {
        // No NVIDIA driver is a valid, passing environment.
        if let Some(cuda) = cuda_host() {
            assert!(
                cuda.compute_capability.major > 0,
                "a real device has a nonzero major capability",
            );
            assert!(cuda.driver_version.major > 0, "a real driver has a version");
            assert_eq!(
                cuda_host(),
                Some(cuda),
                "host/driver properties must be stable across calls",
            );
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
            arch_target: None,
        };
        assert_eq!(base.clone(), base);
        let mut other = base.clone();
        other.vendor = Vendor::Amd;
        assert_ne!(base, other);
    }
}
