<!-- SPDX-License-Identifier: Apache-2.0 -->
# gpu-probe

[![crates.io](https://img.shields.io/crates/v/gpu-probe.svg)](https://crates.io/crates/gpu-probe)
[![docs.rs](https://img.shields.io/docsrs/gpu-probe)](https://docs.rs/gpu-probe)
[![coverage](https://img.shields.io/endpoint?url=https://jorge-menjivar.github.io/gpu-probe/coverage.json)](https://jorge-menjivar.github.io/gpu-probe/)

Cross-platform GPU memory (VRAM) detection for Rust — no vendor SDKs, nothing to install beyond your GPU driver.

| Vendor | Linux | Windows | macOS | Backend |
|:-------|:-----:|:-------:|:-----:|:--------|
| NVIDIA | ✅ | ✅ | ✅<sup>†</sup> | NVML · `system_profiler` |
| AMD    | ✅ | — | ✅<sup>†</sup> | DRM sysfs · KFD · `system_profiler` |
| Intel  | ✅ | — | ✅<sup>†</sup> | DRM sysfs · `system_profiler` |
| Apple  | — | — | ✅ | `system_profiler` + `sysctl` |

<sup>†</sup> Intel Macs only — discrete and integrated GPUs are read from `system_profiler`.

Best-effort: you get an empty list on unsupported platforms, never an error.

**Note:** Verified on NVIDIA hardware and on an AMD BC-250 APU. The Intel and Apple paths are implemented but not yet confirmed on real devices, as are the ROCm and oneAPI install probes — if something doesn't work, please [open an issue](https://github.com/jorge-menjivar/gpu-probe/issues). Help from the community confirming detection on Intel and Apple GPUs is very much appreciated.

## Install

```toml
[dependencies]
gpu-probe = "0.1"
```

NVIDIA support pulls in `nvml-wrapper`. For AMD/Apple-only builds, drop it:

```toml
gpu-probe = { version = "0.1", default-features = false }
```

## Usage

```rust
for gpu in gpu_probe::detect() {
    println!("{gpu}");
    // NVIDIA GeForce RTX 3090 (NVIDIA, sm_86): 24.0 GiB total, 9.8 GiB free
    // AMD cyan_skillfish (AMD, gfx1013): 14.5 GiB total, 14.0 GiB free
}
```

`detect()` returns `Vec<GpuInfo>`:

```rust
pub struct GpuInfo {
    pub name: String,
    pub vendor: Vendor,            // Nvidia | Amd | Intel | Apple | Unknown
    pub total_bytes: u64,
    pub free_bytes: Option<u64>,
    pub used_bytes: Option<u64>,
    pub arch_target: Option<ArchTarget>,  // Gfx | Sm | Xe | Apple
}
```

Check whether a model fits, or pick the emptiest GPU:

```rust
let need = 16 * 1024 * 1024 * 1024; // 16 GiB

let fits = gpu_probe::detect()
    .iter()
    .any(|g| g.free_bytes.unwrap_or(g.total_bytes) >= need);

let emptiest = gpu_probe::detect()
    .into_iter()
    .max_by_key(|g| g.free_bytes.unwrap_or(g.total_bytes));
```

Or run the bundled example: `cargo run --example detect`.

### Architecture targets

Which prebuilt artifact a GPU can run is reported per device, as one field
carrying whichever form the vendor uses. A GPU has at most one, enforced by the
type rather than by convention:

```rust
use gpu_probe::ArchTarget;

for gpu in gpu_probe::detect() {
    match gpu.arch_target {
        // AMD: the ROCm/HIP `--offload-arch` value
        Some(ArchTarget::Gfx(gfx)) => println!("build {gfx}"),   // gfx1013
        // NVIDIA: the CUDA compute capability
        Some(ArchTarget::Sm(sm)) => println!("build sm_{}{}", sm.major, sm.minor),
        // Intel: the architecture family
        Some(ArchTarget::Xe(arch)) => println!("build {arch}"),  // xe-hpg
        // Apple: the Metal feature tier
        Some(ArchTarget::Apple(family)) => println!("targets {family}"),  // apple8
        // `ArchTarget` is #[non_exhaustive], so a wildcard is required — new
        // vendors land as new variants without breaking this match.
        _ => {}
    }
}
```

The four are not equally precise, and the table says why:

| vendor | value | source | selects a build? |
|:-------|:------|:-------|:-----------------|
| AMD | `gfx1013` | KFD sysfs | yes — `--offload-arch` |
| NVIDIA | `sm_86` | NVML | yes — `-arch` |
| Intel | `xe-hpg` | PCI device id | family only; `ocloc -device` is finer |
| Apple | `apple8` | chip name | no — a capability tier, `.metallib` is not per-family |

`gfx()`, `sm()`, `xe()`, and `apple()` pull out one vendor's form when that's
all you need:

```rust
let amd_targets: Vec<_> = gpu_probe::detect()
    .iter()
    .filter_map(|gpu| gpu.arch_target.and_then(ArchTarget::gfx))
    .collect();
```

`ArchTarget` itself is not ordered — comparing an AMD target to an NVIDIA one is
meaningless — but `GfxTarget` and `ComputeCapability` both order `major` first,
so a minimum requirement compares directly:

```rust
use gpu_probe::{ComputeCapability, GfxTarget};

let ampere_or_newer = ComputeCapability::new(8, 0);
let rdna2_or_newer = GfxTarget::new(10, 3, 0);
```

The AMD form comes from KFD sysfs, published by the `amdgpu` kernel driver —
**no ROCm install is required**, and it is reported on machines that have none.

The Intel and Apple forms are derived rather than queried, because neither
platform publishes an architecture a caller can read: Intel's comes from a PCI
device id table, Apple's from the chip name. Both report `None` for anything
their table does not recognise rather than guessing, and **neither has been
verified against real hardware yet**.

### Host toolchains

Toolchain properties describe the machine rather than any one GPU, so they're
returned separately. They are not the same measurement:

| function | reports | source |
|:---------|:--------|:-------|
| `cuda_host()` | CUDA **driver** version | NVML, kernel-side |
| `rocm_host()` | ROCm **userspace** release | `$ROCM_PATH/.info/version` → `/opt/rocm` |
| `oneapi_host()` | oneAPI **toolkit** release | `$ONEAPI_ROOT/compiler/latest` → `/opt/intel/oneapi` |
| `vulkan_host()` | Vulkan **API** version | loader (`libvulkan.so.1`) + ICD manifests under `/usr/share/vulkan/icd.d` |

Only NVIDIA exposes a driver version. `amdgpu` declares no `MODULE_VERSION` and
KFD publishes only a topology counter, so the AMD and Intel probes can report the
userspace install and nothing more — a property of the drivers, not an omission.
Vulkan is a runtime rather than a toolkit, and — unlike the other three — has no
per-GPU architecture to match: SPIR-V is portable and the driver compiles it at
load time, so `vulkan_host()` carries no `arch_target` counterpart.

```rust
use gpu_probe::{ComputeCapability, RocmVersion};

if let Some(cuda) = gpu_probe::cuda_host() {
    println!("{} / CUDA {}", cuda.compute_capability, cuda.driver_version);
    // 8.6 / CUDA 13.3

    if cuda.compute_capability >= ComputeCapability::new(8, 0) {
        // pick an Ampere-or-newer build
    }
}

if let Some(rocm) = gpu_probe::rocm_host() {
    println!("ROCm {}", rocm.version);  // ROCm 6.2.4

    if rocm.version >= RocmVersion::new(6, 0, 0) {
        // pick a ROCm 6 build
    }
}

if let Some(oneapi) = gpu_probe::oneapi_host() {
    println!("oneAPI {}", oneapi.version);  // oneAPI 2024.2.1
}

if let Some(vulkan) = gpu_probe::vulkan_host() {
    println!("Vulkan {}", vulkan.api_version);  // Vulkan 1.3.280
}
```

`Some` is the signal that the stack is installed and the host can run its
builds. `None` is weaker: it means the stack was not found where this crate
looks — NVML unavailable for `cuda_host()` (no driver, no device, the `nvidia`
feature disabled, or unusable values), or no userspace install at the standard
prefixes for the other two. A distro shipping ROCm into `/usr` rather than
`/opt/rocm`, or a container carrying only the runtime libraries, will report
`None` despite working. Treat `Some` as proof and `None` as "probably not,
worth confirming".

What `None` does **not** tell you is that the GPU is unusable. The kernel and
userspace halves are independent: `arch_target` comes from the `amdgpu` driver
and is reported with no ROCm installed at all.

`cuda_host().compute_capability` is device 0's — the same value that GPU reports
as `ArchTarget::Sm` in its own `arch_target`.

### Is a device ready for a model?

Readiness is four separate questions, and the pieces above answer each one:

```rust
use gpu_probe::{ArchTarget, GfxTarget};

let need = 16 * 1024 * 1024 * 1024; // 16 GiB
let built_for = GfxTarget::new(10, 1, 3); // this artifact is gfx1013

// The runtime has to be installed to execute anything.
let runtime_ready = gpu_probe::rocm_host().is_some();

let device_ready = gpu_probe::detect().iter().any(|gpu| {
    // The kernel driver has to expose the GPU for compute — on AMD, a
    // target at all means KFD is live.
    gpu.arch_target == Some(ArchTarget::Gfx(built_for))
        // And the weights have to fit.
        && gpu.free_bytes.unwrap_or(gpu.total_bytes) >= need
});

if runtime_ready && device_ready {
    // load the model
}
```

For NVIDIA the runtime check is `cuda_host().is_some()`, which tests the
*driver* — the usual thing to verify, since frameworks like PyTorch ship their
own CUDA runtime. For Intel, `oneapi_host()` tests for the toolkit and does not
see a distro-packaged Level Zero runtime, so it is the least complete of the
three.

## Notes

- `total_bytes` is dedicated VRAM on discrete GPUs. On integrated/unified GPUs (Intel iGPUs, AMD APUs, Apple Silicon) it's the shared system-memory ceiling, and `free_bytes` / `used_bytes` are usually `None`.
- AMD APUs are a special case: their `mem_info_vram_total` is only a BIOS carveout (512 MiB on a BC-250), so `total_bytes` adds the GTT pool they really allocate from — sized by the kernel's `ttm.pages_limit` — and `free_bytes` / `used_bytes` cover both pools. The result matches Vulkan/RADV to the byte; ROCm reports ~512 MiB less, since KFD publishes only the GTT-backed bank.
- AMD GPU names come from the KFD ASIC codename (`AMD cyan_skillfish`), falling back to the DRM node (`AMD GPU (card1)`) when KFD reports none.
- `oneapi_host()` covers the **toolkit**, not the GPU runtime: a host using a distro-packaged Level Zero driver with no toolkit reports `None`, since reading that runtime's version needs linking rather than a file read. Not yet verified against a real install.
- NVIDIA detection reads NVML from the installed driver at runtime — the CUDA toolkit is not required.
- NVML is initialized once per process and intentionally never shut down. Cycling `nvmlInit`/`nvmlShutdown` leaks a file descriptor each time, so `detect()` is safe to poll on a timer: descriptor use is flat, and each call still returns live memory values.

## License

[Apache-2.0](LICENSE)
