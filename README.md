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
    pub gfx_target: Option<GfxTarget>,                  // AMD:    gfx1013
    pub compute_capability: Option<ComputeCapability>,  // NVIDIA: sm_86
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

Which prebuilt artifact a GPU can run is reported per device. The two fields are
counterparts — a GPU carries whichever one its vendor defines, never both:

```rust
for gpu in gpu_probe::detect() {
    match (gpu.gfx_target, gpu.compute_capability) {
        // AMD: the ROCm/HIP `--offload-arch` value, e.g. gfx1013
        (Some(gfx), _) => println!("build {gfx}"),
        // NVIDIA: the CUDA `sm_` value, e.g. sm_86
        (_, Some(sm)) => println!("build sm_{}{}", sm.major, sm.minor),
        (None, None) => {}
    }
}
```

Both order `major` first, so comparing against a minimum works directly:

```rust
use gpu_probe::{ComputeCapability, GfxTarget};

let ampere_or_newer = ComputeCapability::new(8, 0);
let rdna2_or_newer = GfxTarget::new(10, 3, 0);
```

`gfx_target` comes from KFD sysfs, published by the `amdgpu` kernel driver — **no
ROCm install is required**, and it is reported on machines that have none.

### Host toolchains

Toolchain properties describe the machine rather than any one GPU, so they're
returned separately. They are not the same measurement:

| function | reports | source |
|:---------|:--------|:-------|
| `cuda_host()` | CUDA **driver** version | NVML, kernel-side |
| `rocm_host()` | ROCm **userspace** release | `$ROCM_PATH/.info/version` → `/opt/rocm` |
| `oneapi_host()` | oneAPI **toolkit** release | `$ONEAPI_ROOT/compiler/latest` → `/opt/intel/oneapi` |

Only NVIDIA exposes a driver version. `amdgpu` declares no `MODULE_VERSION` and
KFD publishes only a topology counter, so the AMD and Intel probes can report the
userspace install and nothing more — a property of the drivers, not an omission.

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
```

`None` means that stack is absent — for `cuda_host()`, NVML is unavailable (no
driver, no device, the `nvidia` feature disabled, or unusable values); for the
other two, the userspace install is missing. **Absence says nothing about
whether the GPU works for compute**: `gfx_target` comes from the kernel driver
and is reported with no ROCm installed at all.

`cuda_host().compute_capability` is device 0's — the same value that GPU reports
in its own `compute_capability` field.

## Notes

- `total_bytes` is dedicated VRAM on discrete GPUs. On integrated/unified GPUs (Intel iGPUs, AMD APUs, Apple Silicon) it's the shared system-memory ceiling, and `free_bytes` / `used_bytes` are usually `None`.
- AMD APUs are a special case: their `mem_info_vram_total` is only a BIOS carveout (512 MiB on a BC-250), so `total_bytes` adds the GTT pool they really allocate from — sized by the kernel's `ttm.pages_limit` — and `free_bytes` / `used_bytes` cover both pools.
- AMD GPU names come from the KFD ASIC codename (`AMD cyan_skillfish`), falling back to the DRM node (`AMD GPU (card1)`) when KFD reports none.
- `oneapi_host()` covers the **toolkit**, not the GPU runtime: a host using a distro-packaged Level Zero driver with no toolkit reports `None`, since reading that runtime's version needs linking rather than a file read. Not yet verified against a real install.
- NVIDIA detection reads NVML from the installed driver at runtime — the CUDA toolkit is not required.
- NVML is initialized once per process and intentionally never shut down. Cycling `nvmlInit`/`nvmlShutdown` leaks a file descriptor each time, so `detect()` is safe to poll on a timer: descriptor use is flat, and each call still returns live memory values.

## License

[Apache-2.0](LICENSE)
