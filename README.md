<!-- SPDX-License-Identifier: Apache-2.0 -->
# gpu-probe

[![crates.io](https://img.shields.io/crates/v/gpu-probe.svg)](https://crates.io/crates/gpu-probe)
[![docs.rs](https://img.shields.io/docsrs/gpu-probe)](https://docs.rs/gpu-probe)
[![coverage](https://img.shields.io/endpoint?url=https://jorge-menjivar.github.io/gpu-probe/coverage.json)](https://jorge-menjivar.github.io/gpu-probe/)

Cross-platform GPU memory (VRAM) detection for Rust — no vendor SDKs, nothing to install beyond your GPU driver.

| Vendor | Linux | Windows | macOS | Backend |
|:-------|:-----:|:-------:|:-----:|:--------|
| NVIDIA | ✅ | ✅ | ✅<sup>†</sup> | NVML · `system_profiler` |
| AMD    | ✅ | — | ✅<sup>†</sup> | DRM sysfs · `system_profiler` |
| Intel  | ✅ | — | ✅<sup>†</sup> | DRM sysfs · `system_profiler` |
| Apple  | — | — | ✅ | `system_profiler` + `sysctl` |

<sup>†</sup> Intel Macs only — discrete and integrated GPUs are read from `system_profiler`.

Best-effort: you get an empty list on unsupported platforms, never an error.

**Note:** So far this crate has only been tested on NVIDIA hardware. The AMD, Intel, and Apple paths are implemented but not yet verified on real devices — if something doesn't work, please [open an issue](https://github.com/jorge-menjivar/gpu-probe/issues). Help from the community confirming detection on AMD/Intel/Apple GPUs is very much appreciated.

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
    // NVIDIA GeForce RTX 3090 (NVIDIA): 24.0 GiB total, 9.8 GiB free
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

### CUDA host properties

Compute capability and CUDA driver version describe the host and its driver
rather than any one GPU, so they're returned separately — handy for selecting a
prebuilt artifact that matches the machine:

```rust
use gpu_probe::ComputeCapability;

if let Some(cuda) = gpu_probe::cuda_host() {
    println!("{} / CUDA {}", cuda.compute_capability, cuda.driver_version);
    // 8.6 / CUDA 13.3

    if cuda.compute_capability >= ComputeCapability::new(8, 0) {
        // pick an Ampere-or-newer build
    }
}
```

Both are `major`/`minor` pairs ordered major-first, so comparing against a
minimum requirement works directly. `None` means NVML is unavailable — no
NVIDIA driver, no device, the `nvidia` feature disabled, or a driver reporting
unusable values.

## Notes

- `total_bytes` is dedicated VRAM on discrete GPUs. On integrated/unified GPUs (Intel iGPUs, AMD APUs, Apple Silicon) it's the shared system-memory ceiling, and `free_bytes` / `used_bytes` are usually `None`.
- NVIDIA detection reads NVML from the installed driver at runtime — the CUDA toolkit is not required.
- NVML is initialized once per process and intentionally never shut down. Cycling `nvmlInit`/`nvmlShutdown` leaks a file descriptor each time, so `detect()` is safe to poll on a timer: descriptor use is flat, and each call still returns live memory values.

## License

[Apache-2.0](LICENSE)
