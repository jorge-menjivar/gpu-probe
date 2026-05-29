<!-- SPDX-License-Identifier: Apache-2.0 -->
# gpu-probe

Cross-platform GPU memory (VRAM) detection for Rust with **no vendor SDKs**.

`gpu-probe` reports the GPUs visible on the host and how much memory each has,
using only facilities the OS or driver already ship. There is nothing to install
beyond your normal GPU driver, and nothing links against a vendor SDK at build
time.

## How it detects

| Platform | Vendors | Source | Linked at build? |
|----------|---------|--------|------------------|
| Linux, Windows | NVIDIA | NVML (`libnvidia-ml`) via [`nvml-wrapper`](https://crates.io/crates/nvml-wrapper), `dlopen`ed at runtime | No |
| Linux | AMD, Intel | DRM sysfs under `/sys/class/drm` | No |
| macOS | Apple, AMD, Intel, NVIDIA | `system_profiler SPDisplaysDataType` + `sysctl hw.memsize` | No |

- The CUDA toolkit is **not** required for NVIDIA detection — NVML is loaded
  lazily from the installed driver. A host without an NVIDIA driver simply
  reports no NVIDIA GPUs.
- Detection is **best-effort**: [`detect`] returns an empty `Vec` when no GPU is
  found or the platform is unsupported — never an error.

## Install

```toml
[dependencies]
gpu-probe = "0.1"
```

To drop the NVML dependency entirely (e.g. AMD/Apple-only builds), disable
default features:

```toml
[dependencies]
gpu-probe = { version = "0.1", default-features = false }
```

## Usage

```rust
for gpu in gpu_probe::detect() {
    println!("{gpu}");
}
```

Each result is a `GpuInfo`:

```rust
pub struct GpuInfo {
    pub name: String,            // e.g. "NVIDIA GeForce RTX 4090"
    pub vendor: Vendor,          // Nvidia | Amd | Intel | Apple | Unknown
    pub total_bytes: u64,        // dedicated VRAM, or shared ceiling for iGPUs
    pub free_bytes: Option<u64>, // when the platform reports it
    pub used_bytes: Option<u64>, // when the platform reports it
}
```

### A note on integrated / unified memory

For discrete GPUs, `total_bytes` is dedicated VRAM. For integrated and unified
GPUs (Intel iGPUs, AMD APUs, Apple Silicon) there is no dedicated pool, so
`total_bytes` is the **shared system-memory ceiling** available to the GPU, and
`free_bytes` / `used_bytes` are typically `None`.

## Example

Print every detected GPU:

```sh
cargo run --example detect
```

```
GPU 0: NVIDIA GeForce RTX 4090 (NVIDIA): 24.0 GiB total, 23.1 GiB free
```

## Features

| Feature | Default | Effect |
|---------|---------|--------|
| `nvidia` | yes | NVIDIA support via NVML. Disable to remove the `nvml-wrapper` dependency. |

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
