// SPDX-License-Identifier: Apache-2.0
//! NVIDIA detection via NVML (`libnvidia-ml`), loaded at runtime from the
//! installed driver. The CUDA toolkit is **not** required and nothing links at
//! build time — `nvml-wrapper` `dlopen`s the library lazily, so a host without
//! an NVIDIA driver simply yields no GPUs.
//!
//! Gated behind the default `nvidia` feature; build with
//! `default-features = false` to drop the `nvml-wrapper` dependency entirely.

#[cfg(feature = "nvidia")]
use crate::{GpuInfo, Vendor};

#[cfg(feature = "nvidia")]
pub(crate) fn detect() -> Vec<GpuInfo> {
    let Ok(nvml) = nvml_wrapper::Nvml::init() else {
        return Vec::new();
    };
    let Ok(count) = nvml.device_count() else {
        return Vec::new();
    };
    let mut gpus = Vec::new();
    for index in 0..count {
        let Ok(device) = nvml.device_by_index(index) else {
            continue;
        };
        let Ok(memory) = device.memory_info() else {
            continue;
        };
        gpus.push(GpuInfo {
            name: device.name().unwrap_or_else(|_| "NVIDIA GPU".to_string()),
            vendor: Vendor::Nvidia,
            total_bytes: memory.total,
            free_bytes: Some(memory.free),
            used_bytes: Some(memory.used),
        });
    }
    gpus
}

#[cfg(not(feature = "nvidia"))]
pub(crate) fn detect() -> Vec<crate::GpuInfo> {
    Vec::new()
}
