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
use nvml_wrapper::Nvml;
#[cfg(feature = "nvidia")]
use std::sync::OnceLock;

/// The process-wide NVML handle.
///
/// NVML is initialized at most once and **never shut down**. This is
/// deliberate, not an oversight: cycling `nvmlInit`/`nvmlShutdown` permanently
/// costs one file descriptor per cycle (an `eventfd` that shutdown does not
/// return), so a caller polling [`detect`](crate::detect) on a timer exhausts
/// its fd table and can no longer `accept()` connections. Holding one handle is
/// flat across calls, and queries against it still return live values — so
/// `memory_info()` readouts stay current.
///
/// Do not add a shutdown or make this handle droppable.
#[cfg(feature = "nvidia")]
static NVML: OnceLock<Nvml> = OnceLock::new();

/// The shared NVML handle, initializing it on first use.
///
/// Only *success* is cached. A failed init allocates no file descriptors, so
/// retrying costs nothing but a failed `dlopen`; caching the failure instead
/// would mean a host whose driver loads after the first call — or a daemon that
/// starts before the driver is up — reports "no NVIDIA GPU" until it restarts.
#[cfg(feature = "nvidia")]
fn nvml() -> Option<&'static Nvml> {
    if let Some(nvml) = NVML.get() {
        return Some(nvml);
    }
    let nvml = Nvml::init().ok()?;
    // A concurrent first call may have won the race, in which case our handle
    // is dropped here and theirs is returned. `nvmlInit`/`nvmlShutdown` are
    // reference counted, so the winner's handle stays valid. This costs one fd,
    // once, and only when two threads make the very first call simultaneously.
    Some(NVML.get_or_init(|| nvml))
}

#[cfg(feature = "nvidia")]
pub(crate) fn detect() -> Vec<GpuInfo> {
    let Some(nvml) = nvml() else {
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
