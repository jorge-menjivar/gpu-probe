// SPDX-License-Identifier: Apache-2.0
//! End-to-end checks of the public API. These run against whatever hardware the
//! host actually has, so they assert invariants rather than specific GPUs — a
//! headless CI box with no GPU is a valid, passing environment.

// `GpuInfo` is `#[non_exhaustive]`, so it can't be constructed here; building
// and formatting it is covered by the in-crate unit tests. These integration
// tests consume `detect()` output through the public API instead.

#[test]
fn detect_is_safe_and_repeatable() {
    let first = gpu_probe::detect();
    let second = gpu_probe::detect();

    // Hardware identity is what must be stable across calls. `free_bytes` and
    // `used_bytes` are deliberately excluded: they are live readings, so any
    // process touching the GPU between the two calls changes them, and
    // comparing whole `GpuInfo` values made this test fail whenever the
    // machine was busy. Measured under GPU load: 557 of 20,000 back-to-back
    // pairs differed, versus 0 on an idle GPU — a flake that only shows up on
    // a machine doing real work.
    assert_eq!(
        first.len(),
        second.len(),
        "the set of detected GPUs should be stable",
    );
    for (a, b) in first.iter().zip(&second) {
        assert_eq!(a.name, b.name, "GPU name should be stable");
        assert_eq!(a.vendor, b.vendor, "GPU vendor should be stable");
        assert_eq!(
            a.total_bytes, b.total_bytes,
            "total memory should be stable",
        );
    }
}

#[test]
fn detected_gpus_satisfy_invariants() {
    for gpu in gpu_probe::detect() {
        assert!(!gpu.name.is_empty(), "every GPU must have a name");
        assert!(
            gpu.total_bytes > 0,
            "a detected GPU must report some memory"
        );

        if let Some(free) = gpu.free_bytes {
            assert!(free <= gpu.total_bytes, "free cannot exceed total");
        }
        if let Some(used) = gpu.used_bytes {
            assert!(used <= gpu.total_bytes, "used cannot exceed total");
        }
        // The Display impl must not panic for any detected GPU.
        assert!(!gpu.to_string().is_empty());
    }
}

#[test]
fn gfx_target_is_publicly_usable() {
    // `GfxTarget` is part of the public API and `Copy`/`Ord`/`Display`.
    let target = gpu_probe::GfxTarget::new(10, 1, 3);
    assert_eq!(target.to_string(), "gfx1013");
    assert!(target >= gpu_probe::GfxTarget::new(10, 1, 0));

    // KFD may be absent even on AMD, so a target is never required — but any
    // reported one must be a usable `--offload-arch` value. That it is AMD-only
    // is asserted in `arch_target_matches_its_vendor`.
    for gpu in gpu_probe::detect() {
        if let Some(gfx) = gpu.arch_target.and_then(gpu_probe::ArchTarget::gfx) {
            assert!(gfx.to_string().starts_with("gfx"));
            assert!(gfx.major > 0, "gfx target 0 marks a CPU node, not a GPU");
        }
    }
}

#[test]
fn arch_target_matches_its_vendor() {
    // `ArchTarget` makes "at most one architecture target" structural — it can
    // no longer be constructed with both. What still needs asserting is that
    // the variant lines up with the vendor that reported it.
    for gpu in gpu_probe::detect() {
        match gpu.arch_target {
            Some(target) if target.gfx().is_some() => {
                assert_eq!(gpu.vendor, gpu_probe::Vendor::Amd, "gfx is AMD-only");
            }
            Some(target) if target.sm().is_some() => {
                assert_eq!(
                    gpu.vendor,
                    gpu_probe::Vendor::Nvidia,
                    "compute capability is NVIDIA-only",
                );
            }
            Some(target) if target.xe().is_some() => {
                assert_eq!(gpu.vendor, gpu_probe::Vendor::Intel, "xe is Intel-only");
            }
            Some(target) if target.apple().is_some() => {
                assert_eq!(
                    gpu.vendor,
                    gpu_probe::Vendor::Apple,
                    "a Metal family is Apple-only",
                );
            }
            // A future variant for another vendor lands here.
            Some(_) | None => {}
        }
        // Whatever it is, it must render as something a toolchain accepts.
        if let Some(target) = gpu.arch_target {
            assert!(!target.to_string().is_empty());
        }
    }
}

#[test]
fn rocm_host_is_publicly_usable() {
    // `RocmVersion` is part of the public API and `Copy`/`Ord`/`Display`.
    let version = gpu_probe::RocmVersion::new(6, 2, 4);
    assert_eq!(version.to_string(), "6.2.4");
    assert!(version >= gpu_probe::RocmVersion::new(6, 0, 0));

    // Environment-dependent: most hosts have no ROCm, which is a valid,
    // passing environment. Absence must not imply anything about the GPUs.
    if let Some(rocm) = gpu_probe::rocm_host() {
        assert!(rocm.version.major > 0, "a real install has a major version");
    }
}

#[test]
fn oneapi_host_is_publicly_usable() {
    // `OneApiVersion` is part of the public API and `Copy`/`Ord`/`Display`.
    let version = gpu_probe::OneApiVersion::new(2024, 2, 1);
    assert_eq!(version.to_string(), "2024.2.1");
    assert!(version >= gpu_probe::OneApiVersion::new(2024, 0, 0));

    // Environment-dependent: most hosts have no oneAPI, which is a valid,
    // passing environment.
    if let Some(oneapi) = gpu_probe::oneapi_host() {
        assert!(
            oneapi.version.major > 0,
            "a real install has a release year"
        );
    }
}

#[test]
fn vendor_is_publicly_usable() {
    // `Vendor` is part of the public API and `Copy`/`Display`.
    let v = gpu_probe::Vendor::Apple;
    let copied = v;
    assert_eq!(copied.to_string(), "Apple");
}
