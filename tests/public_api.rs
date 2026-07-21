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
fn vendor_is_publicly_usable() {
    // `Vendor` is part of the public API and `Copy`/`Display`.
    let v = gpu_probe::Vendor::Apple;
    let copied = v;
    assert_eq!(copied.to_string(), "Apple");
}
