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
    // Read-only probing must be deterministic for a stable host.
    assert_eq!(first, second, "detect() should be stable across calls");
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
