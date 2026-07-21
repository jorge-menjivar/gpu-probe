// SPDX-License-Identifier: Apache-2.0
//! Regression test for the file-descriptor leak fixed in `nvidia::detect`.
//!
//! `detect()` used to call `Nvml::init()` per invocation, and cycling
//! `nvmlInit`/`nvmlShutdown` never returns every descriptor init allocated —
//! one fd per call, which exhausted the fd table of any caller polling on a
//! timer. Lives in its own file, as a single test, so it gets a dedicated
//! process: fd counts are process-wide, so a concurrently running test holding
//! even a transient descriptor would show up as growth here.

/// Calls per probe — enough that a per-call leak is unmistakable.
#[cfg(target_os = "linux")]
const CALLS: usize = 200;

/// Descriptors currently open for this process. Linux-only (`/proc`).
#[cfg(target_os = "linux")]
fn open_fds() -> usize {
    std::fs::read_dir("/proc/self/fd").map_or(0, Iterator::count)
}

/// Assert `probe` opens no descriptors it doesn't close.
#[cfg(target_os = "linux")]
fn assert_flat(label: &str, mut probe: impl FnMut()) {
    // Warm-up: the first calls initialize the shared NVML handle and open the
    // descriptors it holds for the process lifetime. Those are the fixed cost
    // the fix trades for; the leak is what happens *after* it.
    for _ in 0..10 {
        probe();
    }

    let before = open_fds();
    for _ in 0..CALLS {
        probe();
    }
    let after = open_fds();

    // The line must be flat, not merely flatter: at the old rate of one fd per
    // call this is +200. The small tolerance absorbs unrelated runtime noise
    // without letting a per-call leak through.
    let growth = after.saturating_sub(before);
    assert!(
        growth <= 2,
        "{label} leaked {growth} fds over {CALLS} calls ({before} -> {after})",
    );
}

#[cfg(target_os = "linux")]
#[test]
fn probing_does_not_leak_file_descriptors() {
    assert_flat("detect()", || {
        let _ = gpu_probe::detect();
    });
    // `cuda_host()` shares the same handle, so polling it must be flat too.
    assert_flat("cuda_host()", || {
        let _ = gpu_probe::cuda_host();
    });
}
