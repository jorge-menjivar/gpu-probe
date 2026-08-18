// SPDX-License-Identifier: Apache-2.0
//! Development-only viewer: prints everything [`gpu_probe`] reports, with a
//! used/free bar per GPU. Built for `watch`, so it prints once and exits.
//!
//! ```text
//! cargo build --example watch
//! watch -n1 target/debug/examples/watch
//! ```
//!
//! All arithmetic here is integer-only — a dev tool has no business pulling in
//! float formatting lints, and tenths of a GiB are plenty of precision.

/// Width of the memory bar, in characters.
const BAR_WIDTH: usize = 40;

const GIB: u64 = 1024 * 1024 * 1024;

/// Format a byte count as `14.5 GiB`, rounding down to tenths.
fn gib(bytes: u64) -> String {
    let tenths = bytes / (GIB / 10);
    format!("{}.{} GiB", tenths / 10, tenths % 10)
}

/// Render `used` of `total` as `[████░░░░] 8.1%`, or an empty bar when the
/// split is unknown (integrated GPUs that report no used/free).
fn bar(used: Option<u64>, total: u64) -> String {
    let Some(used) = used.filter(|_| total > 0) else {
        return format!("[{}] usage unknown", "·".repeat(BAR_WIDTH));
    };
    let used = used.min(total);
    // u128 so the scaling multiply can't overflow on absurd inputs.
    let scaled = u128::from(used) * BAR_WIDTH as u128 / u128::from(total);
    let filled = usize::try_from(scaled).unwrap_or(BAR_WIDTH).min(BAR_WIDTH);
    let tenths = u128::from(used) * 1000 / u128::from(total);
    format!(
        "[{}{}] {}.{}%",
        "█".repeat(filled),
        "░".repeat(BAR_WIDTH - filled),
        tenths / 10,
        tenths % 10,
    )
}

fn main() {
    let gpus = gpu_probe::detect();

    println!("gpu-probe · {} GPU(s)", gpus.len());
    println!();

    if gpus.is_empty() {
        println!("  no GPUs detected");
    }

    for (index, gpu) in gpus.iter().enumerate() {
        println!("[{index}] {} · {}", gpu.name, gpu.vendor);
        // The artifact-selection target, whichever form this vendor reports:
        // `gfx1013` for ROCm/HIP, `sm_89` for CUDA.
        match gpu.arch_target {
            Some(arch) => println!("     arch   {:>10}", arch.to_string()),
            None => println!("     arch      unavailable"),
        }
        println!("     total  {:>10}", gib(gpu.total_bytes));
        match gpu.used_bytes {
            Some(used) => println!("     used   {:>10}", gib(used)),
            None => println!("     used      unknown"),
        }
        match gpu.free_bytes {
            Some(free) => println!("     free   {:>10}", gib(free)),
            None => println!("     free      unknown"),
        }
        println!("     {}", bar(gpu.used_bytes, gpu.total_bytes));
        println!();
    }

    // Host-wide toolchains, unlike the per-GPU fields above. Both rows always
    // print, including when absent: "we looked and found nothing" is the point
    // of a probe tool, and a lone row for one vendor reads like it describes
    // the GPU above it rather than the host.
    //
    // They are not the same measurement. `cuda` is the driver version from
    // NVML; `rocm` and `oneapi` are userspace installs, because neither AMD nor
    // Intel exposes a driver version anywhere. The architecture each build
    // targets is the per-GPU `arch` row, which needs none of them installed.
    println!("host");
    match gpu_probe::oneapi_host() {
        Some(oneapi) => println!("     oneapi {:>10}", oneapi.version.to_string()),
        None => println!("     oneapi    unavailable"),
    }
    match gpu_probe::rocm_host() {
        Some(rocm) => println!("     rocm   {:>10}", rocm.version.to_string()),
        None => println!("     rocm      unavailable"),
    }
    match gpu_probe::cuda_host() {
        Some(cuda) => println!("     cuda   {:>10}", cuda.driver_version.to_string()),
        None => println!("     cuda      unavailable"),
    }
}
