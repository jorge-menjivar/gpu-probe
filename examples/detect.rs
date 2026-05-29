// SPDX-License-Identifier: Apache-2.0
//! Print every GPU detected on this host: `cargo run --example detect`.

fn main() {
    let gpus = gpu_probe::detect();
    if gpus.is_empty() {
        println!("No GPUs detected.");
        return;
    }
    for (index, gpu) in gpus.iter().enumerate() {
        println!("GPU {index}: {gpu}");
    }
}
