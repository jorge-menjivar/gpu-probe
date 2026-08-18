// SPDX-License-Identifier: Apache-2.0
//! Vulkan runtime detection.
//!
//! Vulkan advertises itself through two filesystem facts that need no linking:
//! the loader (`libvulkan.so.1`) and the ICD manifests each installed driver
//! drops under `/usr/share/vulkan/icd.d`. Reading those is enough to answer
//! "can this host run a Vulkan build, and to which API version", which is all
//! a consumer selecting a prebuilt artifact needs.
//!
//! Unlike CUDA and `ROCm` there is no architecture to report: SPIR-V is
//! portable and the driver compiles it at load time, so a Vulkan build that
//! runs anywhere runs everywhere the loader does.

use crate::{VulkanHost, VulkanVersion};
use std::path::Path;

/// Where the loader looks for driver manifests. `/usr/local` first so a
/// locally installed driver wins, matching the loader's own precedence.
const ICD_DIRS: [&str; 2] = ["/usr/local/share/vulkan/icd.d", "/usr/share/vulkan/icd.d"];

/// Loader sonames to probe, newest ABI first.
const LOADER_SONAMES: [&str; 2] = ["libvulkan.so.1", "libvulkan.so"];

/// Directories the loader is normally installed into.
const LIB_DIRS: [&str; 4] = [
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib64",
    "/usr/lib",
    "/usr/local/lib",
];

/// Pull `ICD.api_version` out of a driver manifest.
///
/// The manifest is small, fixed-shape JSON, so this reads the one field it
/// needs rather than modelling the whole document.
fn parse_icd_api_version(content: &str) -> Option<VulkanVersion> {
    let value: serde_json::Value = serde_json::from_str(content).ok()?;
    let text = value.get("ICD")?.get("api_version")?.as_str()?;
    let (major, minor, patch) = crate::parse_dotted_version(text)?;
    Some(VulkanVersion {
        major,
        minor,
        patch,
    })
}

/// Whether the Vulkan loader is installed. Presence of the shared object is
/// the signal; nothing is opened or linked.
#[cfg(target_os = "linux")]
fn loader_present() -> bool {
    LIB_DIRS
        .iter()
        .flat_map(|dir| LOADER_SONAMES.iter().map(move |so| Path::new(dir).join(so)))
        .any(|path| path.exists())
}

#[cfg(not(target_os = "linux"))]
fn loader_present() -> bool {
    false
}

/// The highest API version any installed ICD advertises.
///
/// Highest rather than lowest: a host with both a software rasterizer and a
/// real driver can run what the real driver supports, and the consumer is
/// choosing one build for the machine.
#[cfg(target_os = "linux")]
pub(crate) fn host() -> Option<VulkanHost> {
    if !loader_present() {
        return None;
    }
    let api_version = ICD_DIRS
        .iter()
        .filter_map(|dir| std::fs::read_dir(dir).ok())
        .flatten()
        .flatten()
        .filter(|entry| entry.path().extension().is_some_and(|e| e == "json"))
        .filter_map(|entry| std::fs::read_to_string(entry.path()).ok())
        .filter_map(|content| parse_icd_api_version(&content))
        .max()?;
    Some(VulkanHost { api_version })
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn host() -> Option<VulkanHost> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_an_icd_manifest_api_version() {
        let icd = r#"{"file_format_version":"1.0.0",
                      "ICD":{"library_path":"libvulkan_radeon.so",
                             "api_version":"1.3.280"}}"#;
        assert_eq!(
            parse_icd_api_version(icd),
            Some(VulkanVersion::new(1, 3, 280))
        );
    }

    #[test]
    fn rejects_an_icd_manifest_without_an_api_version() {
        let icd = r#"{"ICD":{"library_path":"libvulkan_radeon.so"}}"#;
        assert_eq!(parse_icd_api_version(icd), None);
        assert_eq!(parse_icd_api_version("not json"), None);
    }

    #[test]
    fn versions_order_major_first() {
        assert!(VulkanVersion::new(1, 3, 0) > VulkanVersion::new(1, 2, 300));
        assert!(VulkanVersion::new(2, 0, 0) > VulkanVersion::new(1, 9, 9));
    }

    #[test]
    fn host_lookup_never_panics() {
        // Environment-dependent: asserts invariants, not the presence of a driver.
        if let Some(v) = host() {
            assert!(v.api_version.major > 0);
        }
    }
}
