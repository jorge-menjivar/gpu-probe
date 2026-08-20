// SPDX-License-Identifier: Apache-2.0
//! Vulkan runtime detection.
//!
//! Vulkan advertises itself through two filesystem facts that need no linking:
//! the loader (`libvulkan.so.1`) and the ICD manifests each installed driver
//! drops under `/usr/share/vulkan/icd.d`. Reading those is enough to answer
//! "can this host run a Vulkan build, and to which API version", which is all
//! a consumer selecting a prebuilt artifact needs.
//!
//! What a manifest yields is that driver's *advertised* version, and `host()`
//! reports the highest across them. That is neither the loader's instance
//! version — the number `vulkaninfo` prints, usually the higher of the two —
//! nor any single device's `apiVersion`; both of those require creating an
//! instance and calling into the loader. Staying on the filesystem is the
//! trade this module makes, and the reason a caller must not read the result
//! as a per-GPU capability.
//!
//! Unlike CUDA and `ROCm` there is no architecture to report: SPIR-V is
//! portable and the driver compiles it at load time, so a Vulkan build that
//! runs anywhere runs everywhere the loader does.

use crate::{VulkanHost, VulkanVersion};
use std::path::Path;

/// Where the loader looks for driver manifests. Every manifest under both
/// directories is read; `host()` takes the highest `api_version` across the
/// union, so this order does not affect which version wins.
const ICD_DIRS: [&str; 2] = ["/usr/local/share/vulkan/icd.d", "/usr/share/vulkan/icd.d"];

/// Loader sonames to probe. Checked with `.any()`, so this is an unordered
/// set, not a preference list.
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

/// Whether the Vulkan loader is installed under any of `dirs`. Presence of the
/// shared object is the signal; nothing is opened or linked.
///
/// Only the sonames a program is linked against count: a bare
/// `libvulkan.so.1.3.280` with no `libvulkan.so.1` beside it is a file the
/// loader could not be found by, so it is not an install.
///
/// Takes its directories rather than reading `LIB_DIRS` directly, the way the
/// `ROCm` probe takes an install root, so the search is testable against a
/// fixture instead of whatever the host happens to have.
fn loader_in<P: AsRef<Path>>(dirs: &[P]) -> bool {
    dirs.iter()
        .flat_map(|dir| LOADER_SONAMES.iter().map(move |so| dir.as_ref().join(so)))
        .any(|path| path.exists())
}

/// The highest API version any manifest under `dirs` advertises.
///
/// Highest rather than lowest: a host with both a software rasterizer and a
/// real driver can run what the real driver supports, and the consumer is
/// choosing one build for the machine.
///
/// Unreadable directories, non-`.json` files, and manifests that do not parse
/// are skipped rather than failing the probe — a stray file in `icd.d` must
/// not hide a working driver's manifest.
fn highest_api_version<P: AsRef<Path>>(dirs: &[P]) -> Option<VulkanVersion> {
    dirs.iter()
        .filter_map(|dir| std::fs::read_dir(dir).ok())
        .flatten()
        .flatten()
        .filter(|entry| entry.path().extension().is_some_and(|e| e == "json"))
        .filter_map(|entry| std::fs::read_to_string(entry.path()).ok())
        .filter_map(|content| parse_icd_api_version(&content))
        .max()
}

/// The runtime installed under `lib_dirs`/`icd_dirs`, if both halves are there.
///
/// Both are required: the loader alone cannot name a version, and manifests
/// alone describe drivers nothing can dispatch to.
fn host_in<L: AsRef<Path>, I: AsRef<Path>>(lib_dirs: &[L], icd_dirs: &[I]) -> Option<VulkanHost> {
    if !loader_in(lib_dirs) {
        return None;
    }
    let api_version = highest_api_version(icd_dirs)?;
    Some(VulkanHost { api_version })
}

/// Probe the conventional locations.
///
/// Unconditional, like the `ROCm` and `oneAPI` probes: the paths this checks
/// are Linux-specific and simply do not exist on other platforms, so the
/// lookups come back empty there without needing a `cfg`.
pub(crate) fn host() -> Option<VulkanHost> {
    host_in(&LIB_DIRS, &ICD_DIRS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A real `radeon_icd.x86_64.json`, trimmed to the fields read here.
    const RADEON: &str = r#"{"file_format_version":"1.0.0",
                             "ICD":{"library_path":"/usr/lib/libvulkan_radeon.so",
                                    "api_version":"1.3.280"}}"#;

    /// Mesa's software rasterizer, which ships by default on many distros and
    /// is an ICD like any other — the case the `README` warns consumers about.
    const LAVAPIPE: &str = r#"{"file_format_version":"1.0.1",
                               "ICD":{"library_path":"/usr/lib/libvulkan_lvp.so",
                                      "api_version":"1.3.255"}}"#;

    /// A scratch directory that deletes itself on drop.
    ///
    /// Written here rather than pulled in as a dev-dependency: this probe is a
    /// filesystem read, so testing the search means giving it real directories,
    /// and that needs nothing beyond `std`.
    struct TempTree(PathBuf);

    impl TempTree {
        /// A fresh empty directory. The pid plus a counter keeps concurrent
        /// tests — and concurrent `cargo test` runs — off each other's fixtures.
        fn new(label: &str) -> Self {
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            let nth = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "gpu-probe-vulkan-{}-{label}-{nth}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("scratch directory is creatable");
            Self(path)
        }

        /// Drop a file into the tree; chainable, so a fixture reads as a listing.
        fn with(&self, name: &str, content: &str) -> &Self {
            std::fs::write(self.0.join(name), content).expect("fixture file is writable");
            self
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn parses_an_icd_manifest_api_version() {
        let icd = r#"{"file_format_version":"1.0.0",
                      "ICD":{"library_path":"libvulkan_radeon.so",
                             "api_version":"1.3.280"}}"#;
        assert_eq!(
            parse_icd_api_version(icd),
            Some(VulkanVersion::new(1, 3, 280))
        );
        assert_eq!(
            parse_icd_api_version(LAVAPIPE),
            Some(VulkanVersion::new(1, 3, 255))
        );
        // NVIDIA's manifest carries extra top-level keys; unknown fields are
        // ignored rather than rejected.
        let nvidia = r#"{"file_format_version":"1.0.1",
                         "ICD":{"library_path":"libGLX_nvidia.so.0",
                                "api_version":"1.3.277",
                                "is_portability_driver":false}}"#;
        assert_eq!(
            parse_icd_api_version(nvidia),
            Some(VulkanVersion::new(1, 3, 277))
        );
    }

    #[test]
    fn rejects_an_icd_manifest_without_an_api_version() {
        let icd = r#"{"ICD":{"library_path":"libvulkan_radeon.so"}}"#;
        assert_eq!(parse_icd_api_version(icd), None);
        assert_eq!(parse_icd_api_version("not json"), None);
        assert_eq!(parse_icd_api_version(""), None);
        assert_eq!(parse_icd_api_version("{}"), None);
        // The version must be under `ICD`, and must be a string: a JSON number
        // would parse as 1.3 and silently lose the patch.
        assert_eq!(parse_icd_api_version(r#"{"api_version":"1.3.280"}"#), None);
        assert_eq!(
            parse_icd_api_version(r#"{"ICD":{"api_version":1.3}}"#),
            None
        );
        assert_eq!(parse_icd_api_version(r#"{"ICD":"1.3.280"}"#), None);
        assert_eq!(
            parse_icd_api_version(r#"{"ICD":[{"api_version":"1.3.0"}]}"#),
            None
        );
    }

    #[test]
    fn rejects_manifests_whose_version_is_not_a_version() {
        let with = |version: &str| {
            parse_icd_api_version(&format!(r#"{{"ICD":{{"api_version":"{version}"}}}}"#))
        };
        // A two-part version is a legal shape for the shared parser; the spec
        // writes all three, but a truncated one still names an API level.
        assert_eq!(with("1.2"), Some(VulkanVersion::new(1, 2, 0)));
        assert_eq!(with("1"), None, "a bare major is not a version");
        assert_eq!(with(""), None);
        assert_eq!(with("one.three.zero"), None);
        assert_eq!(with("1.3.x"), None);
    }

    #[test]
    fn versions_order_major_first() {
        assert!(VulkanVersion::new(1, 3, 0) > VulkanVersion::new(1, 2, 300));
        assert!(VulkanVersion::new(2, 0, 0) > VulkanVersion::new(1, 9, 9));
    }

    #[test]
    fn loader_is_found_under_either_soname() {
        let versioned = TempTree::new("loader-soname-1");
        versioned.with("libvulkan.so.1", "");
        assert!(loader_in(&[versioned.path()]));

        // The development symlink alone is enough; the set is unordered.
        let unversioned = TempTree::new("loader-soname-dev");
        unversioned.with("libvulkan.so", "");
        assert!(loader_in(&[unversioned.path()]));

        // Any one directory in the list satisfies the search.
        let empty = TempTree::new("loader-empty-first");
        assert!(loader_in(&[empty.path(), versioned.path()]));
    }

    #[test]
    fn a_host_without_the_loader_reports_no_install() {
        let empty = TempTree::new("loader-absent");
        assert!(!loader_in(&[empty.path()]));
        assert!(!loader_in(&[Path::new("/nonexistent-vulkan-libdir")]));
        assert!(!loader_in::<&Path>(&[]));

        // The real file with no soname symlink beside it: nothing can dlopen
        // `libvulkan.so.1` here, so this is not an installed loader.
        let unlinked = TempTree::new("loader-unlinked");
        unlinked.with("libvulkan.so.1.3.280", "");
        assert!(!loader_in(&[unlinked.path()]));
    }

    #[cfg(unix)]
    #[test]
    fn a_dangling_loader_symlink_is_not_an_install() {
        // A package removed without its symlink leaves the name behind. The
        // check follows the link, so the broken one does not count.
        let tree = TempTree::new("loader-dangling");
        std::os::unix::fs::symlink("libvulkan.so.1.3.280", tree.path().join("libvulkan.so.1"))
            .expect("symlink is creatable");
        assert!(!loader_in(&[tree.path()]));

        tree.with("libvulkan.so.1.3.280", "");
        assert!(loader_in(&[tree.path()]), "the same link now resolves");
    }

    #[test]
    fn highest_api_version_wins_across_manifests() {
        // A machine with the software rasterizer installed alongside a real
        // driver: the real driver's level is the one a build can target.
        let icd = TempTree::new("icd-highest");
        icd.with("lvp_icd.x86_64.json", LAVAPIPE)
            .with("radeon_icd.x86_64.json", RADEON);
        assert_eq!(
            highest_api_version(&[icd.path()]),
            Some(VulkanVersion::new(1, 3, 280))
        );

        // The two search directories are a union, not a preference order, so
        // the highest wins whichever side it sits on.
        let local = TempTree::new("icd-local");
        local.with("lvp_icd.x86_64.json", LAVAPIPE);
        let shared = TempTree::new("icd-shared");
        shared.with("radeon_icd.x86_64.json", RADEON);
        assert_eq!(
            highest_api_version(&[local.path(), shared.path()]),
            Some(VulkanVersion::new(1, 3, 280))
        );
        assert_eq!(
            highest_api_version(&[shared.path(), local.path()]),
            Some(VulkanVersion::new(1, 3, 280)),
            "directory order must not change the answer",
        );
    }

    #[test]
    fn only_json_manifests_are_read() {
        // `icd.d` collects editor backups and disabled drivers; the loader
        // reads `.json` and so does this.
        let icd = TempTree::new("icd-extensions");
        icd.with("lvp_icd.x86_64.json", LAVAPIPE)
            .with("radeon_icd.x86_64.json.disabled", RADEON)
            .with("radeon_icd.x86_64.json.bak", RADEON)
            .with("notes.txt", RADEON)
            .with("README", RADEON);
        assert_eq!(
            highest_api_version(&[icd.path()]),
            Some(VulkanVersion::new(1, 3, 255)),
            "only the .json manifest counts, so lavapipe's version stands",
        );
    }

    #[test]
    fn a_malformed_manifest_does_not_hide_a_good_one() {
        let icd = TempTree::new("icd-malformed");
        icd.with("broken_icd.json", "{ not json")
            .with("empty_icd.json", "")
            .with("versionless_icd.json", r#"{"ICD":{"library_path":"x.so"}}"#)
            .with("radeon_icd.x86_64.json", RADEON);
        assert_eq!(
            highest_api_version(&[icd.path()]),
            Some(VulkanVersion::new(1, 3, 280))
        );
    }

    #[test]
    fn missing_or_empty_icd_directories_report_nothing() {
        let empty = TempTree::new("icd-empty");
        assert_eq!(highest_api_version(&[empty.path()]), None);
        assert_eq!(
            highest_api_version(&[Path::new("/nonexistent-vulkan-icd-dir")]),
            None,
            "an absent directory is skipped, not an error",
        );
        assert_eq!(highest_api_version::<&Path>(&[]), None);
    }

    #[test]
    fn both_halves_are_required() {
        let lib = TempTree::new("host-lib");
        lib.with("libvulkan.so.1", "");
        let icd = TempTree::new("host-icd");
        icd.with("radeon_icd.x86_64.json", RADEON);
        let empty = TempTree::new("host-empty");

        assert_eq!(
            host_in(&[lib.path()], &[icd.path()]),
            Some(VulkanHost {
                api_version: VulkanVersion::new(1, 3, 280)
            })
        );
        assert_eq!(
            host_in(&[empty.path()], &[icd.path()]),
            None,
            "manifests describe drivers nothing can dispatch to without a loader",
        );
        assert_eq!(
            host_in(&[lib.path()], &[empty.path()]),
            None,
            "a loader with no ICD advertises no version to report",
        );
    }

    #[test]
    fn host_lookup_never_panics() {
        // Environment-dependent: asserts invariants, not the presence of a driver.
        if let Some(v) = host() {
            assert!(v.api_version.major > 0);
        }
    }
}
