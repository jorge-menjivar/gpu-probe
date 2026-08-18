// SPDX-License-Identifier: Apache-2.0
//! Intel `oneAPI` detection.
//!
//! The Intel counterpart of [`crate::rocm_host`], and the same kind of probe: a
//! userspace install, not a driver. Intel exposes no single release marker the
//! way `ROCm` writes `.info/version`, so the release is taken from the
//! component layout the toolkit installs:
//!
//! ```text
//! /opt/intel/oneapi/compiler/latest -> 2024.2
//! ```
//!
//! `ONEAPI_ROOT` — which `setvars.sh` exports — is honoured first, so a
//! non-default or side-by-side install is picked over `/opt/intel/oneapi`.
//!
//! What this does **not** cover: the Level Zero loader and Intel's
//! compute-runtime, which distros package into `/usr/lib` with no version file.
//! Reading those means `dlopen` plus `zeInit`/`zeDriverGetProperties`, which is
//! a linked runtime call rather than a file read, so it stays out of scope
//! here. A host running GPU compute through a distro-packaged Level Zero with
//! no toolkit installed therefore reports `None`.
//!
//! **Unverified against a real install.** Written without Intel hardware or a
//! `oneAPI` install to test against; the parsing is unit-tested and the layout
//! above is the part to confirm first.

use crate::{OneApiHost, OneApiVersion};
use std::path::{Path, PathBuf};

/// Conventional install prefix, used when `ONEAPI_ROOT` is unset.
const DEFAULT_ROOT: &str = "/opt/intel/oneapi";

/// Component directories whose release the toolkit version is read from, in
/// preference order. `compiler` leads because it is the component a GPU build
/// actually needs; the rest are fallbacks for partial installs.
const COMPONENTS: &[&str] = &["compiler", "mkl", "tbb", "dpl", "ccl"];

/// Parse a version directory name (`2024.2`, `2024.2.1`).
fn parse_version(name: &str) -> Option<OneApiVersion> {
    let (major, minor, patch) = crate::parse_dotted_version(name)?;
    Some(OneApiVersion {
        major,
        minor,
        patch,
    })
}

/// Release installed for one component: the `latest` symlink's target, falling
/// back to the newest versioned directory when no such symlink exists.
fn component_version(component: &Path) -> Option<OneApiVersion> {
    let from_symlink = std::fs::read_link(component.join("latest"))
        .ok()
        .and_then(|target| parse_version(&target.file_name()?.to_string_lossy()));
    if let Some(version) = from_symlink {
        return Some(version);
    }
    // No `latest`: take the highest version present rather than directory
    // order, which is arbitrary.
    let mut versions: Vec<OneApiVersion> = std::fs::read_dir(component)
        .ok()?
        .flatten()
        .filter_map(|entry| parse_version(&entry.file_name().to_string_lossy()))
        .collect();
    versions.sort_unstable();
    versions.pop()
}

/// Read the toolkit release from one install root, if it holds one.
fn version_at(root: &Path) -> Option<OneApiVersion> {
    COMPONENTS
        .iter()
        .find_map(|component| component_version(&root.join(component)))
}

/// Install roots to try, most specific first.
fn roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    // `setvars.sh` exports this; an explicit value selects a side-by-side
    // install, so preferring the default prefix would report the wrong release.
    if let Some(path) = std::env::var_os("ONEAPI_ROOT") {
        roots.push(PathBuf::from(path));
    }
    roots.push(PathBuf::from(DEFAULT_ROOT));
    roots
}

pub(crate) fn host() -> Option<OneApiHost> {
    let version = roots().iter().find_map(|root| version_at(root))?;
    Some(OneApiHost { version })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_release_directory_names() {
        assert_eq!(
            parse_version("2024.2"),
            Some(OneApiVersion::new(2024, 2, 0))
        );
        assert_eq!(
            parse_version("2024.2.1"),
            Some(OneApiVersion::new(2024, 2, 1))
        );
        assert_eq!(
            parse_version("2025.0"),
            Some(OneApiVersion::new(2025, 0, 0))
        );
    }

    #[test]
    fn rejects_non_version_directories() {
        // `latest` itself, and the component subdirectories beside it.
        assert_eq!(parse_version("latest"), None);
        assert_eq!(parse_version("bin"), None);
        assert_eq!(parse_version("2024"), None, "a bare year is not a version");
        assert_eq!(parse_version(""), None);
    }

    #[test]
    fn versions_order_by_release_year_first() {
        assert!(OneApiVersion::new(2025, 0, 0) > OneApiVersion::new(2024, 2, 1));
        assert!(OneApiVersion::new(2024, 2, 1) > OneApiVersion::new(2024, 2, 0));
        assert!(OneApiVersion::new(2024, 10, 0) > OneApiVersion::new(2024, 9, 0));
    }

    #[test]
    fn missing_install_root_reports_nothing() {
        assert_eq!(version_at(Path::new("/nonexistent-oneapi-root")), None);
        assert_eq!(component_version(Path::new("/nonexistent-component")), None);
    }

    #[test]
    fn compiler_is_the_preferred_component() {
        assert_eq!(
            COMPONENTS.first(),
            Some(&"compiler"),
            "a GPU build needs the compiler component, so it names the release",
        );
    }

    #[test]
    fn oneapi_root_is_searched_before_the_default() {
        let roots = roots();
        assert_eq!(
            roots.last().map(PathBuf::as_path),
            Some(Path::new(DEFAULT_ROOT)),
            "the conventional prefix is always the fallback",
        );
        if std::env::var_os("ONEAPI_ROOT").is_some() {
            assert_eq!(roots.len(), 2, "an explicit ONEAPI_ROOT is tried first");
        }
    }

    #[test]
    fn host_lookup_never_panics() {
        // Environment-dependent: most hosts have no oneAPI at all.
        if let Some(oneapi) = host() {
            assert!(oneapi.version.major > 0);
        }
    }
}
