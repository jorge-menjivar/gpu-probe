// SPDX-License-Identifier: Apache-2.0
//! `ROCm` userspace detection.
//!
//! Unlike the rest of this crate, `ROCm` has no driver-side version to read: KFD
//! sysfs exposes a topology generation counter and nothing else, and `amdgpu`
//! declares no `MODULE_VERSION`. The release version lives in the userspace
//! install instead, in a plain text file the `rocm-core` package writes:
//!
//! ```text
//! /opt/rocm/.info/version   →   6.2.4-123
//! ```
//!
//! So this is a filesystem probe, not a library call — nothing links, and a
//! host without `ROCm` simply reports `None`. `ROCM_PATH` is honoured first,
//! which is how side-by-side installs (`/opt/rocm-6.2.4`) are selected.
//!
//! Note what this does *not* mean: `ROCm` being absent says nothing about
//! whether the GPU works for compute. The kernel side is independent, and is
//! what [`crate::GpuInfo::gfx_target`] reports.

use crate::{RocmHost, RocmVersion};
use std::path::{Path, PathBuf};

/// Conventional install prefix, used when `ROCM_PATH` is unset.
const DEFAULT_ROOT: &str = "/opt/rocm";

/// Path of the version file relative to an install root.
const VERSION_FILE: &str = ".info/version";

/// Parse a `.info/version` file: `major.minor.patch`, followed by a build
/// suffix this ignores (`6.2.4-123` → `6.2.4`).
fn parse_version(content: &str) -> Option<RocmVersion> {
    // The suffix is a package build number, not part of the release.
    let version = content.trim().split(['-', '+']).next()?;
    let mut parts = version.split('.');
    let major = parts.next()?.trim().parse().ok()?;
    let minor = parts.next()?.trim().parse().ok()?;
    // Some builds ship `6.2` with no patch component.
    let patch = match parts.next() {
        Some(patch) => patch.trim().parse().ok()?,
        None => 0,
    };
    Some(RocmVersion {
        major,
        minor,
        patch,
    })
}

/// Read the `ROCm` version from one install root, if it holds one.
fn version_at(root: &Path) -> Option<RocmVersion> {
    std::fs::read_to_string(root.join(VERSION_FILE))
        .ok()
        .and_then(|content| parse_version(&content))
}

/// Install roots to try, most specific first.
fn roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    // An explicit `ROCM_PATH` wins: it is how a side-by-side install is picked,
    // so honouring `/opt/rocm` over it would report the wrong version.
    if let Some(path) = std::env::var_os("ROCM_PATH") {
        roots.push(PathBuf::from(path));
    }
    roots.push(PathBuf::from(DEFAULT_ROOT));
    roots
}

pub(crate) fn host() -> Option<RocmHost> {
    let version = roots().iter().find_map(|root| version_at(root))?;
    Some(RocmHost { version })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_release_with_build_suffix() {
        // The exact form `rocm-core` writes.
        assert_eq!(
            parse_version("6.2.4-123\n"),
            Some(RocmVersion::new(6, 2, 4))
        );
        assert_eq!(parse_version("5.7.1-63"), Some(RocmVersion::new(5, 7, 1)));
        assert_eq!(
            parse_version("  6.0.0-91  "),
            Some(RocmVersion::new(6, 0, 0))
        );
    }

    #[test]
    fn parses_release_without_suffix_or_patch() {
        assert_eq!(parse_version("6.2.4"), Some(RocmVersion::new(6, 2, 4)));
        assert_eq!(parse_version("6.2"), Some(RocmVersion::new(6, 2, 0)));
        assert_eq!(parse_version("6.2+build"), Some(RocmVersion::new(6, 2, 0)));
    }

    #[test]
    fn rejects_malformed_versions() {
        assert_eq!(parse_version(""), None);
        assert_eq!(parse_version("6"), None, "a bare major is not a version");
        assert_eq!(parse_version("six.two.four"), None);
        assert_eq!(parse_version("6.2.x"), None);
        assert_eq!(parse_version("-6.2.4"), None);
    }

    #[test]
    fn versions_order_major_first() {
        assert!(RocmVersion::new(6, 2, 4) > RocmVersion::new(6, 2, 0));
        assert!(RocmVersion::new(6, 0, 0) > RocmVersion::new(5, 7, 1));
        assert!(RocmVersion::new(6, 10, 0) > RocmVersion::new(6, 9, 9));
    }

    #[test]
    fn missing_install_root_reports_nothing() {
        assert_eq!(version_at(Path::new("/nonexistent-rocm-root")), None);
    }

    #[test]
    fn rocm_path_is_searched_before_the_default() {
        let roots = roots();
        assert_eq!(
            roots.last().map(PathBuf::as_path),
            Some(Path::new(DEFAULT_ROOT)),
            "the conventional prefix is always the fallback",
        );
        if std::env::var_os("ROCM_PATH").is_some() {
            assert_eq!(roots.len(), 2, "an explicit ROCM_PATH is tried first");
        }
    }

    #[test]
    fn host_lookup_never_panics() {
        // Environment-dependent: most hosts have no ROCm at all.
        if let Some(rocm) = host() {
            assert!(rocm.version.major > 0);
        }
    }
}
