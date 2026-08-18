// SPDX-License-Identifier: Apache-2.0
//! AMD compute-topology detection via KFD sysfs
//! (`/sys/class/kfd/kfd/topology/nodes`). No `ROCm` install required — the
//! `amdgpu` kernel driver publishes this tree, and it is world-readable.
//!
//! Each node is either the host CPU (`gfx_target_version 0`, skipped here) or a
//! GPU exposing the `gfx` target its `ROCm`/HIP code objects must be built for.
//! `drm_render_minor` ties a node back to the DRM card the `drm` module found,
//! which is how a [`crate::GfxTarget`] reaches the right [`crate::GpuInfo`].

use crate::GfxTarget;

/// Root of the KFD topology tree.
#[allow(dead_code)] // used on Linux; unused on other targets
const TOPOLOGY: &str = "/sys/class/kfd/kfd/topology/nodes";

/// One GPU node from the KFD topology.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ComputeNode {
    /// Minor number of the node's render device (`128` for `renderD128`).
    pub render_minor: u32,
    /// Kernel codename for the ASIC, e.g. `cyan_skillfish`, when non-empty.
    pub name: Option<String>,
    /// Architecture the node's code objects target.
    pub gfx_target: GfxTarget,
}

/// Look up one `key value` pair in a KFD `properties` file.
#[allow(dead_code)] // used on Linux + in tests; unused on other targets
fn property(content: &str, key: &str) -> Option<u64> {
    for line in content.lines() {
        let mut parts = line.split_whitespace();
        if parts.next() == Some(key) {
            return parts.next()?.parse().ok();
        }
    }
    None
}

/// Decode `gfx_target_version` — `major * 10000 + minor * 100 + step`, so
/// `100103` is `gfx1013`. Returns `None` for `0`, which marks a CPU node.
#[allow(dead_code)] // used on Linux + in tests; unused on other targets
fn parse_gfx_target(version: u64) -> Option<GfxTarget> {
    if version == 0 {
        return None;
    }
    let field = |v: u64| u32::try_from(v).ok();
    Some(GfxTarget::new(
        field(version / 10_000)?,
        field(version / 100 % 100)?,
        field(version % 100)?,
    ))
}

/// Every GPU node the KFD driver reports, in directory order. Empty when the
/// tree is absent (no `amdgpu`, or a kernel built without KFD).
#[cfg(target_os = "linux")]
pub(crate) fn nodes() -> Vec<ComputeNode> {
    let mut nodes = Vec::new();
    let Ok(entries) = std::fs::read_dir(TOPOLOGY) else {
        return nodes;
    };
    for entry in entries.flatten() {
        let dir = entry.path();
        let Ok(properties) = std::fs::read_to_string(dir.join("properties")) else {
            continue;
        };
        // A CPU node has no gfx target; skipping it also skips its render minor
        // of 0, which would otherwise collide with a real card.
        let Some(gfx_target) =
            property(&properties, "gfx_target_version").and_then(parse_gfx_target)
        else {
            continue;
        };
        let Some(render_minor) = property(&properties, "drm_render_minor")
            .and_then(|m| u32::try_from(m).ok())
            .filter(|&m| m > 0)
        else {
            continue;
        };
        nodes.push(ComputeNode {
            render_minor,
            name: std::fs::read_to_string(dir.join("name"))
                .ok()
                .map(|n| n.trim().to_string())
                .filter(|n| !n.is_empty()),
            gfx_target,
        });
    }
    nodes
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn nodes() -> Vec<ComputeNode> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trimmed from a real BC-250 node.
    const NODE: &str = "cpu_cores_count 0\nsimd_count 48\ngfx_target_version 100103\n\
                        vendor_id 4098\ndevice_id 5118\ndrm_render_minor 128\nnum_xcc 1\n";

    #[test]
    fn reads_properties_by_key() {
        assert_eq!(property(NODE, "gfx_target_version"), Some(100_103));
        assert_eq!(property(NODE, "drm_render_minor"), Some(128));
        assert_eq!(property(NODE, "simd_count"), Some(48));
        assert_eq!(property(NODE, "absent"), None);
    }

    #[test]
    fn property_matches_whole_key_only() {
        // A prefix must not match: `simd_count` is not `simd_count_base`.
        assert_eq!(property("simd_count_base 7\n", "simd_count"), None);
        assert_eq!(property("simd 1\nsimd_count 48\n", "simd_count"), Some(48));
    }

    #[test]
    fn property_tolerates_malformed_lines() {
        assert_eq!(property("gfx_target_version\n", "gfx_target_version"), None);
        assert_eq!(
            property("gfx_target_version x\n", "gfx_target_version"),
            None
        );
        assert_eq!(property("", "gfx_target_version"), None);
    }

    #[test]
    fn decodes_gfx_target_versions() {
        // The BC-250 this was written against.
        assert_eq!(parse_gfx_target(100_103), Some(GfxTarget::new(10, 1, 3)));
        assert_eq!(parse_gfx_target(100_300), Some(GfxTarget::new(10, 3, 0)));
        // MI200: step 10, which renders as the `a` in `gfx90a`.
        assert_eq!(parse_gfx_target(90_010), Some(GfxTarget::new(9, 0, 10)));
        assert_eq!(parse_gfx_target(110_000), Some(GfxTarget::new(11, 0, 0)));
    }

    #[test]
    fn cpu_nodes_have_no_gfx_target() {
        assert_eq!(parse_gfx_target(0), None);
    }

    #[test]
    fn node_listing_never_panics() {
        // Environment-dependent: asserts invariants, not the presence of a GPU.
        for node in nodes() {
            assert!(node.render_minor > 0);
            assert!(node.gfx_target.major > 0);
            assert!(node.name.as_ref().is_none_or(|n| !n.is_empty()));
        }
    }
}
