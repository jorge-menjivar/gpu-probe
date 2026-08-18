// SPDX-License-Identifier: Apache-2.0
//! Intel GPU architecture identification from the PCI device id.
//!
//! Neither `i915` nor `xe` publishes an architecture anywhere a caller can read
//! it — there is no Intel equivalent of AMD's KFD topology, and the canonical
//! answer (`ze_device_ip_version_ext_t`) requires linking Level Zero. What is
//! readable is the PCI device id in `/sys/class/drm/card*/device/device`, which
//! identifies the family.
//!
//! The mapping is therefore data rather than mechanism, and it is the part that
//! can be wrong. It is deliberately coarse and deliberately incomplete:
//!
//! - Ids are matched by **range**, one per architecture, not per SKU. Getting
//!   the family right for a whole generation is far more tractable than getting
//!   every product id right, and the family is what callers act on.
//! - Anything unrecognised reports `None`. Pre-Xe integrated graphics (Gen9
//!   through Gen11 — Skylake to Ice Lake) are not mapped at all. Reporting
//!   nothing is the honest answer for an id this table has never seen; a wrong
//!   architecture would be worse than an absent one, because a caller would act
//!   on it.
//!
//! **Unverified against real hardware.** Written without an Intel GPU to test
//! against. The ranges below are the assumption to confirm first: read
//! `/sys/class/drm/card*/device/device` on a real machine and check it against
//! what `ocloc -device` or Level Zero reports for the same part.

use crate::IntelArch;

/// PCI device id ranges, one per architecture family. Inclusive on both ends.
///
/// Ranges rather than exact ids so an unlisted SKU within a known generation
/// still resolves, which is the common case as new parts ship.
const FAMILIES: &[(u16, u16, IntelArch)] = &[
    // DG1 — the first discrete Xe part.
    (0x4905, 0x4909, IntelArch::XeLp),
    // Alder Lake and Raptor Lake integrated.
    (0x4600, 0x46FF, IntelArch::XeLp),
    // Rocket Lake integrated.
    (0x4C80, 0x4CFF, IntelArch::XeLp),
    // Tiger Lake integrated.
    (0x9A40, 0x9AFF, IntelArch::XeLp),
    // Raptor Lake refresh integrated.
    (0xA700, 0xA7FF, IntelArch::XeLp),
    // DG2 — Arc A-series (Alchemist).
    (0x5690, 0x56CF, IntelArch::XeHpg),
    // Ponte Vecchio — Data Center GPU Max.
    (0x0BD0, 0x0BDF, IntelArch::XeHpc),
    // Meteor Lake and Arrow Lake integrated.
    (0x7D00, 0x7DFF, IntelArch::XeLpg),
    // Lunar Lake integrated.
    (0x6400, 0x64FF, IntelArch::Xe2),
    // Battlemage — Arc B-series.
    (0xE200, 0xE2FF, IntelArch::Xe2),
];

/// Architecture family for a PCI device id, or `None` when the id is outside
/// every known range.
#[allow(dead_code)] // used on Linux + in tests; unused on other targets
pub(crate) fn arch_for_device_id(id: u16) -> Option<IntelArch> {
    FAMILIES
        .iter()
        .find(|(first, last, _)| (*first..=*last).contains(&id))
        .map(|&(_, _, arch)| arch)
}

/// Parse a sysfs PCI id file (`0x56a0`) into its numeric value.
#[allow(dead_code)] // used on Linux + in tests; unused on other targets
pub(crate) fn parse_device_id(content: &str) -> Option<u16> {
    let text = content.trim();
    let digits = text
        .strip_prefix("0x")
        .or_else(|| text.strip_prefix("0X"))?;
    u16::from_str_radix(digits, 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sysfs_pci_ids() {
        assert_eq!(parse_device_id("0x56a0\n"), Some(0x56A0));
        assert_eq!(parse_device_id("0X9A49"), Some(0x9A49));
        assert_eq!(parse_device_id("  0xe20b "), Some(0xE20B));
    }

    #[test]
    fn rejects_ids_without_a_hex_prefix() {
        // The bare form would be ambiguous with decimal, so it is not accepted.
        assert_eq!(parse_device_id("56a0"), None);
        assert_eq!(parse_device_id("0xzzzz"), None);
        assert_eq!(parse_device_id("0x1ffff"), None, "wider than u16");
        assert_eq!(parse_device_id(""), None);
    }

    #[test]
    fn maps_discrete_parts_to_their_family() {
        // A770 / A750, and a Battlemage B580.
        assert_eq!(arch_for_device_id(0x56A0), Some(IntelArch::XeHpg));
        assert_eq!(arch_for_device_id(0xE20B), Some(IntelArch::Xe2));
        // DG1 and Ponte Vecchio.
        assert_eq!(arch_for_device_id(0x4905), Some(IntelArch::XeLp));
        assert_eq!(arch_for_device_id(0x0BD5), Some(IntelArch::XeHpc));
    }

    #[test]
    fn maps_integrated_parts_to_their_family() {
        assert_eq!(
            arch_for_device_id(0x9A49),
            Some(IntelArch::XeLp),
            "Tiger Lake"
        );
        assert_eq!(
            arch_for_device_id(0x4680),
            Some(IntelArch::XeLp),
            "Alder Lake"
        );
        assert_eq!(
            arch_for_device_id(0x7D55),
            Some(IntelArch::XeLpg),
            "Meteor Lake"
        );
        assert_eq!(
            arch_for_device_id(0x64A0),
            Some(IntelArch::Xe2),
            "Lunar Lake"
        );
    }

    #[test]
    fn unknown_ids_report_nothing_rather_than_guessing() {
        // Pre-Xe integrated graphics are deliberately unmapped: Skylake (Gen9)
        // and Ice Lake (Gen11).
        assert_eq!(arch_for_device_id(0x1912), None, "Skylake GT2");
        assert_eq!(arch_for_device_id(0x8A52), None, "Ice Lake");
        // A device id no range covers.
        assert_eq!(arch_for_device_id(0x0000), None);
        assert_eq!(arch_for_device_id(0xFFFF), None);
    }

    #[test]
    fn ranges_do_not_overlap() {
        // Overlapping ranges would make the result depend on table order.
        for (i, &(first, last, _)) in FAMILIES.iter().enumerate() {
            assert!(first <= last, "range {i} is inverted");
            for &(other_first, other_last, _) in &FAMILIES[i + 1..] {
                assert!(
                    last < other_first || other_last < first,
                    "ranges {first:#06x}..={last:#06x} and \
                     {other_first:#06x}..={other_last:#06x} overlap",
                );
            }
        }
    }
}
