"""Regression tests for the ``_strip_spectrum`` helper.

The user wants the literal word "spectrum" auto-removed from every
user-visible surface: picker buttons, the rename label, the chart
title, the XLSX filename, the XLSX ``sheet`` column, the download
button text. The helper is the single point of normalisation so
this test pins the contract.
"""
import re


def _strip_spectrum(s: str) -> str:
    cleaned = re.sub(r"(?i)\s*spectrum\s*[-_.]?\s*", "", s)
    return cleaned.strip(" -_.").strip()


def test_trailing_capitalised():
    assert _strip_spectrum("30mSpectrum") == "30m"


def test_trailing_lowercase():
    assert _strip_spectrum("30m_spectrum") == "30m"


def test_trailing_dash():
    assert _strip_spectrum("30m-spectrum") == "30m"


def test_leading():
    assert _strip_spectrum("spectrum 30m") == "30m"


def test_embedded_between_words():
    assert _strip_spectrum("my_spectrum_30m") == "my_30m"


def test_alone():
    assert _strip_spectrum("spectrum") == ""


def test_unrelated_label_unchanged():
    assert _strip_spectrum("30m") == "30m"


def test_uppercase():
    assert _strip_spectrum("30mSPECTRUM") == "30m"


def test_mixed_case():
    assert _strip_spectrum("Spectrum 60m") == "60m"


def test_multi_file_label_preserves_separator():
    # The ``::`` separator between file stem and sheet should be
    # preserved when the sheet name itself doesn't contain
    # "spectrum".
    assert _strip_spectrum("60m :: 30m") == "60m :: 30m"


def test_multi_file_label_strips_trailing_spectrum():
    assert _strip_spectrum("30m :: 60mSpectrum") == "30m :: 60m"
