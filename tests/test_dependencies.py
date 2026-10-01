"""Dependency pins that guard against known-bad releases."""

import re
from pathlib import Path

from packaging.requirements import Requirement

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"


def _requirement(name):
    for line in PYPROJECT.read_text().splitlines():
        match = re.fullmatch(r'\s*"([^"]+)",?\s*', line)
        if match and Requirement(match.group(1)).name == name:
            return Requirement(match.group(1))
    raise AssertionError(f"{name} not found in pyproject.toml dependencies")


class TestRasterioPin:
    """rasterio 1.4.4 bundles libcurl 8.16.0, which deadlocks in
    ``curl_easy_cleanup`` after a lost DNS lookup; repeat DEA mosaics hang."""

    def test_excludes_1_4_4(self):
        assert "1.4.4" not in _requirement("rasterio").specifier

    def test_keeps_the_unaffected_releases_either_side(self):
        spec = _requirement("rasterio").specifier
        # 1.4.3 is what Python 3.10/3.11 resolve to; 1.5 needs Python 3.12.
        assert "1.4.3" in spec
        assert "1.5.1" in spec
