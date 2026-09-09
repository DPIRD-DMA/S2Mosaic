"""README claims that can be checked against the code mechanically.

The README's *Advanced usage* section is the reference for ``mosaic()``'s
parameters, and it drifts silently: nothing fails when a default changes in
``coordinator.py`` and the bullet describing it does not. ``tile_workers`` sat
documented as ``min(4, os.cpu_count() or 1)`` for some time after it became 8.
These tests pin the two properties worth pinning: every parameter is listed,
and every listed default is the one a caller actually gets.
"""

import ast
import inspect
import re
from pathlib import Path

import pytest

from s2mosaic import mosaic
from s2mosaic.aggregation import DEFAULT_TILE_WORKERS
from s2mosaic.config import DEFAULT_ADDITIONAL_QUERY, DEFAULT_BANDS

README = Path(__file__).parent.parent / "README.md"

# Parameters the README documents by their *resolved* value rather than the
# sentinel in the signature, which is the more useful thing for a reader.
# Each entry is a place where a None default is filled in downstream, so the
# mapping doubles as an index of where that happens.
EFFECTIVE_DEFAULTS = {
    "bands": DEFAULT_BANDS,  # normalize_mosaic_inputs
    "additional_query": DEFAULT_ADDITIONAL_QUERY,  # normalize_mosaic_inputs
    "tile_workers": DEFAULT_TILE_WORKERS,  # iter_tile_aggregation
}


def _documented_defaults() -> dict[str, str]:
    """Map parameter name to the default the README shows for it.

    Bullets look like ``- `name` (`default`): description`` or, for the one
    required parameter, ``- `start_year` (required)``. Several bullets cover
    more than one parameter, so this scans the whole section rather than
    matching a bullet at a time.
    """
    text = README.read_text()
    section = text.split("## Advanced usage")[1].split("## Logging")[0]
    documented = dict(re.findall(r"`(\w+)` \(`([^`]+)`\)", section))
    documented.update(
        {n: "required" for n in re.findall(r"`(\w+)` \(required\)", section)}
    )
    return documented


PARAMETERS = inspect.signature(mosaic).parameters


def test_readme_documents_every_mosaic_parameter():
    documented = _documented_defaults()
    assert set(PARAMETERS) - set(documented) == set()


@pytest.mark.parametrize("name", sorted(PARAMETERS))
def test_readme_default_matches_the_code(name):
    documented = _documented_defaults()
    assert name in documented, f"{name} is not documented in README Advanced usage"

    shown = documented[name]
    expected = EFFECTIVE_DEFAULTS.get(name, PARAMETERS[name].default)

    if shown == "required":
        assert expected is inspect.Parameter.empty, (
            f"README calls {name} required, but it defaults to {expected!r}"
        )
        return

    assert expected is not inspect.Parameter.empty, (
        f"README gives {name} a default of {shown}, but it is a required argument"
    )
    assert ast.literal_eval(shown) == expected, (
        f"README documents {name} as {shown}, code gives {expected!r}"
    )
