"""Stage names: titles on screen, names or IDs accepted on the command line."""

import pytest

from cellwarden import runner as R
from cellwarden import stage_names as N
from cellwarden.config import ConfigError
from cellwarden.gates.protocol import GateError, load_gate


def test_every_stage_has_a_unique_name_and_title():
    ids = [s.id for s in R.default_stages()]
    assert set(ids) == set(N.STAGES)
    slugs = [N.slug(i) for i in ids]
    assert len(set(slugs)) == len(slugs) and all(s == s.lower() and " " not in s for s in slugs)
    assert N.title("S2").startswith("Normalise") and N.label("G1") == "QC review (G1)"
    assert N.title("X9") == "X9" and N.slug("X9") == "X9" and N.label("X9") == "X9"  # test/plugin stages


@pytest.mark.parametrize("name", ["S2", "s2", "clustering", "CLUSTERING", "Normalise, PCA, batch correction, clustering"])
def test_until_accepts_id_name_or_title(name):
    stages = R.default_stages()
    assert stages[R._stage_index(stages, name)].id == "S2"


def test_unknown_stage_lists_the_names():
    with pytest.raises(ConfigError, match="clustering") as e:
        R._stage_index(R.default_stages(), "clusterin")
    assert "IDs S0, S1a" in str(e.value)


def test_gate_names():
    assert N.gate_id("context-review") == "G2" and N.gate_id("g3") == "G3" and N.gate_id("qc_review") == "G1"
    assert N.gate_id("clustering") == "clustering"  # not a gate: unchanged, so the caller reports it
    with pytest.raises(GateError, match="interpretation-review"):
        load_gate("nope", ".", {})
