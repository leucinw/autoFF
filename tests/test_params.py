"""Parameter parsing, bounds, and parameter-file rewriting."""

import os

import numpy as np
import pytest

from autoff import params


def test_parse_simple_entry():
    spec = params.parse_opt_params(["vdw-36 3.4050 0.1100"], ["0.20 0.05"])
    assert spec.n_free == 2
    assert list(spec.initial) == [3.4050, 0.1100]
    assert list(spec.lower) == pytest.approx([3.2050, 0.0600])
    assert list(spec.upper) == pytest.approx([3.6050, 0.1600])


def test_zero_range_pins_a_parameter():
    spec = params.parse_opt_params(["vdw-36 3.4050 0.1100"], ["0.20 0"])
    # Only the first value is exposed to the optimizer
    assert spec.n_free == 1
    assert list(spec.initial) == [3.4050]
    # ...but both are written out, the pinned one unchanged
    assert spec.render_lines(np.array([3.5])) == ["vdw   36  3.5  0.11"]


def test_multiple_groups_share_one_vector():
    spec = params.parse_opt_params(
        ["vdw-36 3.4050 0.1100", "vdwpair-401-402 3.71 0.10"],
        ["0.20 0.05", "0.10 0.02"],
    )
    assert spec.n_free == 4
    assert [e.free_start for e in spec.entries] == [0, 2]
    lines = spec.render_lines(np.array([3.4, 0.11, 3.7, 0.1]))
    assert lines[0].startswith("vdw   36")
    # Hyphens in a term key separate the Tinker fields
    assert lines[1].startswith("vdwpair   401   402")


def test_lower_bound_clamped_above_zero():
    # A range wider than the value would let the optimizer drive it negative
    spec = params.parse_opt_params(["vdw-36 0.05 0.1100"], ["0.20 0.05"])
    assert spec.lower[0] == params.MIN_LOWER_BOUND


def test_mismatched_range_length_rejected():
    with pytest.raises(SystemExit, match='must match'):
        params.parse_opt_params(["vdw-36 3.4050 0.1100"], ["0.20"])


def test_all_fixed_rejected():
    with pytest.raises(SystemExit, match='no free parameters'):
        params.parse_opt_params(["vdw-36 3.4050 0.1100"], ["0 0"])


def test_too_many_free_parameters_rejected():
    # Each parameter needs two sidecar slots, and only 99 exist
    values = " ".join(str(1.0 + i) for i in range(60))
    ranges = " ".join("0.1" for _ in range(60))
    with pytest.raises(SystemExit, match='sidecar'):
        params.parse_opt_params([f"vdw-36 {values}"], [ranges])


def test_reconstruct_merges_fixed_values():
    spec = params.parse_opt_params(["vdw-36 1.0 2.0 3.0"], ["0.1 0 0.3"])
    full = spec.entries[0].reconstruct(np.array([1.5, 3.5]))
    assert full == pytest.approx([1.5, 2.0, 3.5])


class TestParamFile:
    def _make(self, tmp_path):
        source = tmp_path / 'ff.prm'
        source.write_text("atom 1 1 O \"Water O\" 8 15.999 2\nvdw   36   3.4050   0.1100\n")
        return params.ParamFile(str(source), str(tmp_path / 'prm')).initialize()

    def test_initialize_snapshots_without_touching_source(self, tmp_path):
        pf = self._make(tmp_path)
        assert os.path.isfile(pf.snapshot)
        assert os.path.isfile(pf.working)
        # The user's input file must never be modified
        assert (tmp_path / 'ff.prm').read_text() == open(pf.snapshot).read()

    def test_write_appends_overrides_to_snapshot(self, tmp_path):
        pf = self._make(tmp_path)
        pf.write(pf.working, ["vdw   36  3.5000  0.1200"])
        text = open(pf.working).read()
        assert text.endswith("vdw   36  3.5000  0.1200\n")
        # Tinker takes the last definition, so the original line stays put
        assert text.count("vdw   36") == 2

    def test_repeated_writes_do_not_accumulate(self, tmp_path):
        pf = self._make(tmp_path)
        for value in (3.5, 3.6, 3.7):
            pf.write(pf.working, [f"vdw   36  {value}  0.11"])
        text = open(pf.working).read()
        assert text.count("vdw   36") == 2
        assert "3.7" in text and "3.5" not in text

    def test_sidecar_naming(self, tmp_path):
        pf = self._make(tmp_path)
        assert pf.sidecar(1).endswith('ff.prm_01')
        assert pf.sidecar(12).endswith('ff.prm_12')

    def test_cleanup_removes_sidecars_and_fep_dirs(self, tmp_path):
        pf = self._make(tmp_path)
        pf.write(pf.sidecar(1), [])
        pf.write(pf.sidecar(2), [])
        systems = tmp_path / 'systems'
        fep = systems / 'phenol' / 'liquid' / 'FEP_01'
        fep.mkdir(parents=True)
        (fep / 'stale.key').write_text('x')
        stale_key = systems / 'phenol' / 'liquid' / 'liquid-e110-v110.key'
        stale_key.write_text('x')

        pf.cleanup_sidecars(str(systems))

        assert not os.path.exists(pf.sidecar(1))
        assert not os.path.exists(pf.sidecar(2))
        assert not fep.exists()
        assert not stale_key.exists()
        # The snapshot and working file must survive cleanup
        assert os.path.isfile(pf.snapshot) and os.path.isfile(pf.working)

    def test_restore_reverts_working_file(self, tmp_path):
        pf = self._make(tmp_path)
        pf.write(pf.working, ["vdw   36  9.9  9.9"])
        pf.restore()
        assert "9.9" not in open(pf.working).read()
