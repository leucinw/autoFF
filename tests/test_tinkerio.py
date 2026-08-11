"""Parsers checked against real Tinker output from a completed run."""

import os
import time

import pytest

from autoff import tinkerio


def test_format_lambda_name():
    assert tinkerio.format_lambda_name('liquid', 0.0, 0.45) == 'liquid-e000-v045'
    assert tinkerio.format_lambda_name('gas', 1.0, 1.0) == 'gas-e100-v100'
    # Perturbed windows live above lambda 1.0
    assert tinkerio.format_lambda_name('liquid', 1.2, 1.2) == 'liquid-e120-v120'


def test_fep_index_roundtrip():
    for idx in range(1, 10):
        lam = tinkerio.lambda_from_fep_index(idx)
        assert tinkerio.fep_index_from_lambda(lam) == idx


def test_read_free_energy(fixtures_dir):
    ene = os.path.join(fixtures_dir, 'ion_hfe_ene', 'liquid-e000-v045.ene')
    fe, err = tinkerio.read_free_energy(ene)
    assert fe == pytest.approx(0.2662, abs=1e-4)
    assert err == pytest.approx(0.0160, abs=1e-4)


def test_ene_complete(fixtures_dir, tmp_path):
    ene = os.path.join(fixtures_dir, 'ion_hfe_ene', 'liquid-e000-v045.ene')
    assert tinkerio.ene_complete(ene)

    truncated = tmp_path / 'partial.ene'
    with open(ene) as f:
        head = f.readlines()[:5]
    truncated.write_text(''.join(head))
    assert not tinkerio.ene_complete(str(truncated))
    assert not tinkerio.ene_complete(str(tmp_path / 'absent.ene'))


def test_bar_sh_steps_match(tmp_path):
    sh = tmp_path / 'bar_x.sh'
    sh.write_text("source env\n"
                  "$BAR9 1 a.arc 300 b.arc 300 N > x.out && \n"
                  "$BAR9 2 x.bar 101 500 1 101 500 1 > x.ene \n")
    assert tinkerio.bar_sh_steps_match(str(sh), 101, 500)
    # A script from a run of a different length must not be accepted
    assert not tinkerio.bar_sh_steps_match(str(sh), 101, 250)
    assert not tinkerio.bar_sh_steps_match(str(tmp_path / 'missing.sh'), 101, 500)


def test_md_crash_reason(tmp_path):
    logp, errp = tmp_path / 'md.log', tmp_path / 'md.err'
    logp.write_text(" Frame Number    12\n Current Potential   -100.0\n")
    assert tinkerio.md_crash_reason(str(logp), str(errp)) is None

    logp.write_text(" Molecular Dynamics Trajectory via r-RESPA MTS Algorithm\n"
                    " Terminating with uncaught exception :  INDUCE  --  Warning, "
                    "Induced Dipoles are not Converged at pcg.cu:172\n")
    assert 'Induced Dipoles are not Converged' in \
        tinkerio.md_crash_reason(str(logp), str(errp))

    # Tinker dumps coordinates even when the log says nothing useful
    logp.write_text(" Frame Number    12\n")
    errp.write_text('  4431\n')
    assert 'md.err' in tinkerio.md_crash_reason(str(logp), str(errp))

    # Neither file present: a run that has not started is not a crash
    assert tinkerio.md_crash_reason(str(tmp_path / 'no.log'),
                                    str(tmp_path / 'no.err')) is None


def _write_bar(path, n0, n1):
    """A .bar with n0 rows in the first block and n1 in the second."""
    row = "       1         -100.0000       -100.0001        46854.9375\n"
    path.write_text(f"{n0:8d}    298.15  Built with Packmol\n" + row * n0 +
                    f"{n1:8d}    298.15  Built with Packmol\n" + row * n1)


def test_bar_file_snapshot_counts(tmp_path):
    bar = tmp_path / 'full.bar'
    _write_bar(bar, 500, 500)
    assert tinkerio.bar_file_snapshot_counts(str(bar)) == (500, 500)

    # A BAR run against a trajectory that was still being written: the first
    # state is complete but the second only got 45 frames in.
    truncated = tmp_path / 'truncated.bar'
    _write_bar(truncated, 500, 45)
    assert tinkerio.bar_file_snapshot_counts(str(truncated)) == (500, 45)

    # Cut off before the second block starts at all
    partial = tmp_path / 'partial.bar'
    partial.write_text("     500    298.15  title\n" + "  1  -1.0  -1.0  1.0\n" * 12)
    assert tinkerio.bar_file_snapshot_counts(str(partial)) == (500, 0)

    assert tinkerio.bar_file_snapshot_counts(str(tmp_path / 'nope.bar')) == (0, 0)
    empty = tmp_path / 'empty.bar'
    empty.write_text('')
    assert tinkerio.bar_file_snapshot_counts(str(empty)) == (0, 0)


def test_count_arc_frames(tmp_path):
    # NPT stride is n_atoms + 2: header, box line, then coordinates
    arc = tmp_path / 'traj.arc'
    frame = ("    2  test\n"
             "   30.0 30.0 30.0 90.0 90.0 90.0\n"
             "    1  O   0.0 0.0 0.0   1  2\n"
             "    2  H   1.0 0.0 0.0   2  1\n")
    arc.write_text(frame * 7)
    assert tinkerio.count_arc_frames(str(arc)) == 7
    assert tinkerio.count_arc_frames(str(tmp_path / 'nope.arc')) == 0


def test_read_txyz_box_and_natoms(examples_dir):
    box_xyz = os.path.join(examples_dir, 'Ion-HFE', 'input', 'Na-water.xyz')
    assert tinkerio.read_txyz_natoms(box_xyz) == 2695
    assert tinkerio.read_txyz_box(box_xyz) == (30.0, 30.0, 30.0)
    # A solute-only file carries no box line
    assert tinkerio.read_txyz_box(os.path.join(examples_dir, 'Ion-HFE', 'input', 'Na.xyz')) is None


def test_parse_system_mass(examples_dir):
    xyz = os.path.join(examples_dir, 'Multi-Property', 'input', 'water_box.xyz')
    prm = os.path.join(examples_dir, 'Multi-Property', 'input', 'amoeba09.prm')
    mass = tinkerio.parse_system_mass(xyz, prm)
    # 898 waters at ~18 g/mol
    assert mass == pytest.approx(898 * 18.015, rel=1e-3)


def test_split_dimer_monomers(examples_dir):
    xyz = os.path.join(examples_dir, 'Phenol-HFE-Dimer', 'input', 'dimers', 'phenol_water.xyz')
    atoms = tinkerio.read_txyz_atoms(xyz)
    mon1, mon2 = tinkerio.split_dimer_monomers(atoms, 13)
    assert len(mon1) == 13
    assert len(mon2) == len(atoms) - 13
    # Bonds are renumbered into each monomer's own 1-based index space
    for mon in (mon1, mon2):
        for atom in mon:
            for bond in atom[6]:
                assert 1 <= bond <= len(mon)


def test_derive_liquid_key_strips_hfe_keywords(tmp_path):
    src = tmp_path / 'hfe.key'
    src.write_text("parameters x.prm\nvdw-annihilate\nligand -1 13\n"
                   "ele-lambda 0.5\nvdw-lambda 0.5\nvdw-cutoff 12.0\n")
    dest = tmp_path / 'neat.key'
    tinkerio.derive_liquid_key(str(src), str(dest))
    text = dest.read_text()
    for keyword in ('vdw-annihilate', 'ligand', 'ele-lambda', 'vdw-lambda'):
        assert keyword not in text
    assert 'vdw-cutoff' in text
    # Tinker9 only writes a trajectory when 'archive' is present
    assert 'archive' in text


def test_rewrite_parameters_line(tmp_path):
    prm = tmp_path / 'ff.prm'
    prm.write_text('')
    out = tinkerio.rewrite_parameters_line("PARAMETERS old.prm\narchive\n", str(prm))
    assert str(prm.resolve()) in out
    assert 'old.prm' not in out


def test_seconds_since_write_takes_the_newest(tmp_path):
    old, new = tmp_path / 'old.arc', tmp_path / 'new.log'
    old.write_text('x')
    new.write_text('x')
    os.utime(old, (time.time() - 7200, time.time() - 7200))
    os.utime(new, (time.time() - 60, time.time() - 60))

    # A quiet .arc does not count as silence while the .log is still moving.
    assert tinkerio.seconds_since_write(str(old)) > 7000
    assert tinkerio.seconds_since_write(str(old), str(new)) < 120
    assert tinkerio.seconds_since_write(str(tmp_path / 'nothing')) is None


def test_stall_reason_needs_silence_across_every_output(tmp_path):
    arc, logp = tmp_path / 'w.arc', tmp_path / 'w.log'
    arc.write_text('x')
    logp.write_text('x')
    stale = time.time() - 7200
    os.utime(arc, (stale, stale))
    os.utime(logp, (stale, stale))

    assert 'no output written' in tinkerio.stall_reason(3600, str(arc), str(logp))
    # Timeout not yet reached, and 0 disables the check outright.
    assert tinkerio.stall_reason(10800, str(arc), str(logp)) is None
    assert tinkerio.stall_reason(0, str(arc), str(logp)) is None

    # A job that has not written its first frame has no mtime to judge and
    # must never be declared dead -- it may simply be slow to start.
    assert tinkerio.stall_reason(3600, str(tmp_path / 'missing.arc')) is None

    # One live output is enough to clear the stall.
    os.utime(logp, None)
    assert tinkerio.stall_reason(3600, str(arc), str(logp)) is None
