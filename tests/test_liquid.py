"""Neat-liquid MD scripts, density parsing, and the density derivative."""

import os
import time

import numpy as np
import pytest

from autoff import config, singlepoint, tinkerio
from autoff.liquid import _density_derivative


def _md_log(frames, lattice_a=30.0):
    """Build a Tinker9-style MD log with *frames* recorded frames."""
    out = []
    for i, a in enumerate(lattice_a if isinstance(lattice_a, list) else [lattice_a] * frames):
        out.append(f" Current Potential           -{10000 + i}.0000 Kcal/mole")
        out.append(f" Lattice Lengths     {a:.6f}   {a:.6f}   {a:.6f}")
        out.append(f" Frame Number           {i + 1}")
    return "\n".join(out) + "\n"


@pytest.fixture
def water_runner(example_run):
    run = example_run('Multi-Property')
    cfg = config.load(str(run / 'config.yaml'))
    runner = singlepoint.Runner(cfg, dry_run=True)
    runner.param_file.initialize()   # liquid setup reads the snapshot for masses
    liquid = runner.liquids[0]
    liquid.setup(dry_run=True)
    return run, liquid


def test_setup_generates_key_and_scripts(water_runner):
    run, liquid = water_runner
    system_dir = run / 'systems' / 'water_neat'
    assert (system_dir / 'water_neat.key').is_file()
    for T in (298, 323):
        assert (system_dir / f'water_neat_{T}K.sh').is_file()
        assert (system_dir / f'water_neat_{T}K.xyz').is_symlink()
        assert (system_dir / f'water_neat_{T}K.key').is_symlink()


def test_neat_liquid_key_has_no_alchemical_keywords(water_runner):
    run, liquid = water_runner
    text = (run / 'systems' / 'water_neat' / 'water_neat.key').read_text()
    for keyword in ('vdw-annihilate', 'ligand', 'ele-lambda', 'vdw-lambda'):
        assert keyword not in text
    assert 'archive' in text


def test_md_script_command_line(water_runner):
    run, liquid = water_runner
    text = (run / 'systems' / 'water_neat' / 'water_neat_298K.sh').read_text()
    # (0.02 + 0.2) ns at 0.1 ps/frame = 2200 frames; 50 steps/frame at 2 fs
    assert liquid.cfg.total_steps == 110000
    assert ('$DYNAMIC9 water_neat_298K.xyz -k water_neat_298K.key '
            '110000 2.0 0.1 4 298.15 1.0 > water_neat_298K.log') in text


def test_md_jobs_skip_complete_trajectories(water_runner, monkeypatch):
    _, liquid = water_runner
    n_total = liquid.cfg.n_equil + liquid.cfg.n_production
    monkeypatch.setattr(tinkerio, 'count_arc_frames', lambda p: n_total)
    assert liquid.md_jobs() == []
    assert liquid.md_complete()


def test_md_jobs_resume_from_checkpoint(water_runner, monkeypatch):
    run, liquid = water_runner
    n_total = liquid.cfg.n_equil + liquid.cfg.n_production
    done = n_total // 2
    monkeypatch.setattr(tinkerio, 'count_arc_frames', lambda p: done)
    for T in liquid.cfg.temperatures:
        open(os.path.join(liquid.dir, f'{liquid._base(T)}.dyn'), 'w').close()

    jobs = liquid.md_jobs()
    assert len(jobs) == 2
    assert all(j.script.endswith('-resume.sh') for j in jobs)
    text = (run / 'systems' / 'water_neat' / 'water_neat_298K-resume.sh').read_text()
    # Only the outstanding steps are requested, appending to the existing log
    assert f' {(n_total - done) * liquid.cfg.steps_per_frame} ' in text
    assert '>> water_neat_298K.log' in text


def test_md_jobs_fresh_discards_old_trajectory(water_runner):
    run, liquid = water_runner
    arc = run / 'systems' / 'water_neat' / 'water_neat_298K.arc'
    arc.write_text('stale trajectory from previous parameters')
    jobs = liquid.md_jobs(fresh=True)
    assert len(jobs) == 2
    assert not arc.exists()


CRASH_LOG = (" Molecular Dynamics Trajectory via r-RESPA MTS Algorithm\n"
             " Terminating with uncaught exception :  INDUCE  --  Warning, "
             "Induced Dipoles are not Converged\n")


def test_md_complete_raises_when_dynamics_died(water_runner, monkeypatch):
    """A crashed run must not read as 'still going': nothing resubmits it."""
    _, liquid = water_runner
    monkeypatch.setattr(tinkerio, 'count_arc_frames', lambda p: 0)

    # Short trajectory with a healthy log: the run is simply not finished yet.
    for T in liquid.cfg.temperatures:
        with open(liquid.log_path(T), 'w') as f:
            f.write(_md_log(3))
    assert liquid.md_complete() is False

    # Same frame count, but the dynamics blew up at one temperature.
    T = liquid.cfg.temperatures[0]
    with open(liquid.log_path(T), 'w') as f:
        f.write(CRASH_LOG)
    with pytest.raises(RuntimeError, match="Induced Dipoles are not Converged"):
        liquid.md_complete()

    # A coordinate dump alone is enough, even with nothing wrong in the log.
    with open(liquid.log_path(T), 'w') as f:
        f.write(_md_log(3))
    open(liquid.err_path(T), 'w').close()
    with pytest.raises(RuntimeError, match=r"\.err"):
        liquid.md_complete()


def test_md_jobs_clear_the_previous_crash_dump(water_runner, monkeypatch):
    """Resubmitting must not inherit the last run's .err, or it looks crashed."""
    _, liquid = water_runner
    monkeypatch.setattr(tinkerio, 'count_arc_frames', lambda p: 0)
    for T in liquid.cfg.temperatures:
        open(liquid.err_path(T), 'w').close()

    assert len(liquid.md_jobs(fresh=True)) == 2
    assert not any(os.path.isfile(liquid.err_path(T)) for T in liquid.cfg.temperatures)
    # With the dumps gone the fresh run reads as pending again, not crashed
    assert liquid.md_complete() is False


def test_densities_from_log(water_runner):
    run, liquid = water_runner
    n_equil, n_prod = liquid.cfg.n_equil, liquid.cfg.n_production
    for T in liquid.cfg.temperatures:
        path = run / 'systems' / 'water_neat' / f'{liquid._base(T)}.log'
        path.write_text(_md_log(n_equil + n_prod))

    means = liquid.densities()
    expected = liquid.total_mass / (tinkerio.DENSITY_FACTOR * 30.0 ** 3)
    assert means == pytest.approx([expected, expected])
    assert len(liquid.rho_frames[0]) == n_prod


def test_densities_drop_equilibration_frames(water_runner):
    run, liquid = water_runner
    n_equil, n_prod = liquid.cfg.n_equil, liquid.cfg.n_production
    # Equilibration frames sit at a different box size than production
    lattice = [29.0] * n_equil + [30.0] * n_prod
    for T in liquid.cfg.temperatures:
        path = run / 'systems' / 'water_neat' / f'{liquid._base(T)}.log'
        path.write_text(_md_log(len(lattice), lattice))

    means = liquid.densities()
    production_only = liquid.total_mass / (tinkerio.DENSITY_FACTOR * 30.0 ** 3)
    assert means[0] == pytest.approx(production_only)


def test_density_derivative_formula():
    # With rho uncorrelated from dE/dl the fluctuation term vanishes
    rho = np.array([1.0, 1.0, 1.0, 1.0])
    e_plus = np.array([1.0, 2.0, 3.0, 4.0])
    e_minus = np.array([0.0, 1.0, 2.0, 3.0])
    assert _density_derivative(rho, e_plus, e_minus, beta=1.0, diff_step=0.5) == pytest.approx(0.0)

    # A positive correlation gives a negative derivative: -beta * covariance
    rho = np.array([1.0, 2.0, 3.0, 4.0])
    derivative = _density_derivative(rho, e_plus, e_minus, beta=2.0, diff_step=0.5)
    dEdl = (e_plus - e_minus) / 1.0
    expected = -2.0 * ((rho * dEdl).mean() - rho.mean() * dEdl.mean())
    assert derivative == pytest.approx(expected)


def test_analyze_jobs_cover_every_sidecar_and_temperature(water_runner):
    run, liquid = water_runner
    n_equil, n_prod = liquid.cfg.n_equil, liquid.cfg.n_production
    for T in liquid.cfg.temperatures:
        arc = run / 'systems' / 'water_neat' / f'{liquid._base(T)}.arc'
        # One frame per line keeps the fixture small; only the trim path matters
        arc.write_text("    1  x\n   30.0 30.0 30.0 90.0 90.0 90.0\n"
                       "    1  O   0.0 0.0 0.0   36\n" * (n_equil + n_prod))

    jobs, log_map = liquid.analyze_jobs([1, 2])
    assert len(jobs) == 4                       # 2 sidecars x 2 temperatures
    assert set(log_map) == {(1, 0), (1, 1), (2, 0), (2, 1)}
    text = (run / 'systems' / 'water_neat' / 'water_neat_298K-prm01-analyze.sh').read_text()
    assert '$ANALYZE9 water_neat_298K-prod.arc' in text
    # The full trajectory is replaced by the production-only copy
    assert not (run / 'systems' / 'water_neat' / 'water_neat_298K.arc').exists()
    assert (run / 'systems' / 'water_neat' / 'water_neat_298K-prod.arc').exists()


def test_md_complete_flags_a_silently_dead_run(water_runner, monkeypatch):
    """Output that stopped arriving is the only trace a killed job leaves.

    Reproduces the isoPrOH_neat 278 K stall of 2026-08-10: the job died with
    its node at 721/3000 frames, its log ended mid-frame with nothing wrong in
    it, and the poller then waited eight hours on a count that never moved.
    """
    _, liquid = water_runner
    liquid.stall_timeout = 3600.0
    monkeypatch.setattr(tinkerio, 'count_arc_frames', lambda p: 3)

    stale = time.time() - 8 * 3600
    for T in liquid.cfg.temperatures:
        # A perfectly healthy log -- md_crash_reason has nothing to find.
        with open(liquid.log_path(T), 'w') as f:
            f.write(_md_log(3))
        open(liquid.arc_path(T), 'w').close()

    # While the files are fresh the run is merely slow, not dead.
    assert liquid.md_complete() is False

    T = liquid.cfg.temperatures[0]
    os.utime(liquid.log_path(T), (stale, stale))
    os.utime(liquid.arc_path(T), (stale, stale))
    with pytest.raises(RuntimeError, match="no output written"):
        liquid.md_complete()


def test_stall_check_is_off_by_default_and_disablable(water_runner, monkeypatch):
    """A zero timeout restores the old behaviour: wait forever, never raise."""
    _, liquid = water_runner
    monkeypatch.setattr(tinkerio, 'count_arc_frames', lambda p: 3)
    stale = time.time() - 8 * 3600
    for T in liquid.cfg.temperatures:
        with open(liquid.log_path(T), 'w') as f:
            f.write(_md_log(3))
        os.utime(liquid.log_path(T), (stale, stale))

    liquid.stall_timeout = 0.0
    assert liquid.md_complete() is False
