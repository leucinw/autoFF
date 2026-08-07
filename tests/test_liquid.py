"""Neat-liquid MD scripts, density parsing, and the density derivative."""

import os

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
