"""HFE setup, generated scripts, and result collection.

The collection test pins the new code to output produced by the original
implementation: the .ene fixtures come from a completed Na+ run whose
committed result.txt reported -90.9062 +/- 0.5426 kcal/mol.
"""

import os
import shutil

import pytest

from autoff import config, singlepoint


@pytest.fixture
def phenol_runner(example_run):
    run = example_run('Phenol-HFE')
    # Pre-place the minimization markers: setup() then trusts the coordinates
    # as given instead of shelling out to a Tinker build that is not present.
    (run / 'systems' / 'phenol').mkdir(parents=True)
    for phase in ('gas', 'liquid'):
        (run / 'systems' / 'phenol' / f'{phase}-min.log').write_text('')
    cfg = config.load(str(run / 'config.yaml'))
    runner = singlepoint.Runner(cfg, dry_run=True)
    runner.setup()
    return run, runner


def test_setup_writes_one_key_per_window(phenol_runner):
    run, runner = phenol_runner
    system = runner.hfe_systems[0]
    n_windows = len(system.orderparams)
    assert n_windows == 18                      # the 'courser' schedule
    for phase in ('gas', 'liquid'):
        keys = list((run / 'systems' / 'phenol' / phase).glob('*.key'))
        assert len(keys) == n_windows


def test_window_key_contents(phenol_runner):
    run, runner = phenol_runner
    key = (run / 'systems' / 'phenol' / 'liquid' / 'liquid-e000-v045.key').read_text()
    assert 'ligand -1 13' in key                # phenol has 13 atoms
    assert 'ele-lambda 0.0' in key
    assert 'vdw-lambda 0.45' in key
    # Keys must point at the run's own copy of the parameters
    assert 'prm/amoeba09.prm' in key


def test_md_script_command_line(phenol_runner):
    run, runner = phenol_runner
    system = runner.hfe_systems[0]
    system.md_jobs()

    liquid = (run / 'systems' / 'phenol' / 'liquid' / 'liquid-e000-v045.sh').read_text()
    assert ('$DYNAMIC9 liquid-e000-v045.xyz -key liquid-e000-v045.key '
            '250000 2.0 1.0 4 300.0 1.0 > liquid-e000-v045.log') in liquid

    gas = (run / 'systems' / 'phenol' / 'gas' / 'gas-e000-v045.sh').read_text()
    # The gas leg runs on Tinker8 with the stochastic integrator and no pressure
    assert ('$DYNAMIC8 gas-e000-v045.xyz -key gas-e000-v045.key '
            '5000000 0.1 1.0 2 300.0 > gas-e000-v045.log') in gas


def test_bar_script_command_line(phenol_runner):
    run, runner = phenol_runner
    system = runner.hfe_systems[0]
    system.md_jobs()
    system.bar_jobs()

    sh = (run / 'systems' / 'phenol' / 'liquid'
          / 'bar_e000-v000_e000-v045.sh').read_text()
    assert '$BAR9 1 liquid-e000-v000.arc 300.0 liquid-e000-v045.arc 300.0 N' in sh
    # BAR discards the first fifth of the trajectory: 500/5 + 1 = 101
    assert '$BAR9 2 liquid-e000-v000.bar 101 500 1 101 500 1' in sh

    gas_sh = (run / 'systems' / 'phenol' / 'gas'
              / 'bar_e000-v045_e000-v000.sh').read_text()
    # The gas leg is integrated in the opposite direction
    assert '$BAR8 1 gas-e000-v045.arc 300.0 gas-e000-v000.arc 300.0 N' in gas_sh


def test_job_queues_and_counts(phenol_runner):
    _, runner = phenol_runner
    system = runner.hfe_systems[0]
    jobs = system.md_jobs()
    gas = [j for j in jobs if 'gas' in j.workdir]
    liquid = [j for j in jobs if j.workdir.endswith('liquid')]
    assert len(gas) == 18 and len(liquid) == 18
    # The small gas system runs on CPU cores; the box needs a GPU
    assert all(j.queue == 'CPU' and j.nproc == 4 for j in gas)
    assert all(j.queue == 'GPU' for j in liquid)


def test_perturbed_windows_reuse_the_coupled_trajectory(phenol_runner):
    run, runner = phenol_runner
    system = runner.hfe_systems[0]
    system.set_sidecars([1])
    system.setup(dry_run=True)
    assert len(system.orderparams) == 19        # 18 real windows plus one reweighting

    key = (run / 'systems' / 'phenol' / 'liquid' / 'liquid-e110-v110.key').read_text()
    # A reweighting window stays fully coupled and only swaps the parameters
    assert 'ele-lambda 1.0' in key
    assert 'vdw-lambda 1.0' in key
    assert 'amoeba09.prm_01' in key


def _bar_text(n0, n1):
    row = "       1         -100.0000       -100.0001        46854.9375\n"
    return (f"{n0:8d}    298.15  title\n" + row * n0 +
            f"{n1:8d}    298.15  title\n" + row * n1)


def test_stale_bar_dropped_when_second_state_truncated(phenol_runner):
    """A BAR that ran against a half-written trajectory must not be trusted.

    Its .ene is all NaN and never converges, so leaving the pair in place
    stalls the run forever; removing it lets the window be rebuilt.
    """
    _, runner = phenol_runner
    system = runner.hfe_systems[0]
    w = next(iter(system._window_pairs('liquid')))
    total, start = w['total'], w['start']

    barpath = os.path.join(w['bardir'], w['stem'] + '.bar')
    enepath = os.path.join(w['bardir'], w['stem'] + '.ene')
    shpath = os.path.join(w['bardir'], w['sh_name'])
    with open(shpath, 'w') as f:
        f.write(f"$BAR9 2 x.bar {start} {total} 1 {start} {total} 1 > x.ene\n")

    # Both states complete: kept.
    with open(barpath, 'w') as f:
        f.write(_bar_text(total, total))
    with open(enepath, 'w') as f:
        f.write(' BAR Estimate of -T*dS   0.01 Kcal/mol\n')
    system._drop_stale_bar(w, barpath, enepath, shpath)
    assert os.path.isfile(barpath) and os.path.isfile(enepath)

    # Second state short: both files go, even though the header says total.
    with open(barpath, 'w') as f:
        f.write(_bar_text(total, 45))
    system._drop_stale_bar(w, barpath, enepath, shpath)
    assert not os.path.isfile(barpath)
    assert not os.path.isfile(enepath)


def test_collect_reproduces_original_result(example_run, fixtures_dir):
    run = example_run('Ion-HFE')
    liquid_dir = run / 'systems' / 'sodium' / 'liquid'
    liquid_dir.mkdir(parents=True)
    for ene in os.listdir(os.path.join(fixtures_dir, 'ion_hfe_ene')):
        if ene.endswith('.ene'):
            shutil.copy(os.path.join(fixtures_dir, 'ion_hfe_ene', ene), liquid_dir)

    cfg = config.load(str(run / 'config.yaml'))
    runner = singlepoint.Runner(cfg, dry_run=False, skip_check=True)
    result = runner.hfe_systems[0].collect()

    assert len(result.windows) == 17
    assert result.fe0 == pytest.approx(-90.9062, abs=1e-4)
    assert result.error == pytest.approx(0.5426, abs=1e-4)


def test_report_lists_every_window(example_run, fixtures_dir, tmp_path):
    run = example_run('Ion-HFE')
    liquid_dir = run / 'systems' / 'sodium' / 'liquid'
    liquid_dir.mkdir(parents=True)
    for ene in os.listdir(os.path.join(fixtures_dir, 'ion_hfe_ene')):
        if ene.endswith('.ene'):
            shutil.copy(os.path.join(fixtures_dir, 'ion_hfe_ene', ene), liquid_dir)

    cfg = config.load(str(run / 'config.yaml'))
    runner = singlepoint.Runner(cfg, dry_run=False, skip_check=True)
    system = runner.hfe_systems[0]
    out = tmp_path / 'hfe_sodium.txt'
    system.write_report(system.collect(), str(out))
    text = out.read_text()
    assert 'SUM OF THE TOTAL FREE ENERGY' in text
    assert '-90.9062' in text
