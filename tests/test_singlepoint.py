"""Job ordering in the gradient round, with the cluster stubbed out.

BAR reads two complete trajectories per window, so it must never be dispatched
while dynamics are still writing: a BAR started against a half-written .arc
builds a .bar that is truncated on one side and an .ene full of NaN that never
converges, which the polling loop then waits on forever.
"""

import pytest

from autoff import config, singlepoint
from autoff.dispatch import Job
from autoff.hfe import HFEResult


class RecordingDispatcher:
    """Stands in for the cluster, keeping the order of every submitted batch."""

    def __init__(self):
        self.batches = []

    def submit(self, jobs):
        self.batches.append([j.label for j in jobs])


@pytest.fixture
def gradient_runner(example_run, monkeypatch):
    """A Runner whose one HFE system reports MD as unfinished at first."""
    run = example_run('Phenol-HFE')
    cfg = config.load(str(run / 'config.yaml'))
    cfg.checking_time = 0            # no real sleeping between polls
    monkeypatch.setattr(singlepoint.Runner, 'setup', lambda self, sidecars=None: self)
    runner = singlepoint.Runner(cfg, dry_run=False)
    runner.dispatcher = RecordingDispatcher()

    system = runner.hfe_systems[0]
    calls = {'md_complete': 0}

    def md_complete():
        # Two polls short of the trajectories being finished.
        calls['md_complete'] += 1
        return calls['md_complete'] > 2

    monkeypatch.setattr(system, 'set_sidecars', lambda sidecars: None)
    monkeypatch.setattr(system, 'setup', lambda dry_run=False: None)
    monkeypatch.setattr(system, 'md_jobs', lambda: [Job('md.sh', '/w', 'GPU', 2, label='md')])
    monkeypatch.setattr(system, 'md_complete', md_complete)
    monkeypatch.setattr(system, 'bar_jobs',
                        lambda: [Job('bar.sh', '/w', 'GPU', 2, label='bar')])
    monkeypatch.setattr(system, 'bar_complete', lambda: True)
    monkeypatch.setattr(system, 'collect', lambda: HFEResult(
        name=system.name, fe0=0.0, error=0.0, windows=[], fep_values={0: -5.0}))
    return runner, calls


def test_bar_is_held_until_md_finishes(gradient_runner):
    runner, calls = gradient_runner
    runner.evaluate_gradient([], {}, 1e-3)

    # MD goes out alone; BAR only follows once md_complete() turns True.
    assert runner.dispatcher.batches == [['md'], ['bar']]
    assert calls['md_complete'] == 3


def test_dry_run_still_emits_bar_scripts(example_run, monkeypatch):
    run = example_run('Phenol-HFE')
    cfg = config.load(str(run / 'config.yaml'))
    monkeypatch.setattr(singlepoint.Runner, 'setup', lambda self, sidecars=None: self)
    runner = singlepoint.Runner(cfg, dry_run=True)
    runner.dispatcher = RecordingDispatcher()

    system = runner.hfe_systems[0]
    monkeypatch.setattr(system, 'set_sidecars', lambda sidecars: None)
    monkeypatch.setattr(system, 'setup', lambda dry_run=False: None)
    monkeypatch.setattr(system, 'md_jobs', lambda: [Job('md.sh', '/w', 'GPU', 2, label='md')])
    monkeypatch.setattr(system, 'bar_jobs',
                        lambda: [Job('bar.sh', '/w', 'GPU', 2, label='bar')])

    assert runner.evaluate_gradient([], {}, 1e-3) == ({}, {})
    assert runner.dispatcher.batches == [['md'], ['bar']]
