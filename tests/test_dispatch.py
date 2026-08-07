"""Job batching and the dry-run manifest."""

import os

import pytest

from autoff import config, singlepoint
from autoff.dispatch import Job, JobDispatcher


def test_batches_by_queue_and_nproc(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr('autoff.submit.submit_jobs',
                        lambda cmds, qt, gpu, cpu, n: calls.append((qt, n, list(cmds))))
    dispatcher = JobDispatcher(nodes=['node1'], dry_run=False)
    dispatcher.submit([
        Job('a.sh', '/w/gas', 'CPU', 4),
        Job('b.sh', '/w/gas', 'CPU', 4),
        Job('c.sh', '/w/liquid', 'GPU', 2),
    ])
    assert len(calls) == 2
    by_queue = {c[0]: c for c in calls}
    assert by_queue['CPU'][1] == 4
    assert by_queue['CPU'][2] == ['cd /w/gas; sh a.sh', 'cd /w/gas; sh b.sh']
    assert by_queue['GPU'][2] == ['cd /w/liquid; sh c.sh']


def test_dry_run_records_instead_of_submitting(tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("dry run must not submit")
    monkeypatch.setattr('autoff.submit.submit_jobs', boom)

    manifest = tmp_path / 'results' / 'submitted_jobs.txt'
    dispatcher = JobDispatcher(dry_run=True, manifest_path=str(manifest))
    dispatcher.submit([Job('a.sh', '/w/gas', 'CPU', 4, label='phenol/gas md')])
    text = manifest.read_text()
    assert 'CPU' in text and 'phenol/gas md' in text and '/w/gas/a.sh' in text


def test_empty_submission_is_a_noop(monkeypatch):
    monkeypatch.setattr('autoff.submit.submit_jobs',
                        lambda *a, **kw: pytest.fail("should not be called"))
    assert JobDispatcher().submit([]) == 0


def test_report_collects_without_submitting(example_run, fixtures_dir, monkeypatch):
    """`autoff report` must read existing output, never start new work."""
    import shutil

    from autoff import cli

    run = example_run('Ion-HFE')
    liquid_dir = run / 'systems' / 'sodium' / 'liquid'
    liquid_dir.mkdir(parents=True)
    for ene in os.listdir(os.path.join(fixtures_dir, 'ion_hfe_ene')):
        if ene.endswith('.ene'):
            shutil.copy(os.path.join(fixtures_dir, 'ion_hfe_ene', ene), liquid_dir)

    def boom(*a, **kw):
        raise AssertionError("report must not submit jobs")
    monkeypatch.setattr('autoff.submit.submit_jobs', boom)
    monkeypatch.setattr('autoff.dispatch.JobDispatcher.submit', boom)

    assert cli.main(['report', str(run / 'config.yaml'), '-s']) == 0
    text = (run / 'results' / 'singlepoint.txt').read_text()
    assert '-90.9062' in text
    assert '-88.7000' in text          # the reference value, for comparison
    # Reporting must not re-minimize either: no Tinker was invoked, so no log
    assert not (run / 'systems' / 'sodium' / 'liquid-min.log').exists()


def test_multi_system_dry_run_covers_every_system(example_run):
    """A whole multi-system run is planned without touching the cluster."""
    run = example_run('Multi-Property')
    # Skip the local minimization step, which needs a real Tinker build
    for name in ('phenol', 'sodium'):
        d = run / 'systems' / name
        d.mkdir(parents=True)
        for phase in ('gas', 'liquid'):
            (d / f'{phase}-min.log').write_text('')

    cfg = config.load(str(run / 'config.yaml'))
    singlepoint.run(cfg, dry_run=True)

    manifest = (run / 'results' / 'submitted_jobs.txt').read_text().splitlines()
    md = [ln for ln in manifest if ' md ' in ln]
    bar = [ln for ln in manifest if ' bar ' in ln]

    # phenol: 18 gas + 18 liquid; sodium: 18 liquid; water: 2 temperatures
    assert len(md) == 18 + 18 + 18 + 2
    # BAR pairs one fewer than windows, for phenol (both phases) and sodium
    assert len(bar) == 17 + 17 + 17
    assert any('water_neat@298K' in ln for ln in manifest)
    assert all(ln.startswith(('CPU', 'GPU')) for ln in manifest)

    # Each system keeps its own directory, so nothing can collide
    for name in ('phenol', 'sodium', 'water_neat'):
        assert (run / 'systems' / name).is_dir()
