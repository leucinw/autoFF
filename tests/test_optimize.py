"""Residual and Jacobian assembly, with the cluster stubbed out.

These tests replace the simulation layer with callables of known behaviour, so
they check the bookkeeping the optimizer is responsible for: target ordering,
weighting, sidecar index allocation, and finite-difference arithmetic.
"""

import numpy as np
import pytest
import ruamel.yaml as yaml

from autoff import config, optimize
from autoff.hfe import HFEResult
from autoff.singlepoint import Evaluation


def _edit(path, mutate):
    with open(path) as f:
        data = yaml.YAML(typ='safe', pure=True).load(f)
    mutate(data)
    with open(path, 'w') as f:
        yaml.YAML().dump(data, f)


@pytest.fixture
def multi_optimizer(example_run, monkeypatch):
    """An Optimizer over 2 HFE systems + 1 liquid at 2 temperatures."""
    run = example_run('Multi-Property')
    _edit(str(run / 'config.yaml'), lambda d: d['job'].__setitem__('type', 'optimize'))
    cfg = config.load(str(run / 'config.yaml'))

    # Building an Optimizer would otherwise minimize coordinates and split dimers
    monkeypatch.setattr('autoff.singlepoint.Runner.setup', lambda self, sidecars=None: self)
    opt = optimize.Optimizer(cfg, dry_run=True)
    opt.param_file.initialize()   # normally done by the stubbed-out setup()
    return run, opt


def test_target_order_follows_config(multi_optimizer):
    _, opt = multi_optimizer
    assert [t.label for t in opt.targets] == [
        'HFE phenol', 'HFE sodium',
        'Density water_neat@298K', 'Density water_neat@323K',
    ]
    assert [t.kind for t in opt.targets] == ['hfe', 'hfe', 'density', 'density']


def test_residual_is_weighted_and_normalized(multi_optimizer):
    _, opt = multi_optimizer
    phenol = opt.targets[0]
    # residual = weight * (calculated - reference) / denominator
    assert phenol.residual(-5.62) == pytest.approx(1.0 / (6.62 ** 0.5))
    assert phenol.residual(phenol.reference) == 0.0


def test_model_func_builds_residual_vector(multi_optimizer, monkeypatch):
    _, opt = multi_optimizer

    def fake_evaluate(self, prm_path=None, sidecars=None, fresh_liquid=False,
                      collect_hfe=True):
        return Evaluation(
            hfe={'phenol': HFEResult('phenol', -6.00, 0.1),
                 'sodium': HFEResult('sodium', -88.00, 0.2)},
            densities={'water_neat': [1000.0, 990.0]},
        )
    monkeypatch.setattr('autoff.singlepoint.Runner.evaluate', fake_evaluate)

    residuals = opt.model_func(opt.spec.initial)
    assert len(residuals) == 4
    expected = [
        1.0 * (-6.00 - (-6.62)) / (6.62 ** 0.5),
        1.0 * (-88.00 - (-88.7)) / (88.7 ** 0.5),
        0.5 * (1000.0 - 997.0) / 4.5,
        0.5 * (990.0 - 988.0) / 4.5,
    ]
    assert residuals == pytest.approx(expected)


def test_trial_step_uses_the_reweighted_hfe(multi_optimizer, monkeypatch):
    _, opt = multi_optimizer
    seen = {}

    def fake_evaluate(self, prm_path=None, sidecars=None, fresh_liquid=False,
                      collect_hfe=True):
        seen['sidecars'] = sidecars
        seen['fresh_liquid'] = fresh_liquid
        seen['prm_path'] = prm_path
        return Evaluation(
            # fe0 is the reference value; FEP_01 is the trial parameters' value
            hfe={'phenol': HFEResult('phenol', -6.00, 0.1, fep_values={1: -6.40}),
                 'sodium': HFEResult('sodium', -88.0, 0.2, fep_values={1: -88.5})},
            densities={'water_neat': [1000.0, 990.0]},
        )
    monkeypatch.setattr('autoff.singlepoint.Runner.evaluate', fake_evaluate)

    trial = opt.spec.initial + 0.01
    residuals = opt.model_func(trial)

    # A trial point rides on sidecar 01 and needs fresh liquid dynamics
    assert seen['sidecars'] == [1]
    assert seen['fresh_liquid'] is True
    assert seen['prm_path'].endswith('_01')
    # The HFE residual must use the reweighted value, not the reference one
    assert residuals[0] == pytest.approx((-6.40 - (-6.62)) / (6.62 ** 0.5))


def test_trial_step_without_reweighting_window_errors(multi_optimizer, monkeypatch):
    _, opt = multi_optimizer
    monkeypatch.setattr(
        'autoff.singlepoint.Runner.evaluate',
        lambda self, **kw: Evaluation(
            hfe={'phenol': HFEResult('phenol', -6.0, 0.1),
                 'sodium': HFEResult('sodium', -88.0, 0.2)},
            densities={'water_neat': [1000.0, 990.0]}),
    )
    with pytest.raises(RuntimeError, match='FEP_01'):
        opt.model_func(opt.spec.initial + 0.01)


def test_jacobian_sidecar_allocation_and_values(multi_optimizer, monkeypatch):
    _, opt = multi_optimizer
    captured = {}

    def fake_gradient(self, sidecars, perturb_map, diff_step):
        captured['sidecars'] = list(sidecars)
        captured['perturb_map'] = dict(perturb_map)
        # Make the central difference come out to exactly 1.0 per parameter
        hfe_fep = {}
        for name in ('phenol', 'sodium'):
            values = {}
            for j, (plus, minus) in perturb_map.items():
                values[plus] = +diff_step
                values[minus] = -diff_step
            hfe_fep[name] = values
        n = len(perturb_map)
        return hfe_fep, {'water_neat': np.full((2, n), 3.0)}

    monkeypatch.setattr('autoff.singlepoint.Runner.evaluate_gradient', fake_gradient)

    J = opt.jacobian(opt.spec.initial)
    n_params = opt.spec.n_free
    assert J.shape == (4, n_params)
    # Two parameters need two sidecars each, starting at 01 on the reference step
    assert captured['sidecars'] == [1, 2, 3, 4]
    assert captured['perturb_map'] == {0: (1, 2), 1: (3, 4)}

    # (plus - minus) / (2*diff_step) == 1.0, then scaled by weight/denom
    assert J[0, 0] == pytest.approx(1.0 / (6.62 ** 0.5))
    assert J[1, 0] == pytest.approx(1.0 / (88.7 ** 0.5))
    assert J[2, 0] == pytest.approx(0.5 * 3.0 / 4.5)


def test_jacobian_sidecars_start_after_the_trial_point(multi_optimizer, monkeypatch):
    _, opt = multi_optimizer
    captured = {}

    def fake_gradient(self, sidecars, perturb_map, diff_step):
        captured['sidecars'] = list(sidecars)
        n = len(perturb_map)
        hfe = {name: {i: 0.0 for i in sidecars} for name in ('phenol', 'sodium')}
        return hfe, {'water_neat': np.zeros((2, n))}

    monkeypatch.setattr('autoff.singlepoint.Runner.evaluate_gradient', fake_gradient)
    opt.jacobian(opt.spec.initial + 0.01)
    # Sidecar 01 still holds the trial point, so differences start at 02
    assert captured['sidecars'] == [2, 3, 4, 5]


def test_missing_gradient_window_is_reported(multi_optimizer, monkeypatch):
    _, opt = multi_optimizer

    def fake_gradient(self, sidecars, perturb_map, diff_step):
        n = len(perturb_map)
        # phenol is missing its second pair
        return ({'phenol': {1: 0.0, 2: 0.0}, 'sodium': {i: 0.0 for i in sidecars}},
                {'water_neat': np.zeros((2, n))})

    monkeypatch.setattr('autoff.singlepoint.Runner.evaluate_gradient', fake_gradient)
    with pytest.raises(RuntimeError, match='missing reweighting windows'):
        opt.jacobian(opt.spec.initial)


def test_dimer_only_fit_needs_no_cluster(example_run, monkeypatch):
    """A dimer-only fit differentiates locally and never allocates sidecars."""
    run = example_run('Phenol-HFE-Dimer')
    _edit(str(run / 'config.yaml'), lambda d: d.pop('hfe_systems'))
    cfg = config.load(str(run / 'config.yaml'))

    monkeypatch.setattr('autoff.singlepoint.Runner.setup', lambda self, sidecars=None: self)
    opt = optimize.Optimizer(cfg, dry_run=True)
    assert [t.kind for t in opt.targets] == ['dimer']

    def boom(*a, **kw):
        raise AssertionError("a dimer-only fit must not submit cluster jobs")
    monkeypatch.setattr('autoff.singlepoint.Runner.evaluate_gradient', boom)

    # A linear response in the fitted parameter gives a known derivative
    target = opt.runner.dimers[0]
    monkeypatch.setattr(type(target), 'evaluate',
                        lambda self, prm: 2.0 * _first_value(prm))
    opt.param_file.initialize()
    J = opt.jacobian(opt.spec.initial)
    assert J.shape == (1, 2)


def test_max_step_caps_the_loosely_bounded_parameters(multi_optimizer):
    """params_range is absolute, so it frees a small epsilon far more than an
    rmin with a similar range. max_step evens that out, without touching the
    parameters that were already held tight."""
    _, opt = multi_optimizer
    opt.cfg.optimize.max_step = 0.25
    lower, upper = opt._step_bounds()

    # rmin 3.4050 +/- 0.15 is only 4% wide: well inside the 25% cap, so it stays
    assert lower[0] == pytest.approx(3.2550)
    assert upper[0] == pytest.approx(3.5550)
    # epsilon 0.1100 +/- 0.04 is 36% wide, and gets pulled back to 25%
    assert lower[1] == pytest.approx(0.1100 * 0.75)
    assert upper[1] == pytest.approx(0.1100 * 1.25)


def test_settings_banner_marks_the_capped_bounds(multi_optimizer, caplog):
    """A fit that stalls on the cap rather than on the data should be able to
    say so from the log alone."""
    _, opt = multi_optimizer
    opt.cfg.optimize.max_step = 0.25
    with caplog.at_level('INFO'):
        opt._log_settings(*opt._step_bounds())
    banner = caplog.text
    assert 'max_step: 0.25 of each starting value' in banner
    # rmin keeps its configured bounds, epsilon is flagged as narrowed
    assert '3.405 [3.255, 3.555],' in banner
    assert '0.11 [0.0825, 0.1375]*' in banner


def test_max_step_zero_leaves_configured_bounds_alone(multi_optimizer):
    _, opt = multi_optimizer
    opt.cfg.optimize.max_step = 0.0
    lower, upper = opt._step_bounds()
    assert lower == pytest.approx(opt.spec.lower)
    assert upper == pytest.approx(opt.spec.upper)


def test_x_scale_follows_parameter_magnitude(multi_optimizer):
    """An absolute trust region is sized by rmin, and would move an epsilon
    thirty times smaller by its whole value in one step."""
    _, opt = multi_optimizer
    assert opt._x_scale() == pytest.approx([3.4050, 0.1100])


def _first_value(prm_path):
    """Read back the first fitted value from a written parameter file."""
    with open(prm_path) as f:
        for line in reversed(f.readlines()):
            if line.startswith('vdw'):
                return float(line.split()[2])
    return 0.0
