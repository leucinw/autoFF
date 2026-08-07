"""Config loading, defaults, and the validation errors users are likely to hit."""

import os

import pytest
import ruamel.yaml as yaml

from autoff import config


def _load_yaml(path):
    with open(path) as f:
        return yaml.YAML(typ='safe', pure=True).load(f)


def _dump_yaml(data, path):
    with open(path, 'w') as f:
        yaml.YAML().dump(data, f)


@pytest.mark.parametrize('name', ['Phenol-HFE', 'Ion-HFE', 'Phenol-HFE-Dimer', 'Multi-Property'])
def test_examples_load(example_run, name):
    cfg = config.load(str(example_run(name) / 'config.yaml'))
    assert cfg.hfe_systems or cfg.liquids or cfg.dimers
    assert os.path.isfile(cfg.parameters)
    assert os.path.isfile(cfg.tinker_env)


def test_multi_property_shape(example_run):
    cfg = config.load(str(example_run('Multi-Property') / 'config.yaml'))
    assert [s.name for s in cfg.hfe_systems] == ['phenol', 'sodium']
    assert [q.name for q in cfg.liquids] == ['water_neat']
    assert cfg.liquids[0].temperatures == [298.15, 323.15]
    # Weights are divided by the temperature count so a liquid's total pull on
    # the fit does not grow just because it was measured at more temperatures
    assert cfg.liquids[0].weights == [0.5, 0.5]


def test_md_defaults_deep_merge(example_run):
    cfg = config.load(str(example_run('Multi-Property') / 'config.yaml'))
    phenol, sodium = cfg.hfe_systems
    assert phenol.liquid.total_time == 0.5          # from shared defaults
    assert sodium.liquid.total_time == 0.25         # system override
    # Un-overridden fields still come from the shared defaults
    assert sodium.liquid.time_step == phenol.liquid.time_step
    assert sodium.liquid.pressure == 1.0


def test_small_solute_disables_gas_phase(example_run):
    cfg = config.load(str(example_run('Ion-HFE') / 'config.yaml'))
    sodium = cfg.hfe_systems[0]
    assert sodium.natom == 1
    assert sodium.ignore_gas
    assert sodium.phases == ['liquid']


def test_derived_md_quantities(example_run):
    cfg = config.load(str(example_run('Phenol-HFE') / 'config.yaml'))
    liquid = cfg.hfe_systems[0].liquid
    assert liquid.total_steps == 250000        # 0.5 ns at 2 fs
    assert liquid.total_snapshots == 500       # 0.5 ns at 1 ps
    assert liquid.steps_per_snapshot == 500
    assert liquid.integrator == '4'            # NPT


def test_removed_keys_rejected(tmp_path, example_run):
    run = example_run('Phenol-HFE')
    data = _load_yaml(str(run / 'config.yaml'))
    data['polar_eps'] = 1e-5
    _dump_yaml(data, str(run / 'config.yaml'))
    with pytest.raises(SystemExit, match='polar_eps'):
        config.load(str(run / 'config.yaml'))


def test_duplicate_names_rejected(example_run):
    run = example_run('Multi-Property')
    data = _load_yaml(str(run / 'config.yaml'))
    data['hfe_systems'][1]['name'] = 'phenol'
    _dump_yaml(data, str(run / 'config.yaml'))
    with pytest.raises(SystemExit, match='duplicate'):
        config.load(str(run / 'config.yaml'))


def test_mismatched_density_lists_rejected(example_run):
    run = example_run('Multi-Property')
    data = _load_yaml(str(run / 'config.yaml'))
    data['liquids'][0]['expt_densities'] = [997.0]
    _dump_yaml(data, str(run / 'config.yaml'))
    with pytest.raises(SystemExit, match='expt_densities'):
        config.load(str(run / 'config.yaml'))


def test_too_few_snapshots_rejected(example_run):
    run = example_run('Phenol-HFE')
    data = _load_yaml(str(run / 'config.yaml'))
    # 0.001 ns at 1 ps per frame gives one snapshot; BAR needs at least six
    data['shared']['md_defaults']['liquid']['total_time'] = 0.001
    _dump_yaml(data, str(run / 'config.yaml'))
    with pytest.raises(SystemExit, match='snapshots'):
        config.load(str(run / 'config.yaml'))


def test_missing_input_file_reported(example_run):
    run = example_run('Phenol-HFE')
    os.remove(run / 'input' / 'phenol.xyz')
    with pytest.raises(SystemExit, match='gas_xyz'):
        config.load(str(run / 'config.yaml'))


def test_optimize_requires_params(example_run):
    run = example_run('Phenol-HFE')
    data = _load_yaml(str(run / 'config.yaml'))
    data['job']['type'] = 'optimize'
    _dump_yaml(data, str(run / 'config.yaml'))
    with pytest.raises(SystemExit, match='opt_params'):
        config.load(str(run / 'config.yaml'))


def test_bad_job_type_rejected(example_run):
    run = example_run('Phenol-HFE')
    data = _load_yaml(str(run / 'config.yaml'))
    data['job']['type'] = 'sinlge-point'
    _dump_yaml(data, str(run / 'config.yaml'))
    with pytest.raises(SystemExit, match='job.type'):
        config.load(str(run / 'config.yaml'))


def test_default_denominators(example_run):
    cfg = config.load(str(example_run('Multi-Property') / 'config.yaml'))
    denoms = config.default_denominators(cfg)
    # A single reference value is normalized by sqrt(|value|)
    assert denoms['hfe']['phenol'] == pytest.approx(6.62 ** 0.5)
    assert denoms['hfe']['sodium'] == pytest.approx(88.7 ** 0.5)
    # Several reference values are normalized by their spread
    assert denoms['density']['water_neat'] == pytest.approx(4.5, abs=1e-6)
