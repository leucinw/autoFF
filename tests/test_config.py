"""Config loading, defaults, and the validation errors users are likely to hit."""

import os

import pytest
import ruamel.yaml as yaml

from autoff import config, tinkerio


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


def _write_txyz(path, natom, body):
    path.write_text(f"    {natom}  test\n" + body)
    return str(path)


def test_gas_phase_decided_by_connectivity_not_atom_count(tmp_path):
    """Atom count cannot tell these apart; the 1-4 pair is what matters."""
    # NH3: 4 atoms, every pair is 1-2 or 1-3 -> gas leg is identically zero
    ammonia = _write_txyz(tmp_path / 'nh3.xyz', 4, (
        "     1  N   0.00  0.00  0.00   27     2     3     4\n"
        "     2  H   1.00  0.00  0.00   28     1\n"
        "     3  H  -0.50  0.87  0.00   28     1\n"
        "     4  H  -0.50 -0.87  0.00   28     1\n"))
    assert not tinkerio.has_intramolecular_nonbonded(ammonia)

    # H-C#C-H: also 4 atoms, but H1...H4 are 1-4 -> gas leg is real
    acetylene = _write_txyz(tmp_path / 'hcch.xyz', 4, (
        "     1  C   0.00  0.00  0.60  123     2     3\n"
        "     2  C   0.00  0.00 -0.60  123     1     4\n"
        "     3  H   0.00  0.00  1.67  124     1\n"
        "     4  H   0.00  0.00 -1.67  124     2\n"))
    assert tinkerio.has_intramolecular_nonbonded(acetylene)


def test_gas_phase_edge_cases(tmp_path):
    # Monatomic ion: nothing to decouple
    ion = _write_txyz(tmp_path / 'na.xyz', 1, "     1  NA  0.00  0.00  0.00  350\n")
    assert not tinkerio.has_intramolecular_nonbonded(ion)

    # Water: 3 atoms, only 1-2 and 1-3
    water = _write_txyz(tmp_path / 'wat.xyz', 3, (
        "     1  O   0.00  0.00  0.00    1     2     3\n"
        "     2  H   0.96  0.00  0.00    2     1\n"
        "     3  H  -0.24  0.93  0.00    2     1\n"))
    assert not tinkerio.has_intramolecular_nonbonded(water)

    # Two separate fragments: nothing excludes their mutual interaction
    pair = _write_txyz(tmp_path / 'two.xyz', 4, (
        "     1  O   0.00  0.00  0.00    1     2\n"
        "     2  H   0.96  0.00  0.00    2     1\n"
        "     3  O   9.00  0.00  0.00    1     4\n"
        "     4  H   9.96  0.00  0.00    2     3\n"))
    assert tinkerio.has_intramolecular_nonbonded(pair)


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


def test_md_stall_timeout_default_and_validation(example_run):
    run = example_run('Multi-Property')
    cfg_path = run / 'config.yaml'
    assert config.load(str(cfg_path)).md_stall_timeout == 3600.0

    text = cfg_path.read_text()
    cfg_path.write_text(text.replace('  checking_time: 120.0',
                                     '  checking_time: 120.0\n  md_stall_timeout: 0'))
    assert config.load(str(cfg_path)).md_stall_timeout == 0.0

    cfg_path.write_text(text.replace('  checking_time: 120.0',
                                     '  checking_time: 120.0\n  md_stall_timeout: -1'))
    with pytest.raises(SystemExit, match='md_stall_timeout'):
        config.load(str(cfg_path))
