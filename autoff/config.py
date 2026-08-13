"""Master configuration: parsing, validation and path resolution.

One YAML file describes an entire run: the shared force field and cluster
settings, any number of HFE solutes, any number of neat liquids, dimer
targets, and which job to perform. Every path in the file is interpreted
relative to the config file's own directory, so a config is portable as long
as it travels with its inputs.

Loading a config resolves it fully — reading atom counts out of coordinate
files, expanding MD defaults into per-system settings, and computing derived
step/snapshot counts — so that downstream modules never re-read the YAML.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import ruamel.yaml as yaml

from . import tinkerio

# Settings keys that existed in the old format and no longer do. Rejecting
# them loudly beats silently ignoring a key the user believes is in effect.
REMOVED_KEYS = {
    'polar_eps': "polar_eps was never read by any code path; set polar-eps in the key file instead",
    'gas_xyz': "top-level gas_xyz/box_xyz moved into the 'hfe_systems' list",
    'box_xyz': "top-level gas_xyz/box_xyz moved into the 'hfe_systems' list",
    'expt_hfe': "expt_hfe moved into the 'hfe_systems' list as 'expt'",
    'expt_density': "expt_density moved into the 'liquids' list as 'expt_densities'",
    'expt_densities': "expt_densities moved into the 'liquids' list",
    'liquid_dir': "liquid_dir is now derived from the liquid's 'name'",
    'dimer_data': "dimer_data moved into the 'dimers' list",
    'lambda_window': "lambda_window moved into each entry of 'hfe_systems'",
}

# BAR discards the first 20% of a trajectory as equilibration
# (startsnapshot = total/5 + 1), so a window needs at least 6 snapshots to
# leave a non-empty sampling range.
MIN_SNAPSHOTS = 6

MIN_BOX_LENGTH = 30.0


class ConfigError(SystemExit):
    """Raised for any user-facing configuration problem."""

    def __init__(self, message):
        super().__init__(f"{tinkerio.RED}[Config error] {message}{tinkerio.ENDC}")


@dataclass
class LiquidMD:
    """Condensed-phase MD settings for one HFE system's liquid leg."""
    total_time: float          # ns
    time_step: float           # fs
    write_freq: float          # ps
    temperature: float         # K
    pressure: float            # atm
    ensemble: str              # NPT | NVT

    @property
    def total_steps(self):
        return int((1000000.0 * self.total_time) / self.time_step)

    @property
    def total_snapshots(self):
        return int(1000 * self.total_time / self.write_freq)

    @property
    def steps_per_snapshot(self):
        return int(round(self.write_freq / self.time_step * 1000))

    @property
    def integrator(self):
        return "4" if self.ensemble == "NPT" else "2"


@dataclass
class GasMD:
    """Gas-phase MD settings for one HFE system's vacuum leg."""
    total_time: float          # ns; 0 disables the gas phase
    time_step: float           # fs
    write_freq: float          # ps
    temperature: float         # K

    @property
    def total_steps(self):
        return int((1000000.0 * self.total_time) / self.time_step)

    @property
    def total_snapshots(self):
        return int(1000 * self.total_time / self.write_freq)

    @property
    def steps_per_snapshot(self):
        return int(round(self.write_freq / self.time_step * 1000))


@dataclass
class HFESystemConfig:
    """One solute whose hydration free energy is computed by BAR."""
    name: str
    gas_xyz: str
    box_xyz: str
    expt: Optional[float]
    weight: float
    denom: Optional[float]
    lambda_window: str
    lambda_file: str
    copy_arc_for_perturb: bool
    manual_ele_scale: bool
    liquid_key: str
    gas_key: str
    liquid: LiquidMD
    gas: GasMD
    natom: int
    orderparams: List[List[float]] = field(default_factory=list)

    @property
    def ignore_gas(self):
        return self.gas.total_time == 0.0

    @property
    def phases(self):
        return ['liquid'] if self.ignore_gas else ['gas', 'liquid']


@dataclass
class LiquidConfig:
    """One neat liquid simulated at one or more temperatures."""
    name: str
    box_xyz: str
    key: Optional[str]
    temperatures: List[float]
    expt_densities: List[float]
    weights: List[float]
    denom: Optional[float]
    equil_time: float          # ns
    production_time: float     # ns
    time_step: float           # fs
    write_freq: float          # ps
    pressure: float            # atm

    @property
    def n_equil(self):
        return round(self.equil_time * 1000.0 / self.write_freq)

    @property
    def n_production(self):
        return round(self.production_time * 1000.0 / self.write_freq)

    @property
    def steps_per_frame(self):
        return round(self.write_freq * 1000.0 / self.time_step)

    @property
    def total_steps(self):
        return (self.n_equil + self.n_production) * self.steps_per_frame

    @property
    def betas(self):
        return [1.0 / (tinkerio.KB * T) for T in self.temperatures]


@dataclass
class DimerConfig:
    """One dimer geometry with a reference interaction energy."""
    name: str
    xyz: str
    frag1_natoms: int
    expt: Optional[float]
    weight: float


@dataclass
class DimerOptConfig:
    """A dimer relaxed under the trial parameters, fitted on binding energy."""
    start_xyz: str
    frag1_natoms: int
    target: Optional[float]
    grad: float
    weight: float
    denom: Optional[float]


@dataclass
class OptimizeConfig:
    """Least-squares settings; only read when job.type is 'optimize'."""
    opt_params: List[str]
    params_range: List[str]
    diff_step: float = 1e-4
    ftol: float = 1e-4
    gtol: float = 1e-4
    xtol: float = 1e-4
    # Largest fractional move any one parameter may make away from its starting
    # value, applied on top of params_range. Those ranges are absolute and so
    # grant very uneven freedom -- 0.15 on an rmin near 4.2 is 4%, while 0.02 on
    # a well depth of 0.0217 is 92% -- and it is the loose ones that let a fit
    # walk into parameters no MD can integrate. 0 disables the cap.
    max_step: float = 0.25


@dataclass
class Config:
    """A fully resolved run configuration."""
    workdir: str
    parameters: str            # user's source prm (absolute)
    tinker_env: str
    node_list: List[str]
    checking_time: float
    md_stall_timeout: float    # s of output silence before MD counts as dead
    verbose: int
    skip_completeness_check: bool
    job_type: str              # single-point | optimize
    hfe_systems: List[HFESystemConfig]
    liquids: List[LiquidConfig]
    dimers: List[DimerConfig]
    dimer_opt: Optional[DimerOptConfig]
    optimize: Optional[OptimizeConfig]

    @property
    def prm_dir(self):
        return os.path.join(self.workdir, 'prm')

    @property
    def systems_dir(self):
        return os.path.join(self.workdir, 'systems')

    @property
    def results_dir(self):
        return os.path.join(self.workdir, 'results')

    def system_dir(self, name):
        return os.path.join(self.systems_dir, name)

    @property
    def has_targets(self):
        """True when at least one property has a reference value to fit."""
        return bool(
            [s for s in self.hfe_systems if s.expt is not None]
            or [q for q in self.liquids if q.expt_densities]
            or [d for d in self.dimers if d.expt is not None]
            or (self.dimer_opt is not None and self.dimer_opt.target is not None)
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _deep_merge(base, override):
    """Merge *override* into a copy of *base*, recursing into nested dicts."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve(base_dir, path):
    """Resolve a config-relative path to an absolute one."""
    if path is None:
        return None
    p = Path(os.path.expanduser(str(path)))
    if not p.is_absolute():
        p = Path(base_dir) / p
    return str(p.resolve())


def _require_file(path, what):
    if not os.path.isfile(path):
        raise ConfigError(f"{what} not found: {path}")
    return path


def _as_float_list(value, what):
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        value = [value]
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        raise ConfigError(f"{what} must be a number or list of numbers, got {value!r}")


def _check_removed_keys(settings):
    found = [k for k in settings if k in REMOVED_KEYS]
    if found:
        details = "\n".join(f"  - {k}: {REMOVED_KEYS[k]}" for k in found)
        raise ConfigError(
            "this config uses keys from the old single-molecule format:\n"
            f"{details}\n"
            "See README.md for the current schema."
        )


def _resolve_lambda_file(base_dir, window):
    """Map a lambda_window value to a schedule file path."""
    name = str(window).strip()
    if name.upper() == 'COURSER':
        return tinkerio.package_data('orderparams_courser')
    if name.upper() == 'DEFAULT':
        return tinkerio.package_data('orderparams_default')
    path = _resolve(base_dir, name)
    return _require_file(path, f"lambda_window file '{window}'")


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------

def _parse_hfe_system(entry, idx, base_dir, md_defaults, shared_prm):
    if 'name' not in entry:
        raise ConfigError(f"hfe_systems[{idx}] is missing 'name'")
    name = str(entry['name'])
    for required in ('gas_xyz', 'box_xyz'):
        if required not in entry:
            raise ConfigError(f"hfe_systems '{name}' is missing '{required}'")

    gas_xyz = _require_file(_resolve(base_dir, entry['gas_xyz']), f"'{name}' gas_xyz")
    box_xyz = _require_file(_resolve(base_dir, entry['box_xyz']), f"'{name}' box_xyz")

    md = _deep_merge(md_defaults, entry.get('md') or {})
    liq_md, gas_md = md.get('liquid') or {}, md.get('gas') or {}

    missing = [k for k in ('total_time', 'time_step', 'write_freq', 'temperature',
                           'pressure', 'ensemble') if k not in liq_md]
    if missing:
        raise ConfigError(
            f"hfe_systems '{name}': liquid MD settings missing {missing}. "
            "Provide them in shared.md_defaults.liquid or the system's own 'md.liquid'."
        )
    liquid = LiquidMD(
        total_time=float(liq_md['total_time']),
        time_step=float(liq_md['time_step']),
        write_freq=float(liq_md['write_freq']),
        temperature=float(liq_md['temperature']),
        pressure=float(liq_md['pressure']),
        ensemble=str(liq_md['ensemble']).upper(),
    )
    if liquid.ensemble not in ('NPT', 'NVT'):
        raise ConfigError(f"hfe_systems '{name}': ensemble must be NPT or NVT, got {liquid.ensemble}")
    if liquid.total_snapshots < MIN_SNAPSHOTS:
        raise ConfigError(
            f"hfe_systems '{name}': liquid total_time ({liquid.total_time} ns) / "
            f"write_freq ({liquid.write_freq} ps) yields only {liquid.total_snapshots} "
            f"snapshots; BAR needs at least {MIN_SNAPSHOTS}"
        )

    missing = [k for k in ('total_time', 'time_step', 'write_freq', 'temperature')
               if k not in gas_md]
    if missing:
        raise ConfigError(
            f"hfe_systems '{name}': gas MD settings missing {missing}. "
            "Provide them in shared.md_defaults.gas or set total_time: 0 to disable the gas phase."
        )
    gas = GasMD(
        total_time=float(gas_md['total_time']),
        time_step=float(gas_md['time_step']),
        write_freq=float(gas_md['write_freq']),
        temperature=float(gas_md['temperature']),
    )
    if gas.total_time > 0 and gas.total_snapshots < MIN_SNAPSHOTS:
        raise ConfigError(
            f"hfe_systems '{name}': gas total_time ({gas.total_time} ns) / "
            f"write_freq ({gas.write_freq} ps) yields only {gas.total_snapshots} "
            f"snapshots; BAR needs at least {MIN_SNAPSHOTS}"
        )

    # A box smaller than the vdW/Ewald cutoffs gives unphysical solvation
    box = tinkerio.read_txyz_box(box_xyz)
    if box is None:
        print(tinkerio.YELLOW + f"[Warning] No box info in {box_xyz}; "
              "it must be supplied in the liquid key instead." + tinkerio.ENDC)
    elif min(box) < MIN_BOX_LENGTH:
        raise ConfigError(
            f"hfe_systems '{name}': box {box} is smaller than "
            f"{MIN_BOX_LENGTH} A in at least one dimension"
        )

    natom = tinkerio.read_txyz_natoms(gas_xyz)
    # AMOEBA excludes intramolecular vdW/multipole interactions between 1-2 and
    # 1-3 neighbours, so a solute with no 1-4 pair has nothing for the gas leg
    # to decouple -- its free energy is identically zero and sampling it only
    # costs wall time. Decided on connectivity, not atom count: NH3 has four
    # atoms and no 1-4 pair (gas leg correctly skipped), while H-C#C-H has four
    # atoms and one (gas leg is real and must be run).
    if gas.total_time != 0.0 and not tinkerio.has_intramolecular_nonbonded(gas_xyz):
        print(tinkerio.YELLOW + f" [Warning] '{name}': solute has {natom} atoms and no "
              "1-4 pair; disabling the gas phase" + tinkerio.ENDC)
        gas.total_time = 0.0

    lambda_window = str(entry.get('lambda_window', 'default'))
    lambda_file = _resolve_lambda_file(base_dir, lambda_window)

    liquid_key = entry.get('liquid_key')
    liquid_key = _require_file(_resolve(base_dir, liquid_key), f"'{name}' liquid_key") \
        if liquid_key else tinkerio.package_data('liquid.key')
    gas_key = entry.get('gas_key')
    gas_key = _require_file(_resolve(base_dir, gas_key), f"'{name}' gas_key") \
        if gas_key else tinkerio.package_data('gas.key')

    expt = entry.get('expt')
    denom = entry.get('denom')
    return HFESystemConfig(
        name=name,
        gas_xyz=gas_xyz,
        box_xyz=box_xyz,
        expt=float(expt) if expt is not None else None,
        weight=float(entry.get('weight', 1.0)),
        denom=float(denom) if denom is not None else None,
        lambda_window=lambda_window,
        lambda_file=lambda_file,
        copy_arc_for_perturb=bool(entry.get('copy_arc_for_perturb', True)),
        manual_ele_scale=bool(entry.get('manual_ele_scale', False)),
        liquid_key=liquid_key,
        gas_key=gas_key,
        liquid=liquid,
        gas=gas,
        natom=natom,
        orderparams=tinkerio.read_order_params(lambda_file),
    )


def _parse_liquid(entry, idx, base_dir, md_defaults):
    if 'name' not in entry:
        raise ConfigError(f"liquids[{idx}] is missing 'name'")
    name = str(entry['name'])
    if 'box_xyz' not in entry:
        raise ConfigError(f"liquids '{name}' is missing 'box_xyz'")
    box_xyz = _require_file(_resolve(base_dir, entry['box_xyz']), f"liquid '{name}' box_xyz")

    temperatures = _as_float_list(entry.get('temperatures'), f"liquids '{name}' temperatures")
    if not temperatures:
        raise ConfigError(f"liquids '{name}': at least one temperature is required")

    expt_densities = _as_float_list(entry.get('expt_densities'),
                                    f"liquids '{name}' expt_densities")
    if expt_densities and len(expt_densities) != len(temperatures):
        raise ConfigError(
            f"liquids '{name}': {len(temperatures)} temperature(s) but "
            f"{len(expt_densities)} expt_densities; they must match"
        )

    weights = _as_float_list(entry.get('weights'), f"liquids '{name}' weights")
    if weights and len(weights) != len(temperatures):
        raise ConfigError(
            f"liquids '{name}': {len(weights)} weight(s) but "
            f"{len(temperatures)} temperature(s); they must match"
        )
    if not weights:
        weights = [1.0] * len(temperatures)
    # Normalize by temperature count so adding temperatures does not inflate
    # this liquid's share of the total cost.
    weights = [w / len(temperatures) for w in weights]

    md = _deep_merge((md_defaults or {}).get('liquid') or {}, entry.get('md') or {})
    if 'production_time' not in entry:
        raise ConfigError(f"liquids '{name}' is missing 'production_time'")

    key = entry.get('key')
    key = _require_file(_resolve(base_dir, key), f"liquid '{name}' key") if key else None
    denom = entry.get('denom')

    cfg = LiquidConfig(
        name=name,
        box_xyz=box_xyz,
        key=key,
        temperatures=temperatures,
        expt_densities=expt_densities,
        weights=weights,
        denom=float(denom) if denom is not None else None,
        equil_time=float(entry.get('equil_time', 0.02)),
        production_time=float(entry['production_time']),
        time_step=float(md.get('time_step', 2.0)),
        write_freq=float(md.get('write_freq', 0.1)),
        pressure=float(md.get('pressure', 1.0)),
    )
    if cfg.n_production < 1:
        raise ConfigError(
            f"liquids '{name}': production_time ({cfg.production_time} ns) / "
            f"write_freq ({cfg.write_freq} ps) yields no production frames"
        )
    return cfg


def _parse_dimer(entry, idx, base_dir):
    if 'xyz' not in entry:
        raise ConfigError(f"dimers[{idx}] is missing 'xyz'")
    xyz = _require_file(_resolve(base_dir, entry['xyz']), f"dimers[{idx}] xyz")
    name = str(entry.get('name') or Path(xyz).stem)
    n1 = int(entry.get('frag1_natoms', 0))
    if n1 <= 0:
        raise ConfigError(f"dimers '{name}': frag1_natoms must be a positive integer")
    natoms = len(tinkerio.read_txyz_atoms(xyz))
    if n1 >= natoms:
        raise ConfigError(
            f"dimers '{name}': frag1_natoms ({n1}) must be less than the "
            f"dimer's {natoms} atoms"
        )
    expt = entry.get('expt')
    return DimerConfig(
        name=name,
        xyz=xyz,
        frag1_natoms=n1,
        expt=float(expt) if expt is not None else None,
        weight=float(entry.get('weight', 1.0)),
    )


def _parse_dimer_opt(entry, base_dir):
    if 'start_xyz' not in entry:
        raise ConfigError("dimer_opt is missing 'start_xyz'")
    start = _require_file(_resolve(base_dir, entry['start_xyz']), "dimer_opt start_xyz")
    n1 = int(entry.get('frag1_natoms', 0))
    if n1 <= 0:
        raise ConfigError("dimer_opt: frag1_natoms must be a positive integer")
    target = entry.get('target')
    denom = entry.get('denom')
    return DimerOptConfig(
        start_xyz=start,
        frag1_natoms=n1,
        target=float(target) if target is not None else None,
        grad=float(entry.get('grad', 0.01)),
        weight=float(entry.get('weight', 1.0)),
        denom=float(denom) if denom is not None else None,
    )


def _parse_optimize(job, n_free_hint=None):
    opt = job.get('optimize') or {}
    raw_params = opt.get('opt_params')
    raw_range = opt.get('params_range')
    if raw_params is None or raw_range is None:
        raise ConfigError(
            "job.type is 'optimize' but job.optimize.opt_params / params_range are missing"
        )
    if isinstance(raw_params, str):
        raw_params = [raw_params]
    if isinstance(raw_range, str):
        raw_range = [raw_range]
    if len(raw_params) != len(raw_range):
        raise ConfigError(
            f"opt_params has {len(raw_params)} entry(ies) but params_range has "
            f"{len(raw_range)}; they must match"
        )
    max_step = float(opt.get('max_step', 0.25))
    if max_step < 0:
        raise ConfigError(f"job.optimize.max_step must be >= 0, got {max_step}")
    return OptimizeConfig(
        opt_params=[str(p) for p in raw_params],
        params_range=[str(r) for r in raw_range],
        diff_step=float(opt.get('diff_step', 1e-4)),
        ftol=float(opt.get('ftol', 1e-4)),
        gtol=float(opt.get('gtol', 1e-4)),
        xtol=float(opt.get('xtol', 1e-4)),
        max_step=max_step,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load(config_path):
    """Load, validate and fully resolve a master config file."""
    config_path = os.path.abspath(os.path.expanduser(config_path))
    if not os.path.isfile(config_path):
        raise ConfigError(f"config file not found: {config_path}")
    base_dir = os.path.dirname(config_path)

    parser = yaml.YAML(typ='safe', pure=True)
    with open(config_path) as f:
        settings = parser.load(f)
    if not isinstance(settings, dict):
        raise ConfigError(f"{config_path} does not contain a YAML mapping")

    _check_removed_keys(settings)

    shared = settings.get('shared') or {}
    _check_removed_keys(shared)

    workdir = _resolve(base_dir, settings.get('workdir', '.'))

    if 'parameters' not in shared:
        raise ConfigError("shared.parameters (the force field .prm file) is required")
    parameters = _require_file(_resolve(base_dir, shared['parameters']), "shared.parameters")

    tinker_env = shared.get('tinker_env')
    tinker_env = _require_file(_resolve(base_dir, tinker_env), "shared.tinker_env") \
        if tinker_env else tinkerio.package_data('tinker.env')

    # A job that dies without writing anything to its log -- evicted, OOM-killed,
    # or lost with its node -- is invisible to the crash parser, and since
    # nothing in this pipeline resubmits, the poller then waits on a frame count
    # that will never move again. An hour of total silence from a run that
    # writes a frame every few seconds is proof enough. 0 disables the check.
    md_stall_timeout = float(shared.get('md_stall_timeout', 3600.0))
    if md_stall_timeout < 0:
        raise ConfigError(
            f"shared.md_stall_timeout must be >= 0 (0 disables), got {md_stall_timeout}")

    md_defaults = shared.get('md_defaults') or {}

    hfe_entries = settings.get('hfe_systems') or []
    liquid_entries = settings.get('liquids') or []
    dimer_entries = settings.get('dimers') or []

    hfe_systems = [_parse_hfe_system(e, i, base_dir, md_defaults, parameters)
                   for i, e in enumerate(hfe_entries)]
    liquids = [_parse_liquid(e, i, base_dir, md_defaults)
               for i, e in enumerate(liquid_entries)]
    dimers = [_parse_dimer(e, i, base_dir) for i, e in enumerate(dimer_entries)]
    dimer_opt = _parse_dimer_opt(settings['dimer_opt'], base_dir) \
        if settings.get('dimer_opt') else None

    if not (hfe_systems or liquids or dimers or dimer_opt):
        raise ConfigError(
            "nothing to do: define at least one entry under 'hfe_systems', "
            "'liquids', or 'dimers' (or a 'dimer_opt' block)"
        )

    # System names become directory names, so collisions would silently make
    # two systems share trajectories.
    names = [s.name for s in hfe_systems] + [q.name for q in liquids] + [d.name for d in dimers]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ConfigError(f"duplicate system name(s) across hfe_systems/liquids/dimers: {dupes}")

    job = settings.get('job') or {}
    job_type = str(job.get('type', 'single-point')).lower()
    if job_type not in ('single-point', 'optimize'):
        raise ConfigError(f"job.type must be 'single-point' or 'optimize', got {job_type!r}")

    optimize = _parse_optimize(job) if job_type == 'optimize' else None

    cfg = Config(
        workdir=workdir,
        parameters=parameters,
        tinker_env=tinker_env,
        node_list=[str(n) for n in (shared.get('node_list') or [])],
        checking_time=float(shared.get('checking_time', 60.0)),
        md_stall_timeout=md_stall_timeout,
        verbose=int(shared.get('verbose', 1)),
        skip_completeness_check=bool(shared.get('skip_completeness_check', False)),
        job_type=job_type,
        hfe_systems=hfe_systems,
        liquids=liquids,
        dimers=dimers,
        dimer_opt=dimer_opt,
        optimize=optimize,
    )

    if job_type == 'optimize' and not cfg.has_targets:
        raise ConfigError(
            "job.type is 'optimize' but no reference values were given. Add "
            "'expt' to an hfe_system or dimer, or 'expt_densities' to a liquid."
        )

    return cfg


def default_denominators(cfg):
    """Compute the ForceBalance-style scale normalizers for every target.

    A residual is ``weight * (calc - reference) / denom``. The denominator
    defaults to the spread of the reference values when several exist, and to
    ``sqrt(|value|)`` for a lone reference, so targets measured in different
    units contribute comparably to the cost.
    """
    denoms = {'hfe': {}, 'density': {}, 'dimer': None, 'dimeropt': None}

    for s in cfg.hfe_systems:
        if s.expt is None:
            continue
        if s.denom is not None:
            denoms['hfe'][s.name] = s.denom
        else:
            denoms['hfe'][s.name] = float(np.sqrt(abs(s.expt))) if s.expt != 0.0 else 1.0

    for q in cfg.liquids:
        if not q.expt_densities:
            continue
        if q.denom is not None:
            denoms['density'][q.name] = q.denom
        elif len(q.expt_densities) > 1:
            denoms['density'][q.name] = float(np.std(q.expt_densities))
        else:
            denoms['density'][q.name] = float(np.sqrt(abs(q.expt_densities[0])))

    qms = [d.expt for d in cfg.dimers if d.expt is not None]
    if qms:
        denoms['dimer'] = float(np.sqrt(np.mean(np.square(qms))))

    if cfg.dimer_opt is not None and cfg.dimer_opt.target is not None:
        denoms['dimeropt'] = cfg.dimer_opt.denom if cfg.dimer_opt.denom is not None \
            else float(np.sqrt(abs(cfg.dimer_opt.target)))

    for group, table in (('hfe', denoms['hfe']), ('density', denoms['density'])):
        for name, value in table.items():
            if value <= 0:
                raise ConfigError(f"{group} denominator for '{name}' must be positive, got {value}")
    return denoms
