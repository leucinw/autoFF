"""Force-field parameter file management for the shared prm.

Every system in a run reads the same parameter file. During an optimization
that file is rewritten once per step, and numbered *sidecars* carry the
finite-difference perturbations. Both are always rebuilt from a pristine
snapshot rather than edited in place, so repeated steps cannot accumulate
duplicate override lines.

Overrides are appended rather than spliced in: Tinker takes the last
definition of a term, so an appended ``vdw 36 3.405 0.11`` supersedes whatever
the body of the file said.
"""

import glob as globmod
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

from .config import ConfigError

# Never let a fitted quantity cross zero: vdW sigma and epsilon become
# unphysical there and Tinker's behavior past that point is meaningless.
MIN_LOWER_BOUND = 1e-4

# Sidecars are numbered _01.._99, and a Jacobian needs 2 per free parameter
# plus one for the trial point.
MAX_SIDECAR_INDEX = 99


@dataclass
class OptEntry:
    """One ``opt_params`` group: a Tinker term and its parameter values."""
    term_idx: str              # e.g. "vdw-36" or "vdwpair-401-402"
    all_params: np.ndarray     # every value, free and fixed
    free_mask: List[bool]
    free_start: int            # offset of this entry's first free value

    @property
    def n_free(self):
        return sum(self.free_mask)

    def render(self, free_params):
        """Return the Tinker override line for this entry."""
        full = self.reconstruct(free_params)
        return self.term_idx.replace('-', '   ') + '  ' + '  '.join(str(p) for p in full)

    def reconstruct(self, free_params):
        """Merge optimizer-controlled values back in with the fixed ones."""
        full = list(self.all_params)
        fi = self.free_start
        for k, is_free in enumerate(self.free_mask):
            if is_free:
                full[k] = free_params[fi]
                fi += 1
        return full


@dataclass
class ParamSpec:
    """The optimizable parameters parsed out of ``job.optimize``."""
    entries: List[OptEntry] = field(default_factory=list)
    initial: np.ndarray = field(default_factory=lambda: np.array([]))
    lower: np.ndarray = field(default_factory=lambda: np.array([]))
    upper: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def n_free(self):
        return len(self.initial)

    def render_lines(self, free_params):
        return [e.render(free_params) for e in self.entries]

    def describe(self, free_params):
        """One-line summary of the current values, marking fixed entries."""
        parts = []
        for e in self.entries:
            full = e.reconstruct(free_params)
            vals = [f"{v:.6g}" + ("" if is_free else "(fixed)")
                    for v, is_free in zip(full, e.free_mask)]
            parts.append(f"{e.term_idx}=[{', '.join(vals)}]")
        return ' | '.join(parts)


def parse_opt_params(opt_params, params_range, log=None):
    """Parse ``opt_params`` / ``params_range`` string pairs into a ParamSpec.

    Each entry looks like ``"vdw-36 3.4050 0.1100"`` with a matching range
    ``"0.20 0.05"``. A range of 0 pins that value: it is written to the
    parameter file unchanged but hidden from the optimizer.
    """
    entries = []
    all_initial, all_lb, all_ub = [], [], []
    free_start = 0

    for op_str, pr_str in zip(opt_params, params_range):
        s = op_str.split()
        if not s:
            raise ConfigError(f"empty opt_params entry: {op_str!r}")
        term_idx = s[0]
        entry_params = np.array([float(x) for x in s[1:]])
        n = len(entry_params)
        if n == 0:
            raise ConfigError(
                f"opt_params entry '{op_str}' has no parameter values after the term key"
            )

        range_vals = [float(v) for v in pr_str.split()]
        if len(range_vals) != n:
            raise ConfigError(
                f"params_range entry '{pr_str}' has {len(range_vals)} value(s) but "
                f"opt_params entry '{op_str}' has {n}; they must match"
            )

        free_mask = [rv != 0.0 for rv in range_vals]
        if sum(free_mask) == 0 and log is not None:
            log.warning(
                "All parameters in '%s' have range=0 and are fixed; the entry is "
                "written unchanged but contributes no free variables.", term_idx
            )

        entry_lb, entry_ub = [], []
        for ep, rv, is_free in zip(entry_params, range_vals, free_mask):
            if is_free:
                entry_lb.append(ep - rv)
                entry_ub.append(ep + rv)
        entry_lb = np.array(entry_lb)
        entry_ub = np.array(entry_ub)

        if len(entry_lb) > 0:
            bad = entry_lb <= 0
            if bad.any():
                if log is not None:
                    log.warning(
                        "params_range for '%s' would drive lb <= 0 at free indices %s; "
                        "clamping to %g.", term_idx, np.where(bad)[0].tolist(), MIN_LOWER_BOUND
                    )
                entry_lb = np.where(bad, MIN_LOWER_BOUND, entry_lb)

        entry = OptEntry(
            term_idx=term_idx,
            all_params=entry_params.copy(),
            free_mask=free_mask,
            free_start=free_start,
        )
        entries.append(entry)
        all_initial.extend(entry_params[np.array(free_mask, dtype=bool)])
        all_lb.extend(entry_lb)
        all_ub.extend(entry_ub)
        free_start += entry.n_free

    spec = ParamSpec(
        entries=entries,
        initial=np.array(all_initial),
        lower=np.array(all_lb),
        upper=np.array(all_ub),
    )
    if spec.n_free == 0:
        raise ConfigError(
            "no free parameters to optimize; set at least one non-zero value in params_range"
        )
    max_needed = 2 * spec.n_free + 1
    if max_needed > MAX_SIDECAR_INDEX:
        raise ConfigError(
            f"{spec.n_free} free parameters need {max_needed} perturbation files but "
            f"only {MAX_SIDECAR_INDEX} sidecar slots exist; fit at most "
            f"{(MAX_SIDECAR_INDEX - 1) // 2} parameters at a time"
        )
    return spec


class ParamFile:
    """The run's working parameter file, its snapshot, and its sidecars.

    All three live under ``<workdir>/prm/`` and are referenced by absolute
    path from every generated key file, so the user's input .prm is never
    touched and cluster nodes resolve them over the shared filesystem.
    """

    def __init__(self, source_prm, prm_dir):
        self.source = str(Path(source_prm).resolve())
        self.dir = str(Path(prm_dir).resolve())
        self.name = os.path.basename(self.source)
        self.working = os.path.join(self.dir, self.name)
        self.snapshot = self.working + '.orig'

    def initialize(self):
        """Create the prm directory and seed the snapshot from the user's file."""
        os.makedirs(self.dir, exist_ok=True)
        # The snapshot is the run's reference copy: seed it once, then leave it
        # alone so a crashed run cannot poison the next one.
        if not os.path.isfile(self.snapshot):
            shutil.copy2(self.source, self.snapshot)
        self.restore()
        return self

    def restore(self):
        """Reset the working file to the pristine snapshot."""
        shutil.copy2(self.snapshot, self.working)

    def sidecar(self, idx):
        """Absolute path of numbered perturbation sidecar *idx*."""
        return os.path.join(self.dir, f"{self.name}_{idx:02d}")

    def scratch(self, suffix):
        """Absolute path of a short-lived local-evaluation parameter file."""
        return os.path.join(self.dir, f"{self.name}.{suffix}")

    def write(self, path, override_lines=()):
        """Write the snapshot plus *override_lines* to *path*."""
        if not os.path.isfile(self.snapshot):
            raise FileNotFoundError(f"Parameter snapshot not found: {self.snapshot}")
        shutil.copy2(self.snapshot, path)
        if override_lines:
            with open(path, 'a') as f:
                for line in override_lines:
                    f.write(line.rstrip('\n') + '\n')
        return path

    def write_final(self, override_lines):
        """Write the optimized parameters to ``<name>.final``."""
        return self.write(self.working + '.final', override_lines)

    def cleanup_sidecars(self, systems_dir=None, max_idx=MAX_SIDECAR_INDEX):
        """Delete every sidecar and the FEP windows derived from it.

        A stale ``FEP_03`` directory from an earlier step would otherwise be
        collected as if it belonged to the current one, silently pairing a
        Jacobian column with the wrong perturbation.
        """
        for path in globmod.glob(os.path.join(self.dir, f"{self.name}_[0-9][0-9]")):
            _remove(path)
        if systems_dir:
            for idx in range(1, max_idx + 1):
                lam = 100 + idx * 10
                for pattern in (
                    os.path.join(systems_dir, '*', '*', f'*e{lam}*'),
                    os.path.join(systems_dir, '*', '*', f'FEP_{idx:02d}'),
                    os.path.join(systems_dir, '*', f'*-prm{idx:02d}-*'),
                ):
                    for path in globmod.glob(pattern):
                        _remove(path)


def _remove(path):
    try:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    except OSError:
        pass
