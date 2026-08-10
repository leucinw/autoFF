"""Joint least-squares fit of one shared parameter file to all targets.

Every property with a reference value contributes one entry to a single
residual vector::

    residual = weight * (calculated - reference) / denominator

The denominator normalizes across units so that a density in kg/m^3 and a free
energy in kcal/mol pull on the fit comparably. Residuals are ordered by
config: HFE systems, then each liquid's temperatures, then dimers, then the
relaxed-dimer binding energy.

Derivatives never rerun dynamics. An HFE column is a central difference over
reweighting windows evaluated on the existing trajectories; a density column
uses the fluctuation formula over re-analyzed frames; dimer columns are
central differences on local energy evaluations.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from .config import default_denominators
from .params import parse_opt_params
from .singlepoint import Runner

log = logging.getLogger(__name__)


@dataclass
class Target:
    """One fitted property: where its value comes from and how it is scaled."""
    kind: str            # hfe | density | dimer | dimeropt
    system: str
    label: str
    reference: float
    weight: float
    denom: float
    temp_index: Optional[int] = None

    def residual(self, value):
        return self.weight * (value - self.reference) / self.denom


def build_targets(cfg):
    """Collect every property that has a reference value, in residual order."""
    denoms = default_denominators(cfg)
    targets = []

    for s in cfg.hfe_systems:
        if s.expt is None:
            continue
        targets.append(Target('hfe', s.name, f"HFE {s.name}", s.expt,
                              s.weight, denoms['hfe'][s.name]))
    for q in cfg.liquids:
        if not q.expt_densities:
            continue
        for i, T in enumerate(q.temperatures):
            targets.append(Target('density', q.name, f"Density {q.name}@{T:.0f}K",
                                  q.expt_densities[i], q.weights[i],
                                  denoms['density'][q.name], temp_index=i))
    for d in cfg.dimers:
        if d.expt is None:
            continue
        targets.append(Target('dimer', d.name, f"Dimer {d.name}", d.expt,
                              d.weight, denoms['dimer']))
    if cfg.dimer_opt is not None and cfg.dimer_opt.target is not None:
        targets.append(Target('dimeropt', 'dimer_opt', "Dimer bind (relaxed)",
                              cfg.dimer_opt.target, cfg.dimer_opt.weight,
                              denoms['dimeropt']))
    return targets


def _values_from(evaluation, targets):
    """Pull the calculated value for each target out of an Evaluation."""
    values = []
    for t in targets:
        if t.kind == 'hfe':
            values.append(evaluation.hfe[t.system].fe0)
        elif t.kind == 'density':
            values.append(evaluation.densities[t.system][t.temp_index])
        elif t.kind == 'dimer':
            values.append(evaluation.dimers[t.system])
        else:
            values.append(evaluation.dimer_opt)
    return np.array(values, dtype=float)


class Optimizer:
    """Drives scipy's least_squares over the configured targets."""

    def __init__(self, cfg, dry_run=False, skip_check=None):
        self.cfg = cfg
        self.runner = Runner(cfg, dry_run=dry_run, skip_check=skip_check)
        self.spec = parse_opt_params(cfg.optimize.opt_params,
                                     cfg.optimize.params_range, log=log)
        self.targets = build_targets(cfg)
        if not self.targets:
            raise SystemExit("[Error] no targets with reference values to fit")
        self.diff_step = cfg.optimize.diff_step
        self.step = 0
        self.best_cost = np.inf
        self.param_file = self.runner.param_file

    # -- helpers ----------------------------------------------------------

    def _is_initial(self, params):
        return np.array_equal(np.asarray(params), self.spec.initial)

    def _step_bounds(self):
        """The configured bounds, narrowed to a relative box around the start.

        ``params_range`` is written in absolute units, so it hands out very
        uneven freedom: a range of 0.15 on an rmin of 4.19 is 4%, while 0.02 on
        a well depth of 0.0217 lets it fall to 0.0017. A well depth near zero
        removes the repulsion that keeps induced dipoles finite, so the fit
        does not come back with a bad density -- the dynamics diverge and the
        step yields nothing at all. Capping the relative move keeps every
        parameter somewhere the simulations can still be integrated.
        """
        x0 = np.asarray(self.spec.initial, dtype=float)
        lower = np.asarray(self.spec.lower, dtype=float)
        upper = np.asarray(self.spec.upper, dtype=float)
        max_step = self.cfg.optimize.max_step
        if max_step <= 0:
            return lower, upper
        # A parameter sitting at zero has no relative scale to cap; leave it be.
        room = max_step * np.abs(x0)
        capped = room > 0
        lower = np.where(capped, np.maximum(lower, x0 - room), lower)
        upper = np.where(capped, np.minimum(upper, x0 + room), upper)
        return lower, upper

    def _x_scale(self):
        """Characteristic size of each parameter, for the trust region.

        least_squares measures its trust region in units of ``x_scale``. Left
        at 1 the region is absolute, so it is sized by the largest parameters
        in the vector -- here rmin and chgpen values of 3-7 -- and a step that
        nudges those moves a well depth of 0.02 by its whole magnitude. Scaling
        by each parameter's own size makes the region relative, so every
        parameter moves by a comparable fraction of itself.
        """
        x0 = np.abs(np.asarray(self.spec.initial, dtype=float))
        span = np.asarray(self.spec.upper) - np.asarray(self.spec.lower)
        scale = np.where(x0 > 0, x0, span)
        return np.where(scale > 0, scale, 1.0)

    def _write_params(self, params, path):
        return self.param_file.write(path, self.spec.render_lines(params))

    def _log_table(self, params, values, residuals):
        col = (30, 14, 14, 13, 14)
        log.info("--- Step %d | %s ---", self.step, self.spec.describe(params))
        log.info(f"{'Property':<{col[0]}}{'Target':>{col[1]}}{'Current':>{col[2]}}"
                 f"{'Diff':>{col[3]}}{'WtNormRes':>{col[4]}}")
        sep = "-" * sum(col)
        log.info(sep)
        for t, value, res in zip(self.targets, values, residuals):
            log.info(f"{t.label:<{col[0]}}{t.reference:>{col[1]}.4f}{value:>{col[2]}.4f}"
                     f"{value - t.reference:>{col[3]}.4f}{res:>{col[4]}.4f}")
        log.info(sep)
        log.info(f"{'Sum of squared residuals':<{col[0]}}"
                 f"{float(np.dot(residuals, residuals)):>{sum(col) - col[0]}.6f}")
        log.info(sep)

    # -- objective --------------------------------------------------------

    def model_func(self, params):
        """Residual vector at *params*."""
        self.step += 1
        is_initial = self._is_initial(params)

        if is_initial:
            # The reference point runs at the working parameters, giving each
            # solute a true BAR free energy rather than a reweighted one.
            self._write_params(params, self.param_file.working)
            prm_path, sidecars = self.param_file.working, []
        else:
            self.param_file.cleanup_sidecars(self.cfg.systems_dir)
            prm_path = self._write_params(params, self.param_file.sidecar(1))
            sidecars = [1]

        ev = self.runner.evaluate(prm_path=prm_path, sidecars=sidecars,
                                  fresh_liquid=not is_initial)

        # A trial point's HFE is the reweighted value from its FEP window
        if not is_initial:
            for name, result in ev.hfe.items():
                if 1 not in result.fep_values:
                    raise RuntimeError(
                        f"[{name}] trial step produced no FEP_01 reweighting value")
                result.fe0 = result.fep_values[1]

        values = _values_from(ev, self.targets)
        residuals = np.array([t.residual(v) for t, v in zip(self.targets, values)])
        self._log_table(params, values, residuals)

        cost = float(np.dot(residuals, residuals))
        if cost < self.best_cost:
            log.info("Cost improved (%.6f < %.6f)", cost, self.best_cost)
            self.best_cost = cost
        else:
            log.info("Cost did not improve (%.6f >= %.6f)", cost, self.best_cost)
        return residuals

    # -- derivatives ------------------------------------------------------

    def jacobian(self, params):
        """Jacobian at *params*, one row per target and one column per parameter."""
        params = np.atleast_1d(np.asarray(params, dtype=float))
        n_params = len(params)
        J = np.zeros((len(self.targets), n_params))
        is_initial = self._is_initial(params)

        needs_cluster = any(t.kind in ('hfe', 'density') for t in self.targets)
        perturb_map, sidecars = {}, []
        if needs_cluster:
            # On a trial step sidecar 01 is still the model point, so the
            # finite-difference pairs start at 02.
            idx = 1 if is_initial else 2
            for j in range(n_params):
                plus, minus = idx, idx + 1
                for target_idx, delta in ((plus, +self.diff_step), (minus, -self.diff_step)):
                    shifted = params.copy()
                    shifted[j] += delta
                    self._write_params(shifted, self.param_file.sidecar(target_idx))
                perturb_map[j] = (plus, minus)
                sidecars.extend((plus, minus))
                idx += 2

            hfe_fep, liquid_jac = self.runner.evaluate_gradient(
                sidecars, perturb_map, self.diff_step)

        for row, t in enumerate(self.targets):
            if t.kind == 'hfe':
                feps = hfe_fep[t.system]
                for j in range(n_params):
                    plus, minus = perturb_map[j]
                    if plus not in feps or minus not in feps:
                        raise RuntimeError(
                            f"[{t.system}] gradient is missing reweighting windows "
                            f"{plus}/{minus}; expected {sorted(sidecars)}, "
                            f"got {sorted(feps)}"
                        )
                    J[row, j] = t.weight * (feps[plus] - feps[minus]) \
                        / (2.0 * self.diff_step) / t.denom
            elif t.kind == 'density':
                J[row, :] = t.weight * liquid_jac[t.system][t.temp_index, :] / t.denom

        # Dimer targets are cheap enough to differentiate directly
        local = [t for t in self.targets if t.kind in ('dimer', 'dimeropt')]
        if local:
            prm_p = self.param_file.scratch('fd_p')
            prm_m = self.param_file.scratch('fd_m')
            for j in range(n_params):
                plus, minus = params.copy(), params.copy()
                plus[j] += self.diff_step
                minus[j] -= self.diff_step
                self._write_params(plus, prm_p)
                self._write_params(minus, prm_m)
                for row, t in enumerate(self.targets):
                    if t.kind == 'dimer':
                        target = next(d for d in self.runner.dimers if d.name == t.system)
                        e_p, e_m = target.evaluate(prm_p), target.evaluate(prm_m)
                    elif t.kind == 'dimeropt':
                        e_p = self.runner.dimer_opt.evaluate(prm_p)
                        e_m = self.runner.dimer_opt.evaluate(prm_m)
                    else:
                        continue
                    J[row, j] = t.weight * (e_p - e_m) / (2.0 * self.diff_step) / t.denom
            for path in (prm_p, prm_m):
                try:
                    os.remove(path)
                except OSError:
                    pass
        return J

    # -- driver -----------------------------------------------------------

    def run(self):
        """Set up, fit, and write the optimized parameter file."""
        self.runner.setup()
        self.param_file.cleanup_sidecars(self.cfg.systems_dir)
        self.param_file.restore()

        lower, upper = self._step_bounds()
        self._log_settings(lower, upper)

        result = least_squares(
            fun=self.model_func,
            x0=self.spec.initial,
            jac=self.jacobian,
            loss='soft_l1',
            method='trf',
            verbose=2,
            bounds=(lower, upper),
            x_scale=self._x_scale(),
            ftol=self.cfg.optimize.ftol,
            gtol=self.cfg.optimize.gtol,
            xtol=self.cfg.optimize.xtol,
        )

        log.info("=== Optimization results ===")
        log.info("Success: %s", result.success)
        log.info("Message: %s", result.message)
        log.info("Optimal parameters: %s", self.spec.describe(result.x))
        log.info("Cost (sum of squared residuals): %.6f", 2 * result.cost)

        final = self.param_file.write_final(self.spec.render_lines(result.x))
        log.info("Optimized parameters written to %s", final)
        return result

    def _log_settings(self, lower, upper):
        log.info("=== Optimization settings ===")
        log.info("diff_step: %g", self.diff_step)
        max_step = self.cfg.optimize.max_step
        log.info("max_step: %s", f"{max_step:g} of each starting value"
                 if max_step > 0 else "disabled (params_range only)")
        log.info("parameter groups (%d):", len(self.spec.entries))
        for e in self.spec.entries:
            parts, fi = [], 0
            for p, is_free in zip(e.all_params, e.free_mask):
                if is_free:
                    i = e.free_start + fi
                    # Flag the bounds max_step tightened, so it is visible when
                    # a fit stops moving because of the cap rather than the data
                    tight = '*' if (lower[i] > self.spec.lower[i]
                                    or upper[i] < self.spec.upper[i]) else ''
                    parts.append(f"{p:.6g} [{lower[i]:.6g}, {upper[i]:.6g}]{tight}")
                    fi += 1
                else:
                    parts.append(f"{p:.6g}(fixed)")
            log.info("  %s: %s", e.term_idx, ", ".join(parts))
        log.info("targets (%d):", len(self.targets))
        for t in self.targets:
            log.info("  %-30s ref=%-12.4f weight=%-8g denom=%.4g",
                     t.label, t.reference, t.weight, t.denom)


def run(cfg, dry_run=False, skip_check=None):
    """Execute an optimize job end to end."""
    optimizer = Optimizer(cfg, dry_run=dry_run, skip_check=skip_check)
    if dry_run:
        # Generate every input file for the reference point without fitting
        optimizer.runner.setup()
        optimizer.runner.evaluate()
        log.info("[dry-run] optimization inputs generated; no fitting performed")
        return None
    return optimizer.run()
