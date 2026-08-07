"""Single-point evaluation: run every system and collect every property.

This module owns the run as a whole. It builds the system objects, submits all
of their work in one pass, and then drives a single polling loop that advances
each system through its own stages independently — so a fast solute reaches
BAR while a slow one is still running dynamics.

The optimizer calls :meth:`Runner.evaluate` once per objective evaluation, so
whatever happens here defines both the reported properties and the fitted
ones.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from . import tinkerio
from .dimer import DimerOptTarget, DimerTarget
from .dispatch import JobDispatcher
from .hfe import HFESystem
from .liquid import NeatLiquidSystem
from .params import ParamFile

log = logging.getLogger(__name__)


@dataclass
class Evaluation:
    """Every property produced by one pass over the configured systems."""
    hfe: Dict[str, object] = field(default_factory=dict)        # name -> HFEResult
    densities: Dict[str, List[float]] = field(default_factory=dict)
    dimers: Dict[str, float] = field(default_factory=dict)
    dimer_opt: Optional[float] = None


class Runner:
    """Owns the systems for one config and evaluates them together."""

    def __init__(self, cfg, dry_run=False, skip_check=None):
        self.cfg = cfg
        self.dry_run = dry_run
        self.skip_check = cfg.skip_completeness_check if skip_check is None else skip_check

        tinkerio.load_tinker_env(cfg.tinker_env)

        self.param_file = ParamFile(cfg.parameters, cfg.prm_dir)
        os.makedirs(cfg.systems_dir, exist_ok=True)
        os.makedirs(cfg.results_dir, exist_ok=True)

        self.hfe_systems = [
            HFESystem(s, cfg.systems_dir, cfg.tinker_env, self.param_file,
                      skip_check=self.skip_check, verbose=cfg.verbose)
            for s in cfg.hfe_systems
        ]
        self.liquids = [
            NeatLiquidSystem(q, cfg.systems_dir, cfg.tinker_env, self.param_file)
            for q in cfg.liquids
        ]
        self.dimers = [DimerTarget(d, cfg.systems_dir) for d in cfg.dimers]
        self.dimer_opt = DimerOptTarget(cfg.dimer_opt, cfg.systems_dir) \
            if cfg.dimer_opt else None

        manifest = os.path.join(cfg.results_dir, 'submitted_jobs.txt')
        if dry_run and os.path.isfile(manifest):
            os.remove(manifest)
        self.dispatcher = JobDispatcher(nodes=cfg.node_list, dry_run=dry_run,
                                        manifest_path=manifest)

    # -- setup ------------------------------------------------------------

    def setup(self, sidecars=None, minimize=True):
        """Prepare the parameter file and every system's working directory."""
        self.param_file.initialize()
        for system in self.hfe_systems:
            system.set_sidecars(sidecars or [])
            system.setup(dry_run=self.dry_run, minimize=minimize)
        for liquid in self.liquids:
            liquid.setup(dry_run=self.dry_run)
        for dimer in self.dimers:
            dimer.setup(dry_run=self.dry_run)
        if self.dimer_opt:
            self.dimer_opt.setup(self.param_file, dry_run=self.dry_run)
        return self

    # -- evaluation -------------------------------------------------------

    def evaluate(self, prm_path=None, sidecars=None, fresh_liquid=False,
                 collect_hfe=True):
        """Run every system to completion and return the resulting properties.

        *prm_path* selects the parameter file the liquids simulate under
        (defaults to the working file). *sidecars* lists perturbation indices
        that HFE systems should carry as extra reweighting windows.
        """
        prm_path = prm_path or self.param_file.working
        # Window keys encode which parameter file each lambda reads, so they
        # must be rewritten whenever the sidecar set changes. Regenerating is
        # idempotent: minimization is skipped once its log exists.
        for system in self.hfe_systems:
            system.set_sidecars(sidecars or [])
            system.setup(dry_run=self.dry_run)

        for liquid in self.liquids:
            liquid.update_key(prm_path)

        self._run_simulations(fresh_liquid=fresh_liquid)

        if self.dry_run:
            log.info("[dry-run] simulations not executed; skipping collection")
            return Evaluation()

        return self.collect(prm_path=prm_path, collect_hfe=collect_hfe)

    def collect(self, prm_path=None, collect_hfe=True):
        """Read every property out of output files already on disk.

        Runs no simulations and submits nothing, so it is safe to call against
        a finished run to regenerate its report.
        """
        prm_path = prm_path or self.param_file.working
        result = Evaluation()
        if collect_hfe:
            for system in self.hfe_systems:
                res = system.collect()
                result.hfe[system.name] = res
                system.write_report(
                    res, os.path.join(self.cfg.results_dir, f"hfe_{system.name}.txt"))
        for liquid in self.liquids:
            result.densities[liquid.name] = liquid.densities()
        for dimer in self.dimers:
            result.dimers[dimer.name] = dimer.evaluate(prm_path)
        if self.dimer_opt:
            result.dimer_opt = self.dimer_opt.evaluate(prm_path)
        return result

    def _run_simulations(self, fresh_liquid=False):
        """Submit all MD, then advance each system through BAR as it finishes."""
        if not (self.hfe_systems or self.liquids):
            return

        # 'md' -> dynamics running; 'bar' -> BAR running; done systems drop out
        states = {}
        jobs = []
        for system in self.hfe_systems:
            jobs.extend(system.md_jobs())
            states[system.name] = ('hfe', system, 'md')
        for liquid in self.liquids:
            jobs.extend(liquid.md_jobs(fresh=fresh_liquid))
            states[liquid.name] = ('liquid', liquid, 'md')
        self.dispatcher.submit(jobs)

        if self.dry_run:
            # Nothing will ever complete, so also emit the BAR scripts that a
            # real run would produce once dynamics finished.
            bar_jobs = []
            for system in self.hfe_systems:
                bar_jobs.extend(system.bar_jobs())
            self.dispatcher.submit(bar_jobs)
            return

        while states:
            advanced = False
            for name in list(states):
                kind, system, stage = states[name]
                if stage == 'md':
                    if not system.md_complete():
                        continue
                    if kind == 'liquid':
                        log.info("[%s] MD complete", name)
                        del states[name]
                    else:
                        bar_jobs = system.bar_jobs()
                        log.info("[%s] MD complete; submitting %d BAR job(s)",
                                 name, len(bar_jobs))
                        self.dispatcher.submit(bar_jobs)
                        states[name] = (kind, system, 'bar')
                    advanced = True
                elif stage == 'bar':
                    if system.bar_complete():
                        log.info("[%s] BAR analysis complete", name)
                        del states[name]
                        advanced = True
            if not states:
                break
            if not advanced:
                now = datetime.now().strftime("%b %d %Y %H:%M:%S")
                log.info("[%s] waiting on %d system(s): %s",
                         now, len(states), ", ".join(states))
                time.sleep(self.cfg.checking_time)

    def evaluate_gradient(self, sidecars, perturb_map, diff_step):
        """Evaluate every perturbed parameter set in one cluster round.

        Returns ``(hfe_fep, liquid_jacobians)`` where *hfe_fep* maps a solute
        name to ``{sidecar_index: HFE}`` and *liquid_jacobians* maps a liquid
        name to its (n_temperatures, n_params) density derivative block.

        HFE derivatives come from reweighting windows over the existing
        trajectories; density derivatives come from re-analyzing the existing
        production trajectory. Neither runs new dynamics.
        """
        for system in self.hfe_systems:
            system.set_sidecars(sidecars)
            system.setup(dry_run=self.dry_run)

        jobs = []
        for system in self.hfe_systems:
            # Links the reweighting windows to the coupled-state trajectory
            jobs.extend(system.md_jobs())

        analyze_logs = {}
        for liquid in self.liquids:
            liquid_jobs, log_map = liquid.analyze_jobs(sidecars)
            analyze_logs[liquid.name] = log_map
            jobs.extend(liquid_jobs)

        for system in self.hfe_systems:
            jobs.extend(system.bar_jobs())
        self.dispatcher.submit(jobs)

        if self.dry_run:
            return {}, {}

        pending = {s.name: ('hfe', s) for s in self.hfe_systems}
        pending.update({q.name: ('liquid', q) for q in self.liquids})
        while pending:
            for name in list(pending):
                kind, system = pending[name]
                done = system.bar_complete() if kind == 'hfe' \
                    else system.analyze_complete(analyze_logs[name])
                if done:
                    del pending[name]
            if not pending:
                break
            log.info("Gradient: waiting on %d system(s): %s",
                     len(pending), ", ".join(pending))
            time.sleep(self.cfg.checking_time)

        hfe_fep = {s.name: s.collect().fep_values for s in self.hfe_systems}
        liquid_jac = {
            q.name: q.jacobian_columns(analyze_logs[q.name], perturb_map, diff_step)
            for q in self.liquids
        }
        return hfe_fep, liquid_jac

    # -- status / reporting -----------------------------------------------

    def status(self):
        """Print per-system completeness without submitting anything."""
        for system in self.hfe_systems:
            system.set_sidecars([])
            md = system.md_complete()
            bar = system.bar_complete() if md else False
            log.info("HFE  %-20s MD:%-8s BAR:%s", system.name,
                     "done" if md else "pending", "done" if bar else "pending")
        for liquid in self.liquids:
            log.info("LIQ  %-20s MD:%s", liquid.name,
                     "done" if liquid.md_complete() else "pending")
        for dimer in self.dimers:
            log.info("DIM  %-20s local evaluation (no cluster jobs)", dimer.name)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _rows(cfg, ev):
    """Flatten an Evaluation into (label, value, reference, unit) rows."""
    rows = []
    for s in cfg.hfe_systems:
        res = ev.hfe.get(s.name)
        if res is not None:
            rows.append((f"HFE {s.name}", res.fe0, s.expt, "kcal/mol", res.error))
    for q in cfg.liquids:
        values = ev.densities.get(q.name)
        if not values:
            continue
        for i, (T, rho) in enumerate(zip(q.temperatures, values)):
            ref = q.expt_densities[i] if q.expt_densities else None
            rows.append((f"Density {q.name}@{T:.0f}K", rho, ref, "kg/m^3", None))
    for d in cfg.dimers:
        if d.name in ev.dimers:
            rows.append((f"Dimer {d.name}", ev.dimers[d.name], d.expt, "kcal/mol", None))
    if ev.dimer_opt is not None and cfg.dimer_opt is not None:
        rows.append(("Dimer bind (relaxed)", ev.dimer_opt, cfg.dimer_opt.target,
                     "kcal/mol", None))
    return rows


def format_report(cfg, ev):
    """Render the single-point property table as text."""
    rows = _rows(cfg, ev)
    width = (34, 14, 14, 12, 12)
    out = [
        "=" * sum(width),
        f"autoFF single-point results  ({datetime.now():%Y-%m-%d %H:%M:%S})",
        f"parameters: {cfg.parameters}",
        "=" * sum(width),
        (f"{'Property':<{width[0]}}{'Value':>{width[1]}}{'Reference':>{width[2]}}"
         f"{'Diff':>{width[3]}}{'Unit':>{width[4]}}"),
        "-" * sum(width),
    ]
    for label, value, ref, unit, err in rows:
        ref_s = f"{ref:.4f}" if ref is not None else "-"
        diff_s = f"{value - ref:+.4f}" if ref is not None else "-"
        out.append(f"{label:<{width[0]}}{value:>{width[1]}.4f}{ref_s:>{width[2]}}"
                   f"{diff_s:>{width[3]}}{unit:>{width[4]}}")
        if err:
            out.append(f"{'  (BAR error)':<{width[0]}}{err:>{width[1]}.4f}")
    out.append("-" * sum(width))
    return "\n".join(out)


def write_report(cfg, ev):
    """Write the property table and a machine-readable copy into results/."""
    os.makedirs(cfg.results_dir, exist_ok=True)
    text = format_report(cfg, ev)
    txt_path = os.path.join(cfg.results_dir, 'singlepoint.txt')
    with open(txt_path, 'w') as f:
        f.write(text + "\n")

    yaml_path = os.path.join(cfg.results_dir, 'singlepoint.yaml')
    with open(yaml_path, 'w') as f:
        f.write("# autoFF single-point results\n")
        if ev.hfe:
            f.write("hfe:\n")
            for name, res in ev.hfe.items():
                f.write(f"  {name}: {{value: {res.fe0:.6f}, error: {res.error:.6f}}}\n")
        if ev.densities:
            f.write("densities:\n")
            for name, values in ev.densities.items():
                pretty = ", ".join(f"{v:.4f}" for v in values)
                f.write(f"  {name}: [{pretty}]\n")
        if ev.dimers:
            f.write("dimers:\n")
            for name, value in ev.dimers.items():
                f.write(f"  {name}: {value:.6f}\n")
        if ev.dimer_opt is not None:
            f.write(f"dimer_opt: {ev.dimer_opt:.6f}\n")
    return txt_path, text


def run(cfg, dry_run=False, skip_check=None):
    """Execute a single-point job end to end."""
    runner = Runner(cfg, dry_run=dry_run, skip_check=skip_check)
    runner.setup()
    ev = runner.evaluate()
    if dry_run:
        log.info("[dry-run] job scripts written under %s", cfg.systems_dir)
        log.info("[dry-run] submission manifest: %s",
                 os.path.join(cfg.results_dir, 'submitted_jobs.txt'))
        return ev
    _, text = write_report(cfg, ev)
    print(text)
    return ev
