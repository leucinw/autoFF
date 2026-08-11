"""Hydration free energy of one solute via Tinker BAR.

A solute is decoupled from water along a schedule of electrostatic and van der
Waals lambda windows. Each window runs its own MD trajectory; consecutive
windows are then combined with BAR, and the per-window free energies sum to
the transfer free energy. The gas-phase leg is subtracted to remove the
solute's intramolecular contribution, and is skipped entirely for species
small enough to have none.

Perturbed parameter sets ride along as extra windows at fictitious lambdas
above 1.0. Because those windows reuse the fully-coupled trajectory, an
alternative parameter set costs one BAR evaluation instead of a full
simulation — which is what makes the optimizer's finite differences
affordable.
"""

import glob
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from . import tinkerio
from .dispatch import Job
from .elescale import scaledownele

log = logging.getLogger(__name__)

# BAR discards the leading fifth of each trajectory as equilibration
EQUIL_FRACTION = 5.0


@dataclass
class HFEResult:
    """Free energies collected from one solute's BAR analysis."""
    name: str
    fe0: float                                  # kcal/mol, unperturbed
    error: float
    windows: List = field(default_factory=list)  # (state0, state1, dG, err)
    fep_values: Dict[int, float] = field(default_factory=dict)  # sidecar idx -> HFE


class HFESystem:
    """Drives setup, MD, BAR and collection for a single solute."""

    def __init__(self, cfg, workdir, tinker_env, param_file,
                 skip_check=False, verbose=1, stall_timeout=0.0):
        self.cfg = cfg
        self.name = cfg.name
        self.dir = os.path.join(workdir, cfg.name)
        self.tinker_env = tinker_env
        self.param_file = param_file
        self.skip_check = skip_check
        self.verbose = verbose
        self.stall_timeout = stall_timeout
        self.sidecars = []
        self.orderparams = list(cfg.orderparams)

    # -- layout -----------------------------------------------------------

    @property
    def phases(self):
        return self.cfg.phases

    def phase_dir(self, phase):
        return os.path.join(self.dir, phase)

    def phase_xyz(self, phase):
        """Canonical coordinate file: a private copy, so inputs stay pristine."""
        return os.path.join(self.dir, f"{phase}.xyz")

    def _source_xyz(self, phase):
        return self.cfg.gas_xyz if phase == 'gas' else self.cfg.box_xyz

    def _md(self, phase):
        return self.cfg.gas if phase == 'gas' else self.cfg.liquid

    def _exe(self, phase, kind):
        # Tinker9 (GPU) runs the condensed phase; Tinker8 handles the small
        # gas-phase system, where GPU offload is not worth the launch overhead.
        suffix = '8' if phase == 'gas' else '9'
        return f"${kind}{suffix}"

    def set_sidecars(self, indices):
        """Declare which perturbation sidecars become extra lambda windows."""
        self.sidecars = sorted(indices or [])
        self.orderparams = list(self.cfg.orderparams)
        for idx in self.sidecars:
            fake = tinkerio.lambda_from_fep_index(idx)
            self.orderparams.append([fake, fake])
        return self

    # -- setup ------------------------------------------------------------

    def setup(self, dry_run=False, minimize=True):
        """Create directories, minimize coordinates, and write every window key.

        Pass ``minimize=False`` to prepare the directory without running
        Tinker, which is what collecting results from a finished run needs.
        """
        os.makedirs(self.dir, exist_ok=True)
        for phase in self.phases:
            os.makedirs(self.phase_dir(phase), exist_ok=True)
            self._prepare_coordinates(phase, dry_run or not minimize)
            self._write_window_keys(phase)
        log.info("[%s] BAR input files generated (%d windows/phase, phases=%s)",
                 self.name, len(self.orderparams), ",".join(self.phases))
        return self

    def _prepare_coordinates(self, phase, dry_run):
        """Copy the input coordinates in and energy-minimize them once."""
        xyz = self.phase_xyz(phase)
        if not os.path.isfile(xyz):
            shutil.copy2(self._source_xyz(phase), xyz)

        keyfile = os.path.join(self.dir, f"{phase}.key")
        template = self.cfg.gas_key if phase == 'gas' else self.cfg.liquid_key
        with open(template) as f:
            lines = f.readlines()
        with open(keyfile, 'w') as f:
            for line in lines:
                if 'parameters' in line.lower():
                    line = f'parameters     {self.param_file.working}\n'
                f.write(line)

        logfile = os.path.join(self.dir, f"{phase}-min.log")
        shfile = os.path.join(self.dir, f"{phase}-min.sh")
        rms_grad = '0.1' if phase == 'gas' else '0.2'
        with open(shfile, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write(f'source {self.tinker_env}\n')
            f.write(f'{self._exe(phase, "MINIMIZE")} {os.path.basename(xyz)} '
                    f'-key {phase}.key {rms_grad} > {os.path.basename(logfile)}\n')
            f.write(f'wait\nmv {os.path.basename(xyz)}_2 {os.path.basename(xyz)}\n')

        # The log doubles as a marker: its presence means these coordinates
        # were already minimized, so a resumed run reuses them as-is.
        if dry_run or os.path.isfile(logfile):
            return
        rc = subprocess.run(['bash', shfile], cwd=self.dir).returncode
        if rc != 0 or not os.path.isfile(xyz):
            raise RuntimeError(
                f"[{self.name}] {phase} minimization failed (rc={rc}); see {logfile}")

    def _write_window_keys(self, phase):
        """Write one key per lambda window and link its coordinates."""
        phasedir = self.phase_dir(phase)
        # Stale keys from a different lambda schedule would be picked up by the
        # BAR stage, which globs the directory rather than the schedule.
        for ext in ('xyz', 'key'):
            for path in glob.glob(os.path.join(phasedir, f"*.{ext}")):
                os.remove(path)

        template = self.cfg.gas_key if phase == 'gas' else self.cfg.liquid_key
        with open(template) as f:
            keylines = f.readlines()

        for elb, vlb in self.orderparams:
            fname = tinkerio.format_lambda_name(phase, elb, vlb)
            is_perturbed = elb * vlb > 1.0
            if is_perturbed and elb != vlb:
                raise RuntimeError(
                    f"[{self.name}] perturbed window lambdas disagree: {elb} vs {vlb}")

            prm = self.param_file.sidecar(tinkerio.fep_index_from_lambda(elb)) \
                if is_perturbed else self.param_file.working

            # A perturbed window is a reweighting of the fully-coupled state,
            # so it keeps lambda=1 and varies only the parameter file.
            eff_elb = 1.0 if is_perturbed else elb
            eff_vlb = 1.0 if is_perturbed else vlb

            keypath = os.path.join(phasedir, fname + ".key")
            with open(keypath, 'w') as fw:
                for line in keylines:
                    if 'parameters' in line.lower():
                        line = f'parameters     {prm}\n'
                    fw.write(line)
                fw.write('\n')
                fw.write(f'ligand -1 {self.cfg.natom}\n')
                if self.cfg.manual_ele_scale:
                    fw.write('ele-lambda 1.0\n')
                else:
                    fw.write(f'ele-lambda {eff_elb}\n')
                fw.write(f'vdw-lambda {eff_vlb}\n')
                if self.cfg.manual_ele_scale:
                    # Atom types come from the solute-only file so solvent
                    # parameters are left alone.
                    scaled = scaledownele(self.phase_xyz('gas'), prm, eff_elb)
                    fw.write(f'\n# electrostatic parameters scaled by {eff_elb}\n')
                    for s in scaled:
                        fw.write(f'{s}\n')

            tinkerio.force_symlink(self.phase_xyz(phase),
                                   os.path.join(phasedir, f"{fname}.xyz"))

    # -- MD ---------------------------------------------------------------

    def md_jobs(self):
        """Return the MD jobs still needed, resuming partial trajectories."""
        jobs = []
        for phase in self.phases:
            phasedir = self.phase_dir(phase)
            md = self._md(phase)
            for path in glob.glob(os.path.join(phasedir, "*.sh")):
                if not os.path.basename(path).startswith('bar_'):
                    os.remove(path)

            for elb, vlb in self.orderparams:
                fname = tinkerio.format_lambda_name(phase, elb, vlb)
                arcpath = os.path.join(phasedir, fname + ".arc")

                # A perturbed window reweights the coupled-state trajectory,
                # so it needs no dynamics of its own.
                if (elb * vlb > 1.0) and self.cfg.copy_arc_for_perturb:
                    src = os.path.join(phasedir, f"{phase}-e100-v100.arc")
                    tinkerio.force_symlink(src, arcpath)
                    continue

                remaining, existing = self._remaining_steps(arcpath, md)
                if remaining <= 0:
                    if self.verbose > 0:
                        log.info("[%s] %s: already has %d snapshots, skipping",
                                 self.name, fname, md.total_snapshots)
                    continue

                dynpath = os.path.join(phasedir, fname + ".dyn")
                errpath = os.path.join(phasedir, fname + ".err")
                resuming = existing > 0 and os.path.isfile(dynpath)
                if resuming and os.path.isfile(errpath):
                    log.error("[%s] %s: .err file present, refusing to resume; "
                              "inspect %s first", self.name, fname, errpath)
                    continue
                if resuming and self.verbose > 0:
                    log.info("[%s] %s: resuming at %d/%d snapshots (%d steps left)",
                             self.name, fname, existing, md.total_snapshots, remaining)
                elif existing > 0 and not resuming and self.verbose > 0:
                    log.warning("[%s] %s: arc has %d snapshots but no .dyn; restarting",
                                self.name, fname, existing)

                sh_name = self._write_md_script(phase, fname, remaining, resuming)
                jobs.append(Job(
                    script=sh_name, workdir=phasedir,
                    queue='CPU' if phase == 'gas' else 'GPU',
                    nproc=4 if phase == 'gas' else 2,
                    label=f"{self.name}/{fname} md",
                ))
        return jobs

    def _remaining_steps(self, arcpath, md):
        """Return (steps still needed, snapshots already present)."""
        if not os.path.isfile(arcpath):
            return md.total_steps, 0
        existing = tinkerio.count_arc_frames(arcpath)
        # Tinker restarts from the .dyn checkpoint and appends to the arc, so
        # only the outstanding steps need to be requested.
        remaining = md.total_steps - existing * md.steps_per_snapshot
        return max(0, remaining), existing

    def _write_md_script(self, phase, fname, steps, resuming):
        md = self._md(phase)
        sh_name = fname + '.sh'
        redirect = '>>' if resuming else '>'
        exe = self._exe(phase, 'DYNAMIC')
        with open(os.path.join(self.phase_dir(phase), sh_name), 'w') as f:
            f.write(f'source {self.tinker_env}\n')
            if phase == 'gas':
                f.write(f"{exe} {fname}.xyz -key {fname}.key {steps} {md.time_step} "
                        f"{md.write_freq} 2 {md.temperature} {redirect} {fname}.log\n")
            elif md.ensemble == 'NPT':
                f.write(f"{exe} {fname}.xyz -key {fname}.key {steps} {md.time_step} "
                        f"{md.write_freq} 4 {md.temperature} {md.pressure} "
                        f"{redirect} {fname}.log\n")
            else:
                f.write(f"{exe} {fname}.xyz -key {fname}.key {steps} {md.time_step} "
                        f"{md.write_freq} 2 {md.temperature} {redirect} {fname}.log\n")
        return sh_name

    def md_complete(self):
        """True once every window that runs dynamics has a full trajectory.

        Raises RuntimeError for windows whose output has stopped arriving.
        With one lambda schedule per phase a single lost window is enough to
        hold up the whole solute, and a job killed with its node leaves nothing
        in its log to distinguish it from one that is merely slow.
        """
        if self.skip_check:
            return True
        done = True
        stalled = []
        for phase in self.phases:
            md = self._md(phase)
            for elb, vlb in self.orderparams:
                if (elb * vlb > 1.0) and self.cfg.copy_arc_for_perturb:
                    continue
                fname = tinkerio.format_lambda_name(phase, elb, vlb)
                arcpath = os.path.join(self.phase_dir(phase), fname + ".arc")
                if not os.path.isfile(arcpath):
                    if phase == 'gas' and md.total_snapshots == 0:
                        continue
                    if self.verbose > 0:
                        log.info("[%s] %s.arc missing", self.name, fname)
                    done = False
                    continue
                n = tinkerio.count_arc_frames(arcpath)
                if n < md.total_snapshots:
                    logpath = os.path.join(self.phase_dir(phase), fname + ".log")
                    reason = tinkerio.stall_reason(self.stall_timeout, arcpath, logpath)
                    if reason:
                        stalled.append(
                            f"{fname} ({n}/{md.total_snapshots} snapshots): {reason}")
                    elif self.verbose > 0:
                        pct = int(n / md.total_snapshots * 100) if md.total_snapshots else 0
                        log.info("[%s] %s: %d/%d snapshots (%d%%)",
                                 self.name, fname, n, md.total_snapshots, pct)
                    done = False
        if stalled:
            raise RuntimeError(
                f"[{self.name}] {len(stalled)} MD window(s) stopped producing output "
                f"and will not finish on their own:\n  " + "\n  ".join(stalled) +
                f"\nThese usually died with their node rather than blowing up. "
                f"Rerunning each window's .sh in {self.dir} resumes it."
            )
        return done

    # -- BAR --------------------------------------------------------------

    def _window_pairs(self, phase):
        """Yield the consecutive lambda pairs BAR combines, with their layout."""
        md = self._md(phase)
        start = int(md.total_snapshots / EQUIL_FRACTION) + 1
        for (elb0, vlb0), (elb1, vlb1) in zip(self.orderparams, self.orderparams[1:]):
            # Every perturbed window is compared against the coupled state,
            # not against the preceding perturbation.
            if elb1 > 1.0:
                elb0, vlb0 = 1.0, 1.0
            e0, v0 = f"{round(elb0 * 100):03d}", f"{round(vlb0 * 100):03d}"
            e1, v1 = f"{round(elb1 * 100):03d}", f"{round(vlb1 * 100):03d}"
            fname0 = f"{phase}-e{e0}-v{v0}"
            fname1 = f"{phase}-e{e1}-v{v1}"

            is_fep = int(e1) > 100 and int(v1) > 100
            idx = tinkerio.fep_index_from_lambda(elb1) if is_fep else None
            bardir = os.path.join(self.phase_dir(phase), f"FEP_{idx:02d}") if is_fep \
                else self.phase_dir(phase)

            if phase == 'gas':
                # The gas leg is integrated in the opposite direction
                stem, sh_name = fname1, f"bar_e{e1}-v{v1}_e{e0}-v{v0}.sh"
                states = (fname1, fname0)
            else:
                stem, sh_name = fname0, f"bar_e{e0}-v{v0}_e{e1}-v{v1}.sh"
                states = (fname0, fname1)

            yield {
                'phase': phase, 'fname0': fname0, 'fname1': fname1,
                'stem': stem, 'sh_name': sh_name, 'states': states,
                'bardir': bardir, 'is_fep': is_fep, 'fep_index': idx,
                'start': start, 'total': md.total_snapshots,
            }

    def bar_jobs(self):
        """Write and return the BAR jobs that still need to run."""
        jobs = []
        for phase in self.phases:
            md = self._md(phase)
            if phase == 'gas' and md.total_time == 0:
                continue
            for w in self._window_pairs(phase):
                if w['is_fep']:
                    self._link_fep_window(w)

                barpath = os.path.join(w['bardir'], w['stem'] + ".bar")
                enepath = os.path.join(w['bardir'], w['stem'] + ".ene")
                shpath = os.path.join(w['bardir'], w['sh_name'])

                if not self.skip_check:
                    self._drop_stale_bar(w, barpath, enepath, shpath)

                if self._bar_done(barpath, enepath, shpath):
                    if self.verbose > 0:
                        log.info("[%s] %s.ene already complete", self.name, w['stem'])
                    continue

                self._write_bar_script(w, shpath)
                jobs.append(Job(
                    script=w['sh_name'], workdir=w['bardir'],
                    queue='CPU' if phase == 'gas' else 'GPU',
                    nproc=4 if phase == 'gas' else 2,
                    label=f"{self.name}/{w['stem']} bar",
                ))
        return jobs

    def _link_fep_window(self, w):
        """Populate a FEP_NN directory with links to the trajectories it reweights."""
        os.makedirs(w['bardir'], exist_ok=True)
        phasedir = self.phase_dir(w['phase'])
        arc0 = w['fname0'] + ".arc"
        arc1 = w['fname1'] + ".arc"
        tinkerio.force_symlink(os.path.join(phasedir, arc0),
                               os.path.join(w['bardir'], arc0))
        for keyname in (w['fname0'] + ".key", w['fname1'] + ".key"):
            tinkerio.force_symlink(os.path.join(phasedir, keyname),
                                   os.path.join(w['bardir'], keyname))
        # With copy_arc_for_perturb the perturbed state has no trajectory of
        # its own: both sides of the comparison read the coupled-state arc.
        src = arc0 if self.cfg.copy_arc_for_perturb else arc1
        tinkerio.force_symlink(os.path.join(phasedir, src),
                               os.path.join(w['bardir'], arc1))

    def _drop_stale_bar(self, w, barpath, enepath, shpath):
        """Delete .bar/.ene left over from a run with different sampling."""
        reason = None
        if os.path.isfile(barpath) and os.path.isfile(shpath) and \
                not tinkerio.bar_sh_steps_match(shpath, w['start'], w['total']):
            reason = "snapshot range changed"
        elif os.path.isfile(barpath):
            # Both states must be full length: a BAR that ran against a
            # trajectory still being written leaves the second block short,
            # and its .ene comes out as NaN that no amount of waiting fixes.
            counts = tinkerio.bar_file_snapshot_counts(barpath)
            if counts != (w['total'], w['total']):
                states = " and ".join(str(c) for c in counts)
                reason = f"built from {states} snapshots, expected {w['total']} each"
        if reason:
            log.warning("[%s] %s: %s; removing stale .bar/.ene",
                        self.name, w['stem'], reason)
            for p in (barpath, enepath):
                if os.path.isfile(p):
                    os.remove(p)

    def _bar_done(self, barpath, enepath, shpath):
        if not (os.path.isfile(barpath) and os.path.isfile(shpath)):
            return False
        # skip_check means the user has vouched for these files; existence is
        # then enough and the (expensive) content scan is skipped.
        return os.path.isfile(enepath) if self.skip_check else tinkerio.ene_complete(enepath)

    def _write_bar_script(self, w, shpath):
        phase = w['phase']
        md = self._md(phase)
        exe = self._exe(phase, 'BAR')
        T = md.temperature
        stem = w['stem']
        # Step 1 builds the .bar energy matrix from both trajectories; step 2
        # solves it over the post-equilibration window.
        first, second = (w['fname1'], w['fname0']) if phase == 'gas' \
            else (w['fname0'], w['fname1'])
        with open(shpath, 'w') as f:
            f.write(f"source {self.tinker_env}\n")
            f.write(f"{exe} 1 {first}.arc {T} {second}.arc {T} N > {stem}.out && \n")
            f.write(f"{exe} 2 {stem}.bar {w['start']} {w['total']} 1 "
                    f"{w['start']} {w['total']} 1 > {stem}.ene \n")

    def bar_complete(self):
        """True once every BAR window has a converged .ene file."""
        if self.skip_check:
            return True
        pending = []
        for enepath, _, _ in self._ene_plan():
            if not tinkerio.ene_complete(enepath):
                pending.append(os.path.basename(enepath))
        if pending:
            if self.verbose > 0:
                log.info("[%s] %d BAR window(s) pending: %s", self.name, len(pending),
                         ", ".join(pending[:4]) + (" ..." if len(pending) > 4 else ""))
            return False
        return True

    def _ene_plan(self):
        """Return [(enepath, states, fep_index)] in result-reporting order."""
        plan, seen = [], set()
        for phase in self.phases:
            md = self._md(phase)
            if phase == 'gas' and md.total_time == 0:
                continue
            for w in self._window_pairs(phase):
                enepath = os.path.join(w['bardir'], w['stem'] + ".ene")
                if enepath in seen:
                    continue
                seen.add(enepath)
                plan.append((enepath, w['states'], w['fep_index']))
        return plan

    # -- results ----------------------------------------------------------

    def collect(self):
        """Read every .ene file and assemble this solute's free energies."""
        base_windows, fep_by_index = [], {}
        for enepath, states, fep_index in self._ene_plan():
            if not os.path.isfile(enepath):
                raise RuntimeError(f"[{self.name}] missing BAR output: {enepath}")
            value = tinkerio.read_free_energy(enepath)
            if value is None:
                raise RuntimeError(f"[{self.name}] no free energy found in {enepath}")
            fe, err = value
            if fep_index is None:
                base_windows.append((states[0], states[1], fe, err))
            else:
                # Each perturbation contributes one window per phase; the legs
                # add because gas and liquid are independent transformations.
                acc = fep_by_index.setdefault(fep_index, [0.0, 0.0])
                acc[0] += fe
                acc[1] += err ** 2

        fe0 = float(np.sum([w[2] for w in base_windows]))
        err0 = float(np.sqrt(np.sum([w[3] ** 2 for w in base_windows])))

        # A perturbed state's HFE is the reference plus the reweighting term.
        fep_values = {idx: fe0 + acc[0] for idx, acc in fep_by_index.items()}

        return HFEResult(name=self.name, fe0=fe0, error=err0,
                         windows=base_windows, fep_values=fep_values)

    def write_report(self, result, path):
        """Write the per-window free energy table for one solute."""
        with open(path, 'w') as fo:
            fo.write(f"# Hydration free energy for '{self.name}'\n")
            header = "%20s%20s%30s%20s" % (
                "StateA", "StateB", "FreeEnergy(kcal/mol)", "Error(kcal/mol)")
            fo.write(header + "\n")
            for state0, state1, fe, err in result.windows:
                fo.write("%20s%20s%25.4f%20.4f\n" % (state0, state1, fe, err))
            fo.write("%40s%25.4f%20.4f\n" % (
                "SUM OF THE TOTAL FREE ENERGY (FE0)", result.fe0, result.error))
            for idx in sorted(result.fep_values):
                fo.write("    FEP_%03d%45.4f\n" % (idx, result.fep_values[idx]))
        return path
