"""Dimer targets evaluated locally with gas-phase Tinker calls.

Both targets are cheap enough to run on the driver machine — a handful of
small-cluster single points — so they never touch the job dispatcher and are
evaluated inline during an objective call.

``DimerTarget``
    Interaction energy at a fixed geometry: E_int = E_dimer - E_mon1 - E_mon2.

``DimerOptTarget``
    Binding energy after letting the trial parameters relax the dimer's own
    geometry. The relaxed monomer energies are cached once, which is valid
    only while the fitted terms leave intramolecular energies untouched (true
    for a vdW class whose 1-2 interactions are excluded).
"""

import logging
import os
import subprocess


from . import tinkerio

log = logging.getLogger(__name__)


def _analyze_energy(exe, txyz, key, cwd):
    """Return the total potential energy of *txyz* under *key*."""
    out = subprocess.run([exe, txyz, '-k', key, 'E'],
                         capture_output=True, text=True, cwd=cwd).stdout
    for line in out.splitlines():
        if 'Total Potential Energy' in line:
            for token in line.replace(':', ' ').split():
                try:
                    return float(token)
                except ValueError:
                    continue
    raise RuntimeError(f"analyze produced no energy for {txyz}\n{out[-400:]}")


def _write_key(path, prm_file):
    with open(path, 'w') as f:
        f.write(f"parameters {os.path.abspath(prm_file)}\npolar-eps 0.00001\n")
    return path


class DimerTarget:
    """Interaction energy of one dimer at a fixed geometry."""

    def __init__(self, cfg, workdir, analyze_exe=None):
        self.cfg = cfg
        self.name = cfg.name
        self.dir = os.path.join(workdir, cfg.name)
        self.analyze = analyze_exe or os.environ.get('ANALYZE8')
        self.dimer_xyz = os.path.join(self.dir, f"{self.name}_dimer.xyz")
        self.mon1_xyz = os.path.join(self.dir, f"{self.name}_mon1.xyz")
        self.mon2_xyz = os.path.join(self.dir, f"{self.name}_mon2.xyz")

    def setup(self, dry_run=False):
        """Split the dimer into monomers once; geometry is parameter-independent."""
        os.makedirs(self.dir, exist_ok=True)
        if not self.analyze:
            raise RuntimeError(
                f"dimer '{self.name}': $ANALYZE8 is not set. Check that "
                "shared.tinker_env points at a valid Tinker environment file."
            )
        atoms = tinkerio.read_txyz_atoms(self.cfg.xyz)
        mon1, mon2 = tinkerio.split_dimer_monomers(atoms, self.cfg.frag1_natoms)
        tinkerio.write_txyz_atoms(self.dimer_xyz, atoms, f"{self.name} dimer")
        tinkerio.write_txyz_atoms(self.mon1_xyz, mon1, f"{self.name} mon1")
        tinkerio.write_txyz_atoms(self.mon2_xyz, mon2, f"{self.name} mon2")
        return self

    def evaluate(self, prm_file):
        """Return E_int in kcal/mol under *prm_file*."""
        key = _write_key(os.path.join(self.dir, 'dimer.key'), prm_file)
        e_dimer = _analyze_energy(self.analyze, self.dimer_xyz, key, self.dir)
        e_mon1 = _analyze_energy(self.analyze, self.mon1_xyz, key, self.dir)
        e_mon2 = _analyze_energy(self.analyze, self.mon2_xyz, key, self.dir)
        return e_dimer - e_mon1 - e_mon2


class DimerOptTarget:
    """Binding energy of a dimer relaxed under the trial parameters."""

    def __init__(self, cfg, workdir, minimize_exe=None, analyze_exe=None):
        self.cfg = cfg
        self.name = 'dimer_opt'
        self.dir = os.path.join(workdir, self.name)
        self.minimize = minimize_exe or os.environ.get('MINIMIZE8')
        self.analyze = analyze_exe or os.environ.get('ANALYZE8')
        self.start_xyz = os.path.join(self.dir, 'dopt_dimer.xyz')
        self.mon1_xyz = os.path.join(self.dir, 'dopt_mon1.xyz')
        self.mon2_xyz = os.path.join(self.dir, 'dopt_mon2.xyz')
        self.e_monomers = None

    def setup(self, param_file, dry_run=False):
        """Write the starting geometries and cache the relaxed monomer energies."""
        os.makedirs(self.dir, exist_ok=True)
        if not (self.minimize and self.analyze):
            raise RuntimeError(
                "dimer_opt: $MINIMIZE8/$ANALYZE8 are not set. Check that "
                "shared.tinker_env points at a valid Tinker environment file."
            )
        atoms = tinkerio.read_txyz_atoms(self.cfg.start_xyz)
        mon1, mon2 = tinkerio.split_dimer_monomers(atoms, self.cfg.frag1_natoms)
        tinkerio.write_txyz_atoms(self.start_xyz, atoms, "dimeropt start")
        tinkerio.write_txyz_atoms(self.mon1_xyz, mon1, "mon1 start")
        tinkerio.write_txyz_atoms(self.mon2_xyz, mon2, "mon2 start")

        if dry_run:
            return self

        # Monomer energies are cached under the pristine parameters: the fitted
        # terms are assumed not to change intramolecular energy.
        key = _write_key(os.path.join(self.dir, 'dopt.key'), param_file.snapshot)
        e1 = _analyze_energy(self.analyze, self._minimize(self.mon1_xyz, key), key, self.dir)
        e2 = _analyze_energy(self.analyze, self._minimize(self.mon2_xyz, key), key, self.dir)
        self.e_monomers = e1 + e2
        log.info("[dimer_opt] cached relaxed monomer energy %.4f kcal/mol", self.e_monomers)
        return self

    def _minimize(self, start_xyz, key):
        # Tinker appends _2/_3 rather than overwriting, so a leftover file from
        # the previous step would be picked up as this step's result.
        for stale in (start_xyz + '_2', start_xyz + '_3'):
            if os.path.exists(stale):
                os.remove(stale)
        subprocess.run([self.minimize, start_xyz, '-k', key, str(self.cfg.grad)],
                       capture_output=True, text=True, cwd=self.dir)
        optimized = start_xyz + '_2'
        if not os.path.isfile(optimized):
            raise RuntimeError(f"minimize produced no {optimized}")
        return optimized

    def evaluate(self, prm_file):
        """Relax the dimer under *prm_file* and return its binding energy."""
        key = _write_key(os.path.join(self.dir, 'dopt.key'), prm_file)
        relaxed = self._minimize(self.start_xyz, key)
        return _analyze_energy(self.analyze, relaxed, key, self.dir) - self.e_monomers
