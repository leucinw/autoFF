"""Pure file I/O helpers for Tinker inputs and outputs.

Everything here is a free function over paths: no configuration state, no
cluster access, no subprocess dispatch beyond ``wc -l``. That keeps the
parsers unit-testable against the trajectory fixtures in ``tests/fixtures``.
"""

import os
import re
import subprocess
import time
from collections import deque
from pathlib import Path

import numpy as np

RED = '\033[91m'
ENDC = '\033[0m'
GREEN = '\033[92m'
YELLOW = '\33[93m'

# kB in kcal/(mol K)
KB = 0.001987204

# rho (kg/m^3) = M (g/mol) / (DENSITY_FACTOR * V (A^3))
DENSITY_FACTOR = 0.0006022140857

# Keywords valid in an HFE key file that must be absent from a neat-liquid key.
# Tinker keywords are case-insensitive; match at the start of a line.
HFE_ONLY_RE = re.compile(
    r'^\s*(vdw-annihilate|vdw-lambda|ele-lambda|ligand)\b',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def force_symlink(src, dst):
    """Create a symlink at *dst*, replacing whatever is already there."""
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def read_last_lines(filepath, n=5):
    """Read the last *n* lines of a file using a fixed-size deque."""
    with open(filepath) as f:
        return deque(f, maxlen=n)


# ---------------------------------------------------------------------------
# Lambda window naming
# ---------------------------------------------------------------------------

def format_lambda_name(phase, elb, vlb):
    """Format a lambda window name from phase and lambda values."""
    return f"{phase}-e{round(elb * 100):03d}-v{round(vlb * 100):03d}"


def fep_index_from_lambda(elb):
    """Return the FEP sidecar index encoded in a perturbed lambda value.

    Perturbation *i* is represented as the fake lambda 1.0 + 0.1*i, so a
    window at elb=1.2 belongs to sidecar 02 and directory ``FEP_02``.
    """
    return int(round(elb * 100) / 10) - 10


def lambda_from_fep_index(idx):
    """Inverse of :func:`fep_index_from_lambda`."""
    return round(1.0 + idx * 0.1, 1)


def read_order_params(path):
    """Read a two-column lambda schedule file into [[elb, vlb], ...]."""
    orderparams = []
    with open(path) as f:
        for line in f:
            if "#" in line:
                continue
            d = line.split()
            if len(d) >= 2:
                orderparams.append([float(d[0]), float(d[1])])
    return orderparams


# ---------------------------------------------------------------------------
# Trajectory (.arc) helpers
# ---------------------------------------------------------------------------

def count_arc_frames(file_path):
    """Return the number of frames in a Tinker .arc file, or 0 if unreadable.

    The per-frame stride is n_atoms+1 for an NVT trajectory and n_atoms+2 for
    NPT, where the extra line holds the box dimensions. The frame count comes
    from ``wc -l``, which is far faster than iterating in Python over the
    multi-gigabyte trajectories these runs produce.
    """
    if not os.path.isfile(file_path):
        return 0
    try:
        with open(file_path, 'rb') as f:
            first_line = f.readline().decode(errors='replace').split()
            if not first_line:
                return 0
            n_atoms = int(first_line[0])
            second_line = f.readline().decode(errors='replace').split()
            if not second_line:
                return 1
            stride = (n_atoms + 1) if second_line[0] == "1" else (n_atoms + 2)

        result = subprocess.run(['wc', '-l', file_path],
                                capture_output=True, text=True)
        total_lines = int(result.stdout.split()[0])

        # wc undercounts by one when the file has no trailing newline
        with open(file_path, 'rb') as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b'\n':
                total_lines += 1

        return total_lines // stride
    except (OSError, ValueError, IndexError):
        return 0


def production_start(n_total, n_equil, n_production):
    """Index of the first production frame in a trajectory of *n_total* frames.

    Production is the LAST *n_production* frames, not everything past the first
    *n_equil*. Counting from the end keeps the sample the right length and the
    right age whenever the trajectory does not come out at exactly
    ``n_equil + n_production`` -- a resumed run that overshoots its target
    would otherwise fold the extra frames into the average, and two
    trajectories of different lengths would be averaged over different spans.

    The floor at *n_equil* is what stops a short trajectory from reaching back
    into equilibration to make up the count: better to average fewer frames,
    and say so, than to average unequilibrated ones.
    """
    return max(n_equil, n_total - n_production)


def trim_arc_to_production(full_arc, prod_arc, n_equil, n_production):
    """Write a production-only arc holding the last *n_production* frames.

    Never reaches back past *n_equil*; see :func:`production_start`.
    """
    n_total = count_arc_frames(full_arc)
    skip_frames = production_start(n_total, n_equil, n_production)
    if skip_frames == 0:
        if full_arc != prod_arc:
            import shutil
            shutil.copy(full_arc, prod_arc)
        return
    _, stride, _ = _arc_layout(full_arc)
    skip = skip_frames * stride
    with open(full_arc, 'rb') as f_in, open(prod_arc, 'wb') as f_out:
        for i, line in enumerate(f_in):
            if i >= skip:
                f_out.write(line)


# ---------------------------------------------------------------------------
# BAR output (.bar / .ene) helpers
# ---------------------------------------------------------------------------

def read_free_energy(enefile):
    """Read (free_energy, error) from a BAR .ene file, or None if absent.

    Prefers the iterative BAR estimate over the bootstrap one.
    """
    with open(enefile) as f:
        for line in f:
            if "Free Energy via BAR Iteration" in line:
                tokens = line.split()
                return float(tokens[-4]), float(tokens[-2])
        f.seek(0)
        for line in f:
            if "Free Energy via BAR Bootstrap" in line:
                tokens = line.split()
                return float(tokens[-4]), float(tokens[-2])
    return None


def ene_complete(enepath):
    """Return True if *enepath* exists and contains the BAR convergence line."""
    if not os.path.isfile(enepath):
        return False
    return any("BAR Estimate of -T*dS" in line for line in read_last_lines(enepath))


def md_crash_reason(logpath, errpath):
    """Return why a Tinker MD run died, or None if there is no sign of a crash.

    A dynamics run that blows up ends its log with Tinker's uncaught-exception
    line and dumps the offending coordinates to a .err file. Either is proof
    that the run is over and no further frames are coming, which is the
    difference between a trajectory that is merely slow and one that will never
    finish -- a distinction a frame count alone cannot make.
    """
    if os.path.isfile(logpath):
        for line in read_last_lines(logpath):
            if "Terminating with uncaught exception" in line:
                return line.split(':', 1)[-1].strip() or line.strip()
    if os.path.isfile(errpath):
        return f"coordinates dumped to {os.path.basename(errpath)}"
    return None


def seconds_since_write(*paths):
    """Seconds since the most recently written of *paths*, or None if none exist.

    A running Tinker job appends to its .arc and .log continuously, so the
    newest of those mtimes is when the job was last demonstrably alive. This is
    the only evidence available for a job that dies *silently* -- killed by a
    node reboot, an eviction, or the OOM killer -- which leaves the log clean
    and so slips past :func:`md_crash_reason` entirely.
    """
    newest = None
    for path in paths:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if newest is None or mtime > newest:
            newest = mtime
    if newest is None:
        return None
    return max(0.0, time.time() - newest)


def stall_reason(timeout, *paths):
    """Describe an output set untouched for *timeout* seconds, else None.

    *timeout* of 0 (or less) disables the check. Callers pass every file the
    job writes; the check fires only once all of them have gone quiet, and
    never before any of them exists -- a job whose first frame has not landed
    yet has no mtime to judge and must not be declared dead.
    """
    if not timeout or timeout <= 0:
        return None
    idle = seconds_since_write(*paths)
    if idle is None or idle < timeout:
        return None
    return f"no output written for {idle / 60.0:.0f} min"


def bar_file_snapshot_counts(barpath):
    """Return both trajectory lengths recorded in a .bar file, or (0, 0).

    A .bar holds one block per state, each introduced by a header line whose
    first field is that state's snapshot count::

        <n0>  <temperature>  <title>
        ... n0 energy rows ...
        <n1>  <temperature>  <title>
        ... n1 energy rows ...

    Both counts matter. A BAR started while the second state's trajectory was
    still being written produces a full first block and a truncated second one,
    so checking only the header of the file misses the failure: BAR then
    evaluates its snapshot range against frames that do not exist and writes an
    .ene of NaN that never converges.
    """
    try:
        with open(barpath) as f:
            parts = f.readline().split()
            if not parts:
                return 0, 0
            n0 = int(parts[0])
            for _ in range(n0):          # skip the first block's energy rows
                if not f.readline():
                    return n0, 0
            parts = f.readline().split()
            return n0, (int(parts[0]) if parts else 0)
    except (IOError, ValueError):
        return 0, 0


def bar_sh_steps_match(shpath, expected_start, expected_total):
    """Return True if the BAR step-2 line in *shpath* uses the expected range.

    The step-2 line looks like::

        $BAR? 2 barfile <start> <total> 1 <start> <total> 1 > enefile

    so parts[1]=='2', parts[3]==start, parts[4]==total. A stale script from a
    run with different simulation lengths therefore fails to match, which is
    how callers detect that a .bar/.ene pair must be regenerated.
    """
    if not os.path.isfile(shpath):
        return False
    with open(shpath) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5 and parts[1] == '2':
                try:
                    return int(parts[3]) == expected_start and int(parts[4]) == expected_total
                except (ValueError, IndexError):
                    pass
    return False


# ---------------------------------------------------------------------------
# Tinker .xyz helpers
# ---------------------------------------------------------------------------

def read_txyz_natoms(path):
    """Return the atom count from the first line of a Tinker .xyz."""
    with open(path) as f:
        return int(f.readline().split()[0])


def read_txyz_box(path):
    """Return (a, b, c) box lengths from a Tinker .xyz, or None if absent.

    A box line is only present when the file has n_atoms+2 lines; solute-only
    coordinate files have no box and yield None.
    """
    with open(path) as f:
        lines = f.readlines()
    n_atoms = int(lines[0].split()[0])
    if n_atoms != len(lines) - 2:
        return None
    tokens = lines[1].split()
    if len(tokens) < 3:
        raise ValueError(
            f"{path} line 2 must contain at least 3 values for box dimensions (a, b, c)"
        )
    return float(tokens[0]), float(tokens[1]), float(tokens[2])


def read_txyz_atoms(path):
    """Parse a Tinker .xyz into [[idx, sym, x, y, z, type, [bonded...]], ...]."""
    with open(path) as f:
        lines = f.read().splitlines()
    n = int(lines[0].split()[0])
    # Skip an optional box line so dimer files with lattice info still parse
    body = lines[1:]
    if body and len(body[0].split()) >= 6:
        try:
            [float(p) for p in body[0].split()[:6]]
            body = body[1:]
        except ValueError:
            pass
    atoms = []
    for ln in body[:n]:
        s = ln.split()
        atoms.append([int(s[0]), s[1], float(s[2]), float(s[3]), float(s[4]),
                      int(s[5]), [int(b) for b in s[6:]]])
    return atoms


def write_txyz_atoms(path, atoms, title="monomer"):
    """Write an atom list produced by :func:`read_txyz_atoms` to a Tinker .xyz."""
    with open(path, 'w') as f:
        f.write(f"{len(atoms)}  {title}\n")
        for i, a in enumerate(atoms):
            _, sym, x, y, z, typ, bonds = a
            conn = ' '.join(str(b) for b in bonds)
            f.write(f"{i+1:>3} {sym:<3} {x:12.6f} {y:12.6f} {z:12.6f} {typ:>5}  {conn}\n")


def split_dimer_monomers(atoms, n1):
    """Split a dimer (first *n1* atoms = fragment 1) and renumber bonds."""
    def build(sel):
        old2new = {old0 + 1: k + 1 for k, old0 in enumerate(sel)}   # 1-based map
        mon = []
        for old0 in sel:
            _, sym, x, y, z, typ, bonds = atoms[old0]
            nb = [old2new[b] for b in bonds if b in old2new]
            mon.append([old2new[old0 + 1], sym, x, y, z, typ, nb])
        return mon
    return build(list(range(0, n1))), build(list(range(n1, len(atoms))))


def parse_system_mass(xyz_file, prm_file):
    """Return total system mass (g/mol) by summing atomic masses from the prm."""
    atom_types = []
    with open(xyz_file) as f:
        lines = f.readlines()

    # Skip the header, and the box line when line 2 holds 6 floats
    start = 1
    parts1 = lines[1].split()
    if len(parts1) >= 6:
        try:
            [float(p) for p in parts1[:6]]
            start = 2
        except ValueError:
            pass

    for line in lines[start:]:
        parts = line.split()
        if len(parts) >= 6:
            atom_types.append(int(parts[5]))

    # atom record format: atom  type  class  symbol  "description"  atomicnum  mass  valence
    type_mass = {}
    atom_re = re.compile(
        r'^\s*atom\s+(\d+)\s+\d+\s+\S+\s+"[^"]+"\s+\d+\s+([0-9.]+)\s+\d+'
    )
    with open(prm_file) as f:
        for line in f:
            m = atom_re.match(line)
            if m:
                type_mass[int(m.group(1))] = float(m.group(2))

    total_mass = 0.0
    for atype in atom_types:
        if atype not in type_mass:
            raise KeyError(f"Atom type {atype} from {xyz_file} not found in {prm_file}")
        total_mass += type_mass[atype]
    return total_mass


# ---------------------------------------------------------------------------
# Key file helpers
# ---------------------------------------------------------------------------

def rewrite_parameters_line(contents, prm_path, ensure_archive=True):
    """Return *contents* with its PARAMETERS line pointing at *prm_path*."""
    prm_abs = str(Path(prm_path).resolve())
    contents = re.sub(
        r'(?im)^PARAMETERS\s+.*$',
        f'PARAMETERS   {prm_abs}',
        contents,
    )
    if ensure_archive and not re.search(r'(?im)^archive\s*$', contents):
        contents = contents.rstrip('\n') + '\narchive\n'
    return contents


def derive_liquid_key(src_key, dest_key):
    """Write *dest_key* from *src_key* with HFE-only keywords removed.

    Strips vdw-annihilate, vdw-lambda, ele-lambda and ligand lines so the
    result describes an unperturbed neat-liquid simulation.
    """
    with open(src_key) as f:
        lines = f.readlines()
    kept = [ln for ln in lines if not HFE_ONLY_RE.match(ln)]
    cleaned = []
    prev_blank = False
    for ln in kept:
        is_blank = ln.strip() == ''
        if is_blank and prev_blank:
            continue
        cleaned.append(ln)
        prev_blank = is_blank
    while cleaned and cleaned[-1].strip() == '':
        cleaned.pop()
    # archive is required for Tinker9 to write the .arc trajectory
    if not any(ln.strip().lower() == 'archive' for ln in cleaned):
        cleaned.append('archive\n')
    else:
        cleaned.append('\n')
    os.makedirs(str(Path(dest_key).parent), exist_ok=True)
    with open(dest_key, 'w') as f:
        f.writelines(cleaned)


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def cell_volume(a, b, c, alpha=90.0, beta=90.0, gamma=90.0):
    """Volume (A^3) of a unit cell from its lengths and angles."""
    ca, cb, cg = (np.cos(np.radians(x)) for x in (alpha, beta, gamma))
    factor = 1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg
    return a * b * c * np.sqrt(max(factor, 0.0))


def parse_arc_densities(arc_file, total_mass, n_equil, n_production):
    """Per-frame densities (kg/m^3) from the box line of every .arc frame.

    Returns ``(production_densities, n_frames_parsed)``, the densities being
    the last *n_production* frames as chosen by :func:`production_start` --
    the same window :func:`trim_arc_to_production` cuts for the reweighted
    energies, so the two series stay frame-for-frame aligned.

    The trajectory is the record of what was simulated, so densities are taken
    from it rather than from the MD log. The two are not interchangeable: the
    log and the archive are flushed independently, so a run that is killed
    partway leaves them at different lengths, and the .arc is also what the
    reweighted energies of the density derivative are computed over. Reading
    both series from the same file keeps frame i of one aligned with frame i of
    the other.

    Only the box line of each frame is needed, so the extraction runs through
    awk rather than decoding several GB of coordinates in Python.
    """
    n_atoms, stride, has_box = _arc_layout(arc_file)
    if not has_box:
        raise RuntimeError(
            f"{arc_file} has no box line per frame (NVT trajectory); density "
            f"needs a volume, so the run must be NPT."
        )

    # Frame block: header, box, then n_atoms coordinate lines.
    result = subprocess.run(
        ['awk', f'NR % {stride} == 2', arc_file],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"could not read box lines from {arc_file}: {result.stderr.strip()}")

    rho_list = []
    for line in result.stdout.splitlines():
        f = line.split()
        if len(f) < 3:
            continue
        try:
            dims = [float(x) for x in f[:6]] if len(f) >= 6 else [float(x) for x in f[:3]]
        except ValueError:
            continue
        V = cell_volume(*dims)
        if V > 0.0:
            rho_list.append(total_mass / (DENSITY_FACTOR * V))

    start = production_start(len(rho_list), n_equil, n_production)
    return np.array(rho_list[start:]), len(rho_list)


def _arc_layout(arc_file):
    """Return (n_atoms, stride, has_box) for a Tinker .arc file.

    A second line that starts with the atom index ``1`` is the first coordinate
    row, which means the frame carries no box line (NVT).
    """
    with open(arc_file, 'rb') as f:
        first = f.readline().decode(errors='replace').split()
        if not first:
            raise RuntimeError(f"Empty arc file: {arc_file}")
        n_atoms = int(first[0])
        second = f.readline().decode(errors='replace').split()
    has_box = bool(second) and second[0] != "1"
    return n_atoms, (n_atoms + 2) if has_box else (n_atoms + 1), has_box


def parse_analyze_energies(log_path):
    """Return per-frame total potential energies from an ANALYZE log."""
    energies = []
    with open(log_path) as fh:
        for line in fh:
            if 'Total Potential Energy' in line:
                m = re.search(r'Total Potential Energy\s*:\s*([-\d.]+)', line)
                if m:
                    energies.append(float(m.group(1)))
    if not energies:
        raise RuntimeError(f"No energy entries found in analyze log: {log_path}")
    return np.array(energies)


def count_analyze_energies(log_path):
    """Return how many energy entries an analyze log holds so far."""
    if not os.path.isfile(log_path):
        return 0
    try:
        with open(log_path) as fh:
            return sum(1 for line in fh if 'Total Potential Energy' in line)
    except OSError:
        return 0


# ---------------------------------------------------------------------------
# Tinker environment
# ---------------------------------------------------------------------------

def load_tinker_env(tinkerenv):
    """Source *tinkerenv* and merge its exported variables into os.environ."""
    result = subprocess.run(
        ['bash', '-c', f'source {tinkerenv} && env'],
        capture_output=True, text=True,
    )
    for line in result.stdout.splitlines():
        key, sep, val = line.partition('=')
        if sep:
            os.environ[key] = val


def package_data(name):
    """Return the path to a file shipped in ``autoff/data``."""
    return str(Path(__file__).resolve().parent / 'data' / name)
