# autoFF

Automated free-energy simulation and force-field parameter fitting with
[Tinker](https://dasher.wustl.edu/tinker/) and
[Tinker9](https://github.com/TinkerTools/tinker9).

One configuration file describes an entire study: any number of solutes whose
hydration free energies you want, any number of neat liquids whose densities
you want, and any number of dimer geometries. autoFF either **reports** every
one of those properties (`single-point`) or **fits** a shared parameter file so
they all match reference values at once (`optimize`).

---

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Configuration reference](#configuration-reference)
- [Job types](#job-types)
- [Directory layout](#directory-layout)
- [Command reference](#command-reference)
- [Examples](#examples)
- [Notes and limitations](#notes-and-limitations)

---

## Install

```bash
git clone https://github.com/leucinw/autoFF.git
cd autoFF
pip install -e .
```

Dependencies are numpy, scipy and ruamel.yaml; `environment.yml` provides a
conda environment. Installing puts two commands on your path: `autoff` and
`autoff-submit`.

You also need working Tinker8 and Tinker9 builds. Their locations come from a
small environment file listing `$DYNAMIC8/9`, `$BAR8/9`, `$ANALYZE8/9` and
`$MINIMIZE8/9`. The bundled default is `autoff/data/tinker.env`; point
`shared.tinker_env` at your own copy:

```bash
export TINKER8=/path/to/tinker
export  DYNAMIC8="$TINKER8/dynamic"
export  ANALYZE8="$TINKER8/analyze"
export      BAR8="$TINKER8/bar"
export MINIMIZE8="$TINKER8/minimize"

export tk9home=/path/to/tinker9/build
export  DYNAMIC9="$tk9home/dynamic9"
export  ANALYZE9="$tk9home/analyze9"
export      BAR9="$tk9home/bar9"
export MINIMIZE9="$tk9home/minimize9"
```

---

## Quick start

```bash
cd examples/Phenol-HFE

autoff run config.yaml --dry-run   # generate every input file, submit nothing
autoff run config.yaml             # submit, wait, and report
autoff check config.yaml           # progress of a run already under way
```

`--dry-run` is the fastest way to see what a config will actually do: it
writes every key file and job script and records the intended submissions in
`results/submitted_jobs.txt`, without touching the cluster.

---

## How it works

**Hydration free energy.** A solute is decoupled from water in stages: first
its electrostatics are switched off, then its van der Waals interactions.
Each stage is a *lambda window* with its own MD trajectory. Consecutive
windows are combined with Bennett Acceptance Ratio (BAR), and the per-window
free energies sum to the transfer free energy. The same decoupling is repeated
in the gas phase and subtracted, which removes the solute's intramolecular
contribution. Solutes with fewer than five atoms have no such contribution, so
their gas leg is skipped automatically.

**Neat-liquid density.** One NPT trajectory per temperature; the density is
the mean over production frames, after discarding an equilibration segment.

**Dimer energies.** `E_int = E_dimer - E_mon1 - E_mon2` from single-point
energy evaluations, optionally after letting the trial parameters relax the
dimer geometry.

**Fitting.** Every property with a reference value becomes one entry in a
single residual vector:

```
residual = weight * (calculated - reference) / denominator
```

The denominator normalizes across units, so a density in kg/m³ and a free
energy in kcal/mol pull on the fit comparably. It defaults to the spread
(standard deviation) of the reference values when several exist, and to
`sqrt(|value|)` for a lone one — the ForceBalance convention. `scipy`'s
`least_squares` (trust-region reflective, soft-L1 loss) minimizes the sum of
squares subject to the bounds you give.

**Why fitting is affordable.** Derivatives never rerun dynamics. To evaluate a
perturbed parameter set, autoFF adds an extra window at a fictitious lambda
above 1.0 that *reweights the existing fully-coupled trajectory* — so a trial
parameter set costs one BAR evaluation instead of a full simulation. Density
derivatives use the fluctuation formula (Eq. 4 of Wang et al., *J. Chem.
Theory Comput.* 2013):

```
d⟨ρ⟩/dλ = -β ( ⟨ρ·dE/dλ⟩ - ⟨ρ⟩⟨dE/dλ⟩ )
```

where `dE/dλ` comes from re-analyzing the stored production frames.

---

## Configuration reference

### `shared`

| Key | Required | Meaning |
|---|---|---|
| `parameters` | yes | The one `.prm` file every system uses |
| `tinker_env` | no | Path to a Tinker environment file (default: bundled) |
| `node_list` | no | Cluster hostnames; empty falls back to the site node file |
| `checking_time` | no | Seconds between completeness polls (default 60) |
| `md_stall_timeout` | no | Seconds of output silence before an unfinished MD is declared dead (default 3600; 0 disables) |
| `verbose` | no | 0 quieter, 1 normal (default 1) |
| `skip_completeness_check` | no | Trust existing `.arc`/`.ene` files (default false) |
| `md_defaults` | no | `liquid:` and `gas:` blocks inherited by every HFE system |

### `hfe_systems` (a list)

| Key | Required | Meaning |
|---|---|---|
| `name` | yes | Unique; becomes the directory name |
| `gas_xyz` | yes | Solute alone, Tinker `.xyz` |
| `box_xyz` | yes | Solute in a solvent box, with a box line; each side ≥ 30 Å |
| `expt` | for fitting | Reference HFE, kcal/mol |
| `weight` | no | Relative weight in the fit (default 1.0) |
| `denom` | no | Overrides the automatic scale normalizer |
| `lambda_window` | no | `courser` (18 windows), `default` (26), or a path |
| `copy_arc_for_perturb` | no | Reweight the coupled trajectory for perturbed parameters (default true) |
| `manual_ele_scale` | no | Scale multipoles in the key instead of using `ele-lambda` |
| `liquid_key` / `gas_key` | no | Custom key templates |
| `md` | no | Per-system overrides deep-merged onto `md_defaults` |

MD blocks take `total_time` (ns), `time_step` (fs), `write_freq` (ps),
`temperature` (K); liquid also takes `pressure` (atm) and `ensemble`
(`NPT`/`NVT`). Setting a gas `total_time` of 0 disables the gas leg.

### `liquids` (a list)

| Key | Required | Meaning |
|---|---|---|
| `name` | yes | Unique; becomes the directory name |
| `box_xyz` | yes | Neat-liquid box |
| `temperatures` | yes | One or more, K |
| `expt_densities` | for fitting | One per temperature, kg/m³ |
| `weights` | no | One per temperature; normalized internally by their count |
| `denom` | no | Overrides the automatic scale normalizer |
| `equil_time` | no | ns discarded before averaging (default 0.02) |
| `production_time` | yes | ns averaged |
| `key` | no | Key template; otherwise derived from the bundled liquid key |
| `md` | no | `time_step`, `write_freq`, `pressure` |

### `dimers` (a list) and `dimer_opt`

| Key | Required | Meaning |
|---|---|---|
| `xyz` / `start_xyz` | yes | Dimer geometry |
| `frag1_natoms` | yes | The first fragment is the leading N atoms |
| `expt` / `target` | for fitting | Reference interaction/binding energy, kcal/mol |
| `weight` | no | Relative weight (default 1.0) |
| `grad` | `dimer_opt` only | RMS gradient convergence for the relaxation |

### `job`

```yaml
job:
  type: single-point            # or: optimize
  optimize:
    opt_params:   ["vdw-36 3.4050 0.1100"]
    params_range: ["0.20   0.05"]
    diff_step: 0.0001           # finite-difference step
    ftol: 0.0001                # convergence tolerances
    gtol: 0.0001
    xtol: 0.0001
```

Each `opt_params` entry is `"<term_key> <value1> <value2> ..."` with a matching
range string. The term key uses hyphens where the Tinker line has fields, so
`vdw-36` writes `vdw   36   ...` and `vdwpair-401-402` writes
`vdwpair   401   402   ...`. Bounds are `value ± range`; **a range of 0 pins
that value** — it is written unchanged but hidden from the optimizer.

Optimized parameters are written as override lines appended to a pristine copy
of your `.prm`. Tinker takes the last definition of a term, so the appended
line supersedes the original. The result lands in `prm/<name>.prm.final`.

---

## Job types

### `single-point`

Runs every configured simulation and reports every derived property, with the
deviation from each reference value you supplied. Output goes to
`results/singlepoint.txt` (human-readable), `results/singlepoint.yaml`
(machine-readable), and one `results/hfe_<name>.txt` per solute with its
per-window BAR breakdown.

### `optimize`

Fits the shared parameter file to all targets jointly, reusing the
single-point machinery to evaluate the objective. Each step logs a table of
every target with its current value, reference, difference, and weighted
residual.

Systems without a reference value still run — they are simply reported rather
than fitted.

---

## Directory layout

Everything a run generates lives under `workdir`, separated per system so
nothing collides:

```
<workdir>/
├── config.yaml
├── prm/
│   ├── amoeba09.prm            # working copy, rewritten each optimizer step
│   ├── amoeba09.prm.orig       # pristine snapshot; your input is never touched
│   ├── amoeba09.prm_01 …       # perturbation sidecars
│   └── amoeba09.prm.final      # optimized output
├── systems/
│   ├── phenol/{gas,liquid}/    # lambda windows, trajectories, BAR output, FEP_NN/
│   ├── sodium/liquid/
│   └── water_neat/             # one trajectory per temperature
└── results/
    ├── singlepoint.txt / .yaml
    ├── hfe_<name>.txt
    ├── autoff.log
    └── submitted_jobs.txt      # dry-run manifest
```

Your input files are only ever read. Coordinates are copied into the system
directory before minimization, and the parameter file is snapshotted, so
rerunning never mutates what you supplied.

---

## Command reference

```
autoff run    config.yaml [--dry-run] [-s/--skip-check] [-v N]
autoff setup  config.yaml [--dry-run]
autoff check  config.yaml
autoff report config.yaml [-s/--skip-check]
```

- `run` — execute the job named by `job.type`, end to end.
- `setup` — generate directories, keys and scripts; submit nothing.
- `check` — print per-system completeness without changing anything.
- `report` — collect results from output files already on disk.
- `--skip-check` — assume `.arc`/`.ene` files are complete. Skips the expensive
  scans; only use it when you are sure.

Jobs are dispatched over SSH by `autoff-submit`, which polls `nproc`/`top` for
CPU nodes and `nvidia-smi` for free GPUs and retries until every job is placed.
It can also be used standalone:

```bash
autoff-submit -x run_md.sh -t GPU -nodes node103 node104
```

Submission is fire-and-forget — there is no queue system and no job IDs — so
completion is detected by polling output files. All systems are submitted
together and advanced by one loop, so a fast system reaches BAR while a slow
one is still running dynamics; the wall time is set by the slowest system
rather than by their sum.

---

## Examples

| Example | Shows |
|---|---|
| `examples/Phenol-HFE` | Single solute HFE, the simplest possible config |
| `examples/Ion-HFE` | Monatomic ion; the gas leg is disabled automatically |
| `examples/Multi-Property` | **Two solutes plus a neat liquid at two temperatures, one shared parameter file** |
| `examples/Phenol-HFE-Dimer` | Fitting van der Waals parameters to an HFE and a dimer energy |

Run any of them with `autoff run config.yaml --dry-run` first to inspect what
would be submitted.

> The dimer geometry and reference energy in `Phenol-HFE-Dimer` are
> illustrative placeholders, not QM-derived values. Replace them before drawing
> conclusions from a fit.

---

## Notes and limitations

- **BAR equilibration.** The first 20% of every trajectory is discarded. A
  window therefore needs at least 6 snapshots, which is validated at load time.
- **Box shape.** Neat-liquid densities assume a cubic box (`V = a³`).
- **Parameter count.** Perturbation sidecars are numbered `_01`.., and a
  gradient needs two per free parameter plus one, so at most 49 parameters can
  be fitted at once. This is checked when the config loads.
- **Monomer caching.** The `dimer_opt` target caches relaxed monomer energies
  once, which assumes the fitted terms do not change intramolecular energy —
  true for a van der Waals class whose 1-2 interactions are excluded, but not
  in general.
- **Saturated clusters.** Submission blocks until every job is placed, so a
  full cluster stalls the polling loop.
- **Resuming.** A partial trajectory with a `.dyn` checkpoint is resumed for
  only its outstanding steps; a complete one is reused. Stale `.bar`/`.ene`
  files from a run with different sampling are detected and regenerated.
- **Dead jobs.** Nothing resubmits, so an MD that ends early has to be noticed
  or the poll loop waits forever. A blow-up is read out of the log or the
  `.err` dump; a job killed with its node leaves neither, and is caught only by
  `md_stall_timeout` — its `.arc` and `.log` simply stop being written. Both
  raise, naming the temperature or window to restart. The timeout has to sit
  well above the gap between frames: these runs write one every few seconds,
  so the hour-long default is far outside the noise, but a much slower protocol
  should raise it.

## Testing

```bash
pytest
```

The suite runs without a cluster or a Tinker build: parsers are checked
against real BAR output from a completed run, generated scripts are asserted
against their exact expected command lines, and the optimizer's residual and
Jacobian assembly is exercised with the simulation layer stubbed out.

---

## Author

Chengwen Liu — liuchw2010@gmail.com
