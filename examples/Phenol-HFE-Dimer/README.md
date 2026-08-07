# Phenol: HFE + dimer interaction energy

A parameter-fitting example that optimizes against **two** targets at once:

1. **Hydration free energy** — from BAR over the alchemical decoupling windows.
2. **Dimer interaction energy** — `E_int = E_dimer − E_mon1 − E_mon2` from a
   gas-phase Tinker energy evaluation, compared against a QM reference.

A target contributes to the fit whenever it has a reference value, so leaving
`expt` off a system reports it without fitting it. This example defines no
`liquids`, so density plays no part.

## Files

| File | Purpose |
|------|---------|
| `config.yaml` | The whole run: systems, targets, and the fit |
| `input/amoeba09.prm` | AMOEBA parameter file, shared by every system |
| `input/phenol.xyz` | Gas-phase phenol |
| `input/phenol_solv.xyz` | Solvated phenol box |
| `input/dimers/phenol_water.xyz` | Phenol–water H-bonded dimer |

## What is optimized

The van der Waals `R` and `epsilon` of the phenol hydroxyl oxygen (class 36):

```yaml
opt_params:   ["vdw-36 3.4050 0.1100"]
params_range: ["0.20   0.05"]        # ±0.20 Å on R, ±0.05 on epsilon
```

Both targets depend on this one pair of numbers, which is what makes the fit
interesting: the hydration free energy constrains the parameters in bulk
water, the dimer energy constrains them at a single hydrogen bond, and the
optimizer has to satisfy both.

## Run it

```bash
cd examples/Phenol-HFE-Dimer

autoff run config.yaml --dry-run   # inspect the generated inputs first
autoff run config.yaml
```

Each step logs a target-vs-current table to `results/autoff.log`. The
optimized parameter file is written to `prm/amoeba09.prm.final`; your input
`.prm` is never modified.

## ⚠️ Placeholder dimer data — replace before real use

`input/dimers/phenol_water.xyz` is a **constructed** starting geometry (O–H···O
hydrogen bond at ~1.9 Å), **not** QM-optimized, and the reference energy of
`-6.9 kcal/mol` in `config.yaml` is **illustrative**. Both exist to demonstrate
the file format — substitute your own QM-optimized geometry and interaction
energy before fitting for production.

`frag1_natoms: 13` says the first 13 atoms are fragment 1 (phenol) and the rest
are fragment 2 (water); the monomers are split automatically.

## Adding a density target

Add a `liquids:` entry with a neat-liquid box, its temperatures, and the
measured densities:

```yaml
liquids:
  - name: water_neat
    box_xyz: input/water_box.xyz
    temperatures: [298.15]
    expt_densities: [997.0]
    production_time: 0.2
```

See `examples/Multi-Property` for a config that combines solutes and a liquid.
