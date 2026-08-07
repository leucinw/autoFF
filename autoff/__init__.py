"""autoFF — automated force field property simulation and parameter fitting.

Computes condensed-phase and gas-phase observables with Tinker — hydration
free energies by BAR, neat-liquid densities, dimer interaction energies — and
optionally tunes a shared force field so they reproduce reference values.

The package is organized around two job types, both driven by a single master
config file (see :mod:`autoff.config`):

``single-point``
    Run every simulation defined in the config and report all derived
    properties (per-solute hydration free energies, neat-liquid densities,
    dimer interaction energies) alongside their reference values.

``optimize``
    Joint least-squares fit of one shared parameter file against every
    property that has a reference value, reusing the single-point machinery
    to evaluate the objective.
"""

__version__ = "2.0.0"
