"""Command line interface.

    autoff run    config.yaml    execute the job named by job.type
    autoff setup  config.yaml    generate directories, keys and scripts only
    autoff check  config.yaml    report per-system completeness
    autoff report config.yaml    collect results from existing output files
"""

import argparse
import logging
import os
import sys

from . import config as config_mod
from . import optimize, singlepoint


def _setup_logging(results_dir, verbose=1, log_name='autoff.log'):
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger('autoff')
    root.setLevel(logging.DEBUG if verbose > 1 else logging.INFO)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        handler = logging.FileHandler(os.path.join(results_dir, log_name), mode='a')
        handler.setFormatter(fmt)
        root.addHandler(handler)
    return root


def build_parser():
    parser = argparse.ArgumentParser(
        prog='autoff',
        description="Automated force field property simulation and parameter fitting "
                    "for Tinker/AMOEBA.",
    )
    sub = parser.add_subparsers(dest='command', required=True)

    for name, help_text in (
        ('run', "run the job defined by job.type in the config"),
        ('setup', "generate all input files without submitting jobs"),
        ('check', "print per-system completeness"),
        ('report', "collect results from existing output files"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument('config', help="path to the master config YAML")
        p.add_argument('-v', '--verbose', type=int, default=None,
                       help="verbosity (overrides shared.verbose)")
        if name in ('run', 'setup'):
            p.add_argument('--dry-run', action='store_true',
                           help="generate every file but submit nothing")
        if name in ('run', 'report'):
            p.add_argument('-s', '--skip-check', action='store_true',
                           help="assume .arc/.ene files exist and are complete")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = config_mod.load(args.config)
    if args.verbose is not None:
        cfg.verbose = args.verbose

    log = _setup_logging(cfg.results_dir, cfg.verbose)
    skip_check = getattr(args, 'skip_check', False) or None
    dry_run = getattr(args, 'dry_run', False)

    log.info("config: %s", os.path.abspath(args.config))
    log.info("workdir: %s", cfg.workdir)
    log.info("job type: %s | %d HFE system(s), %d liquid(s), %d dimer(s)",
             cfg.job_type, len(cfg.hfe_systems), len(cfg.liquids), len(cfg.dimers))

    if args.command == 'setup':
        runner = singlepoint.Runner(cfg, dry_run=dry_run)
        runner.setup()
        log.info("Input files generated under %s", cfg.systems_dir)
        return 0

    if args.command == 'check':
        runner = singlepoint.Runner(cfg, dry_run=True)
        runner.status()
        return 0

    if args.command == 'report':
        # Collect only: never submit jobs, never re-minimize coordinates
        runner = singlepoint.Runner(cfg, dry_run=False, skip_check=skip_check)
        runner.setup(minimize=False)
        _, text = singlepoint.write_report(cfg, runner.collect())
        print(text)
        return 0

    if cfg.job_type == 'optimize':
        optimize.run(cfg, dry_run=dry_run, skip_check=skip_check)
    else:
        singlepoint.run(cfg, dry_run=dry_run, skip_check=skip_check)
    return 0


if __name__ == '__main__':
    sys.exit(main())
