"""Job dispatch: turns per-system job descriptions into cluster submissions.

Systems produce :class:`Job` records; the dispatcher batches them by queue and
hands each batch to :func:`autoff.submit.submit_jobs`. Nothing here tracks
completion — the cluster is fire-and-forget, so the orchestrator polls output
files instead.

In dry-run mode no SSH happens at all: the batches are appended to a manifest
so a run can be inspected end-to-end on a machine with no cluster access.
"""

import logging
import os
from dataclasses import dataclass

from . import submit

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Job:
    """One script to run in one directory on one kind of node."""
    script: str      # basename, relative to workdir
    workdir: str     # absolute
    queue: str       # CPU | GPU
    nproc: int = 2
    label: str = ""  # for the dry-run manifest only


class JobDispatcher:
    """Submits :class:`Job` batches, or records them when dry-running."""

    def __init__(self, nodes=None, dry_run=False, manifest_path=None):
        self.nodes = list(nodes or [])
        self.dry_run = dry_run
        self.manifest_path = manifest_path
        self.submitted = []

    def submit(self, jobs):
        """Dispatch *jobs*, grouped by (queue, nproc). Returns the count sent."""
        jobs = [j for j in jobs if j is not None]
        if not jobs:
            return 0

        batches = {}
        for job in jobs:
            batches.setdefault((job.queue, job.nproc), []).append(job)

        for (queue, nproc), batch in sorted(batches.items()):
            commands = [f"cd {j.workdir}; sh {j.script}" for j in batch]
            self.submitted.extend(batch)
            if self.dry_run:
                self._record(queue, nproc, batch)
                log.info("[dry-run] would submit %d %s job(s) with -n %d",
                         len(batch), queue, nproc)
                continue
            log.info("Submitting %d %s job(s) with -n %d", len(batch), queue, nproc)
            # Node overrides apply to both queues; submit_jobs falls back to the
            # site node file when the list is empty.
            gpu_nodes = self.nodes or None
            cpu_nodes = self.nodes or None
            submit.submit_jobs(commands, queue, gpu_nodes, cpu_nodes, nproc)

        return len(jobs)

    def _record(self, queue, nproc, batch):
        if not self.manifest_path:
            return
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        with open(self.manifest_path, 'a') as f:
            for job in batch:
                f.write(f"{queue:<4} n={nproc:<3} {job.label or '-':<28} "
                        f"{job.workdir}/{job.script}\n")
