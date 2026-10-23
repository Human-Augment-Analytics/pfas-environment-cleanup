"""Batch review and submission; batches never imply calculation dependencies."""

import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Unpack

from dashboard.core import execution, models

if TYPE_CHECKING:
    from dashboard.core.jobs.tasks import Manager

from dashboard.core.jobs.persistence import now
from dashboard.core.slurm.scheduler import settings as scheduler_settings

MOLECULE_PROBE = """import shutil,json,sys
import rdkit,pymatgen.core,ase,openbabel
from rdkit import Chem
assert shutil.which('obabel') and shutil.which('cif2cell')
results={}
for id,smiles in json.load(sys.stdin).items():
    molecule=Chem.MolFromSmiles(smiles)
    results[id] = sorted({a.GetSymbol() for a in molecule.GetAtoms()} | {'H','C','O','F'}) if molecule is not None else None
print(json.dumps(results))
"""
PROGRESS_INTERVAL = 100
MAX_BATCH_CANDIDATES = 1001
MAX_BATCH_ID_LENGTH = 100
BATCH_ENTRY_FIELDS = ("candidate", "processes", "status", "task_id", "reason")


@dataclass
class ReviewData:
    """Per-preview caches; submission rereads history and source hashes."""

    request: models.BatchReviewRequest
    resources: models.Resources
    image_hash: str | None
    molecules: dict
    related: dict = field(default_factory=dict)
    prepared: dict = field(default_factory=dict)
    all_prepared: list[models.TaskRecord] = field(default_factory=list)
    pseudo_hashes: dict = field(default_factory=dict)


class Batches:
    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(execution.APPLICATION_LOGGER_NAME)

    def __init__(self, manager: "Manager") -> None:
        self.manager = manager

    def review(
        self,
        estimate_processes: list[int] | None = None,
        **request: Unpack[models.BatchReviewRequest],
    ) -> dict:
        if estimate_processes is None:
            return self._review(**request)
        self.validate_process_counts(estimate_processes, request)
        reviews = []
        for rank in estimate_processes:
            rank_request = request.copy()
            rank_request["processes"] = rank
            reviews.append(self._review(**rank_request))
        result = dict(reviews[0])
        # Each input is initialized at every selected rank before moving on.
        result["entries"] = [
            review["entries"][index]
            for index in range(len(request["candidates"]))
            for review in reviews
        ]
        result["resources"] = {**result["resources"], "cpus": max(estimate_processes)}
        result["estimate_processes"] = estimate_processes
        return result

    def validate_process_counts(
        self, estimate_processes: list[int], request: models.BatchReviewRequest
    ) -> None:
        if request.get("kind") != "estimate_ram" or request.get("target") != "slurm":
            raise ValueError("Process-count groups require Slurm RAM estimation")
        if (
            not estimate_processes
            or len(estimate_processes) > execution.MAX_ESTIMATION_RANKS
            or len(set(estimate_processes)) != len(estimate_processes)
            or any(
                type(n) is not int or not 1 <= n <= execution.MAX_SLURM_PROCESSES
                for n in estimate_processes
            )
        ):
            raise ValueError("Provide distinct process counts from 1 to 256")

    def _review(
        self,
        candidates: list[str],
        kind: str,
        system: str = "candidate",
        runtime: str = "native",
        processes: int = 1,
        memory_gib: float = 4,
        timeout: float | None = None,
        retry: bool = False,
        rerun_completed: bool = False,
        target: str = "local",
        issue_tokens: bool = True,
        scheduler: models.SchedulerSettings | None = None,
        **_: object,
    ) -> dict:
        """Retain API options; review data and individual entries in focused steps."""
        request = models.BatchReviewRequest(
            candidates=candidates,
            kind=kind,
            system=system,
            runtime=runtime,
            processes=processes,
            memory_gib=memory_gib,
            timeout=timeout,
            retry=retry,
            rerun_completed=rerun_completed,
            target=target,
            issue_tokens=issue_tokens,
            scheduler=scheduler,
        )
        return self.review_request(request)

    def validate_review(self, request: models.BatchReviewRequest) -> None:
        self.manager.validate_target(request["target"], request["kind"])
        if request["kind"] not in execution.TaskKind or request["system"] not in (
            "candidate",
            "complex",
            "tfa",
        ):
            raise ValueError("Unknown batch action")
        candidates = request["candidates"]
        if (
            not candidates
            or len(candidates) > MAX_BATCH_CANDIDATES
            or len(set(candidates)) != len(candidates)
        ):
            raise ValueError("Provide distinct selected candidate IDs")
        if any(candidate not in self.manager.candidates for candidate in candidates):
            raise ValueError("Unknown candidate")

    def probe_molecules(self, request: models.BatchReviewRequest) -> dict:
        if request["kind"] != "prepare":
            return {}
        # One environment/SMILES probe serves the entire selected batch.
        molecules = {
            candidate: self.manager.candidates[candidate]["smiles"]
            for candidate in request["candidates"]
        }
        result = self.manager.runtimes.chemistry_probe(
            request["runtime"], MOLECULE_PROBE, input_text=json.dumps(molecules)
        )
        if result.returncode:
            raise ValueError("Preparation tools unavailable: " + result.stderr[-1000:])
        return json.loads(getattr(result, "stdout", "{}"))

    def review_data(self, request: models.BatchReviewRequest) -> ReviewData:
        self.validate_review(request)
        request = request.copy()
        request["timeout"] = self.manager.default_timeout(
            request["kind"], request["timeout"]
        )
        cpus = (
            request["processes"]
            if request["kind"] in execution.CALCULATION_KINDS
            else 1
        )
        resources = self.manager.resources(
            request["runtime"],
            cpus,
            request["memory_gib"],
            request["timeout"],
            request["target"],
        )
        identity = self.manager.runtime_identity(request["runtime"], request["target"])
        data = ReviewData(request, resources, identity, self.probe_molecules(request))
        # Build history indices once, avoiding candidate-count database scans.
        for task in self.manager.store.tasks():
            data.related.setdefault(
                (task["candidate"], task["kind"], task["system"]), []
            ).append(task)
            if task["kind"] == "prepare" and task["status"] == "succeeded":
                data.prepared.setdefault(task["candidate"], []).append(task)
                data.all_prepared.append(task)
        return data

    def review_request(self, request: models.BatchReviewRequest) -> dict:
        self.logger.info(
            "Batch preview: reviewing %s selected %s jobs",
            len(request["candidates"]),
            request["kind"],
        )
        data = self.review_data(request)
        entries = []
        for candidate in request["candidates"]:
            entries.append(self.review_entry(candidate, data))
            self.review_progress(len(entries), len(request["candidates"]))
        return self.review_response(data, entries)

    def review_progress(self, completed: int, total: int) -> None:
        if completed % PROGRESS_INTERVAL == 0 or completed == total:
            self.logger.info("Batch preview: reviewed %s/%s jobs", completed, total)

    def review_response(self, data: ReviewData, entries: list[dict]) -> dict:
        request = data.request
        return {
            "entries": entries,
            "resources": data.resources,
            "image_hash": data.image_hash,
            "runtime": request["runtime"],
            "target": request["target"],
            "scheduler_settings": scheduler_settings(
                request["scheduler"], request["kind"]
            )
            if request["target"] == "slurm"
            else None,
        }

    def preparation_problem(self, candidate: str, data: ReviewData) -> str:
        if candidate not in data.molecules:
            return ""
        elements = data.molecules[candidate]
        if elements is None:
            return "Invalid representative SMILES"
        missing = [
            element + ".UPF"
            for element in elements
            if not (self.manager.config.pseudos / (element + ".UPF")).is_file()
        ]
        return "Missing pseudopotentials: " + ", ".join(missing) if missing else ""

    def related_tasks(
        self, candidate: str, data: ReviewData
    ) -> list[models.TaskRecord]:
        request = data.request
        system = (
            request["system"]
            if request["kind"] in execution.CALCULATION_KINDS
            else "tfa"
            if candidate == "tfa"
            else "candidate"
        )
        related = data.related.get((candidate, request["kind"], system), [])
        if request["kind"] == "estimate_ram":
            related = [
                task
                for task in related
                if task.get("processes") == request["processes"]
            ]
        return related

    def review_entry(self, candidate: str, data: ReviewData) -> dict:
        entry = {
            "candidate": candidate,
            "processes": data.request["processes"],
            "status": "eligible",
            "reason": "",
        }
        problem = self.preparation_problem(candidate, data)
        related = self.related_tasks(candidate, data)
        if problem:
            entry.update(status="unavailable", reason=problem)
        elif any(task["status"] in execution.ACTIVE_TASK_STATUSES for task in related):
            entry.update(status="active", reason="Already queued or running")
        else:
            self.review_available(entry, data, related)
        return entry

    def preview_options(self, candidate: str, data: ReviewData) -> dict:
        request = data.request
        prepared = (
            data.all_prepared
            if request["system"] == "tfa"
            else data.prepared.get(candidate, [])
        )
        return {
            "candidate": candidate,
            "system": request["system"],
            "processes": request["processes"],
            "target": request["target"],
            "runtime": request["runtime"],
            "kind": request["kind"],
            "issue_token": request["issue_tokens"],
            "_tasks": prepared,
            "_pseudo_hashes": data.pseudo_hashes,
            "scheduler": request["scheduler"],
            "memory_gib": request["memory_gib"],
            "timeout": request["timeout"],
        }

    def matching_result(
        self, task: models.TaskRecord, preview: dict, data: ReviewData
    ) -> bool:
        if (
            task["status"] != "succeeded"
            or task.get("input_hash") != preview["input_hash"]
            or task.get("pseudopotentials") != preview["pseudopotentials"]
        ):
            return False
        request = data.request
        if request["kind"] == "qe":
            return bool(task.get("evidence", {}).get("confirmed"))
        return bool(
            task.get("ram_estimate", {}).get("per_process")
            and task.get("source_hash") == preview["source_hash"]
            and task.get("runtime", "native") == request["runtime"]
            and task.get("processes") == request["processes"]
            and task.get("image_hash") == data.image_hash
        )

    def review_calculation(
        self, entry: dict, data: ReviewData, related: list[models.TaskRecord]
    ) -> None:
        preview = self.manager.preview(**self.preview_options(entry["candidate"], data))
        entry.update(preview)
        request = data.request
        rerun = request["retry"] or (
            request["kind"] == "estimate_ram" and request["rerun_completed"]
        )
        if not rerun and any(
            self.matching_result(task, preview, data) for task in related
        ):
            reason = (
                "Matching confirmed result exists"
                if request["kind"] == "qe"
                else "Matching QE RAM estimate exists"
            )
            entry.update(status="completed", reason=reason)

    def existing_artifacts(
        self, entry: dict, kind: str, related: list[models.TaskRecord]
    ) -> bool:
        names = (
            ["diagram.png"]
            if kind == "diagram"
            else ["tfa.in"]
            if entry["candidate"] == "tfa"
            else ["candidate.in", "complex.in", "tfa.in", "manifest.json"]
        )
        return any(
            task["status"] == "succeeded"
            and all(
                (self.manager.config.artifacts / task["id"] / name).is_file()
                for name in names
            )
            for task in related
        )

    def review_available(
        self, entry: dict, data: ReviewData, related: list[models.TaskRecord]
    ) -> None:
        try:
            if data.request["kind"] in execution.CALCULATION_KINDS:
                self.review_calculation(entry, data, related)
            elif not data.request["retry"] and self.existing_artifacts(
                entry, data.request["kind"], related
            ):
                entry.update(
                    status="completed", reason="Existing artifacts are available"
                )
        except (ValueError, OSError) as error:
            entry.update(status="unavailable", reason=str(error))

    def submission_review(self, request: dict) -> dict:
        review = self.review(**request, issue_tokens=False)
        if review["image_hash"] != request.get("image_hash"):
            raise ValueError("Container image changed; review the batch again")
        return review

    def new_batch(self, request: dict, review: dict, batch_id: str) -> dict:
        # Persist this acceptance record before any task so restart is idempotent.
        return {
            "id": batch_id,
            "accepting": True,
            "target": request.get("target", "local"),
            "scheduler": review["scheduler_settings"],
            "estimate_processes": request.get("estimate_processes"),
            "created": now(),
            "kind": request["kind"],
            "system": request.get("system", "candidate"),
            "runtime": review["runtime"],
            "resources": review["resources"],
            "image_hash": review["image_hash"],
            "rerun_completed": request.get("rerun_completed", False)
            if request["kind"] == "estimate_ram"
            else False,
            "candidates": request["candidates"],
            "entries": [],
            "task_ids": [],
        }

    def entry_matches(self, old: dict | None, entry: dict, request: dict) -> bool:
        if not old or old.get("status") != "eligible":
            return False
        if request["kind"] not in execution.CALCULATION_KINDS:
            return True
        fields = ["input_hash", "pseudopotentials"]
        if request["kind"] == "estimate_ram":
            fields.append("source_hash")
        if request.get("target") == "slurm":
            fields.extend(("resources", "scheduler_settings"))
        return all(old.get(field) == entry.get(field) for field in fields)

    def queue_options(
        self, request: dict, entry: dict, old: dict, batch: dict, position: int
    ) -> dict:
        token_required = (
            request["kind"] == "estimate_ram" or request.get("target") == "slurm"
        )
        return {
            "kind": request["kind"],
            "candidate": entry["candidate"],
            "system": request.get("system", "candidate"),
            "processes": entry["processes"],
            "target": request.get("target", "local"),
            "scheduler": request.get("scheduler"),
            "timeout": request.get("timeout"),
            "expected_hash": entry.get("input_hash"),
            "preview_id": old.get("preview_id") if token_required else None,
            "expected_pseudos": entry.get("pseudopotentials"),
            "runtime": batch["runtime"],
            "memory_gib": request.get("memory_gib", 4),
            "batch_id": batch["id"],
            "position": position,
            "checked_preflight": True,
        }

    def queue_entry(
        self, request: dict, entry: dict, old: dict, batch: dict, position: int
    ) -> dict:
        try:
            task = self.manager.queue(
                **self.queue_options(request, entry, old, batch, position)
            )
            batch["task_ids"].append(task["id"])
            return {
                "candidate": entry["candidate"],
                "processes": entry["processes"],
                "status": "queued",
                "task_id": task["id"],
                "reason": "",
            }
        except Exception as error:
            self.logger.exception(
                "Batch %s: cluster %s could not be queued",
                batch["id"],
                entry["candidate"],
            )
            return {
                "candidate": entry["candidate"],
                "status": "unavailable",
                "reason": str(error),
            }

    def accept_entry(
        self, request: dict, entry: dict, old: dict | None, batch: dict, position: int
    ) -> dict:
        if entry["status"] != "eligible":
            return entry
        if not self.entry_matches(old, entry, request):
            return {
                **entry,
                "status": "unavailable",
                "reason": "Entry changed since preview; review again",
            }
        assert old is not None
        return self.queue_entry(request, entry, old, batch, position)

    def accept_entries(self, request: dict, review: dict, batch: dict) -> None:
        expected = {
            (
                entry["candidate"],
                entry.get("processes", request.get("processes", 1)),
            ): entry
            for entry in request.get("expected", [])
        }
        for position, entry in enumerate(review["entries"]):
            old = expected.get((entry["candidate"], entry["processes"]))
            accepted = self.accept_entry(request, entry, old, batch, position)
            batch["entries"].append(
                {
                    key: value
                    for key, value in accepted.items()
                    if key in BATCH_ENTRY_FIELDS
                }
            )
            self.manager.store.put_batch(batch)
            if (position + 1) % PROGRESS_INTERVAL == 0 or position + 1 == len(
                review["entries"]
            ):
                self.logger.info(
                    "Batch %s: accepted %s/%s entries (%s queued)",
                    batch["id"],
                    position + 1,
                    len(review["entries"]),
                    len(batch["task_ids"]),
                )

    def submit(self, request: dict) -> dict:
        batch_id = request.get("id") or uuid.uuid4().hex
        if (
            not isinstance(batch_id, str)
            or not 1 <= len(batch_id) <= MAX_BATCH_ID_LENGTH
        ):
            raise ValueError("Invalid batch ID")
        # One guard covers acceptance; the scheduler waits for accepting=False.
        with self.manager.guard:
            previous = self.manager.store.batch(batch_id)
            if previous:
                return previous
            self.logger.info(
                "Batch %s: validating selected jobs before queueing", batch_id
            )
            review = self.submission_review(request)
            batch = self.new_batch(request, review, batch_id)
            self.manager.store.put_batch(batch)
            self.accept_entries(request, review, batch)
            batch["accepting"] = False
            self.manager.store.put_batch(batch)
            self.manager.wake.set()
            return batch

    def retry_candidates(self, batch: dict) -> list[str]:
        failed = {
            task["candidate"]
            for task_id in batch["task_ids"]
            if self.unsuccessful(task := self.manager.store.get(task_id))
        }
        failed.update(
            entry["candidate"]
            for entry in batch["entries"]
            if entry["status"] == "unavailable"
        )
        return [candidate for candidate in batch["candidates"] if candidate in failed]

    def unsuccessful(self, task: models.TaskRecord) -> bool:
        return task["status"] in ("failed", "canceled", "interrupted") or (
            task["kind"] == "qe"
            and task["status"] == "succeeded"
            and not task.get("evidence", {}).get("confirmed")
        )

    def retry(self, id: str) -> dict:
        batch = self.manager.store.batch(id)
        if not batch:
            raise ValueError("Unknown batch")
        resources = batch["resources"]
        return {
            "candidates": self.retry_candidates(batch),
            "kind": batch["kind"],
            "system": batch["system"],
            "runtime": batch["runtime"],
            "memory_gib": resources["memory_bytes"] / execution.GIB,
            "processes": resources["cpus"],
            "timeout": resources["timeout"],
            "retry": batch.get("target") != "slurm" or batch["kind"] != "estimate_ram",
            "target": batch.get("target", "local"),
            "scheduler": batch.get("scheduler"),
            "estimate_processes": batch.get("estimate_processes"),
        }


def validate_positive(value: float, label: str, integer: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value <= 0
        or (integer and int(value) != value)
    ):
        raise ValueError(
            f"{label} must be a positive {'integer' if integer else 'number'}"
        )
