import type { ReactElement } from "react";
import {
  SLURM_MODE,
  DEFAULT_SLURM_PROCESSES,
  DEFAULT_LOCAL_PROCESSES,
  ESTIMATION_PROCESS_COUNTS,
} from "./execution";
import { useEffect, useState } from "react";
import {
  post,
  BatchRequest,
  BatchPreview,
  RuntimeInfo,
  SchedulerSettings,
} from "./data";
import { SlurmSettings } from "./SlurmSettings";
import { SlurmExportControls } from "./SlurmExportControls";

export function BatchControls({
  ids,
  runtimes,
  onQueued,
  initial,
}: {
  ids: string[];
  runtimes?: RuntimeInfo;
  onQueued: () => void;
  initial?: BatchRequest;
}): ReactElement {
  const slurm = runtimes?.execution?.mode === SLURM_MODE;
  const [scheduler, setScheduler] = useState<SchedulerSettings>(
    initial?.scheduler || {},
  );
  const [reservation, setReservation] = useState(
    initial?.target === "slurm" ? String(initial.memory_gib) : "",
  );
  const [location, setLocation] = useState(
    initial?.target || (slurm ? "slurm" : "local"),
  );
  const [rankGroup, setRankGroup] = useState(
    initial?.estimate_processes !== undefined
      ? Boolean(initial.estimate_processes?.length)
      : true,
  );
  const [kind, setKind] = useState(initial?.kind || "prepare");
  const [system, setSystem] = useState(initial?.system || "candidate");
  const [runtime, setRuntime] = useState(
    initial?.runtime ||
      (runtimes?.apptainer.available ? "apptainer" : "native"),
  );
  const [rerunCompleted, setRerunCompleted] = useState(
    initial?.rerun_completed || false,
  );
  const [processes, setProcesses] = useState(
    String(
      initial?.processes ||
        (slurm ? DEFAULT_SLURM_PROCESSES : DEFAULT_LOCAL_PROCESSES),
    ),
  );
  const [memory, setMemory] = useState(String(initial?.memory_gib || 4));
  const [timeout, setTimeout] = useState(
    initial?.timeout ? String(initial.timeout) : "",
  );
  const [preview, setPreview] = useState<BatchPreview | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [operation, setOperation] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [page, setPage] = useState(0);
  useEffect(() => {
    if (!busy) return;
    setElapsed(0);
    const started = Date.now();
    const timer = window.setInterval(
      () => setElapsed(Math.floor((Date.now() - started) / 1000)),
      1000,
    );
    return () => window.clearInterval(timer);
  }, [busy]);
  const [reviewKey, setReviewKey] = useState("");
  const [submissionID, setSubmissionID] = useState("");
  const remote =
    ["qe", "estimate_ram"].includes(kind) && location === SLURM_MODE;
  const automaticMemory =
    remote && kind === "qe" && scheduler.memory_mode === "estimate";
  const estimationProcesses =
    initial?.estimate_processes || ESTIMATION_PROCESS_COUNTS;
  const groupedEstimation = remote && kind === "estimate_ram" && rankGroup;
  const requestedProcesses = groupedEstimation
    ? Math.max(...estimationProcesses)
    : Number(processes);
  const request: BatchRequest = {
    candidates: ids,
    kind,
    system,
    runtime,
    processes: ["qe", "estimate_ram"].includes(kind)
      ? requestedProcesses
      : DEFAULT_LOCAL_PROCESSES,
    memory_gib: remote
      ? Number(reservation || (automaticMemory ? DEFAULT_LOCAL_PROCESSES : 0))
      : Number(memory),
    target: remote ? "slurm" : "local",
    scheduler: remote
      ? {
          ...scheduler,
          memory_mode:
            kind === "estimate_ram"
              ? "manual"
              : (scheduler.memory_mode ?? "manual"),
        }
      : undefined,
    estimate_processes: groupedEstimation ? estimationProcesses : undefined,
    timeout: timeout ? Number(timeout) : undefined,
    retry: initial?.retry || false,
    rerun_completed: kind === "estimate_ram" && rerunCompleted,
  };
  const key = JSON.stringify(request);
  const valid =
    (remote
      ? automaticMemory || Number(reservation) > 0
      : Number(memory) > 0) &&
    (!["qe", "estimate_ram"].includes(kind) ||
      (Number.isInteger(requestedProcesses) &&
        requestedProcesses >= DEFAULT_LOCAL_PROCESSES)) &&
    (!timeout || Number(timeout) > 0);
  const current = key === reviewKey ? preview : null;
  async function review() {
    setOperation(`Reviewing ${ids.length} selected jobs`);
    setBusy(true);
    setError("");
    try {
      setPreview(await post<BatchPreview>("batches/preview", request));
      setPage(0);
      setReviewKey(key);
      setSubmissionID(crypto.randomUUID());
    } catch (e) {
      setError(String(e));
      setPreview(null);
    } finally {
      setBusy(false);
    }
  }
  async function submit() {
    if (!current) return;
    setOperation(`Queueing ${counts?.eligible || 0} jobs`);
    setBusy(true);
    setError("");
    try {
      await post("batches", {
        ...request,
        id: submissionID,
        expected: current.entries.map(
          ({
            candidate,
            processes,
            resources,
            scheduler_settings,
            status,
            input_hash,
            source_hash,
            preview_id,
            pseudopotentials,
          }) => ({
            candidate,
            processes,
            resources,
            scheduler_settings,
            status,
            input_hash,
            source_hash,
            preview_id,
            pseudopotentials,
          }),
        ),
        image_hash: current.image_hash,
      });
      setPreview(null);
      onQueued();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  const counts = current?.entries.reduce<Record<string, number>>((all, e) => {
    all[e.status] = (all[e.status] || 0) + 1;
    return all;
  }, {});
  const actionSelector = (
    <label>
      Batch action{" "}
      <select
        aria-label="Batch action"
        value={`${kind}/${system}`}
        onChange={(e) => {
          const [k, s] = e.target.value.split("/");
          setKind(k);
          setSystem(s);
        }}
      >
        <option value="diagram/candidate">Generate missing diagrams</option>
        <option value="prepare/candidate">Prepare missing inputs</option>
        <option value="qe/candidate">Run QE · isolated candidate</option>
        <option value="qe/complex">Run QE · TFA complex</option>
        <option value="estimate_ram/candidate">
          Estimate RAM · isolated candidate
        </option>
        <option value="estimate_ram/complex">Estimate RAM · TFA complex</option>
        {!initial && (
          <option value="slurm/both">
            Export SLURM batch · candidate + complex
          </option>
        )}
      </select>
    </label>
  );
  if (kind === "slurm") {
    return (
      <section className="batch-controls">
        <h3>Actions for selected molecules</h3>
        {actionSelector}
        <SlurmExportControls ids={ids} />
      </section>
    );
  }
  return (
    <section className="batch-controls">
      <h3>
        {initial ? "Retry unsuccessful jobs" : "Actions for selected molecules"}
      </h3>
      <p>
        {ids.length} selected · Jobs start in the displayed order. Each molecule
        has its own attempt.
      </p>
      <div className="controls">
        {actionSelector}
        <label>
          Execution runtime{" "}
          <select
            aria-label="Execution runtime"
            value={runtime}
            onChange={(e) => setRuntime(e.target.value)}
          >
            <option value="native">Native · installed tools</option>
            <option value="apptainer" disabled={!runtimes?.apptainer.available}>
              Apptainer ·{" "}
              {runtimes?.execution?.allocation || remote
                ? "Slurm resources"
                : "hard memory cap"}
            </option>
          </select>
        </label>
        {(runtime === "apptainer" || remote) && !automaticMemory && (
          <label>
            Memory {remote ? "reservation" : "per job"} (GiB){" "}
            <input
              aria-label="Memory per job (GiB)"
              type="number"
              min="0.125"
              step="0.125"
              value={remote ? reservation : memory}
              onChange={(e) =>
                (remote ? setReservation : setMemory)(e.target.value)
              }
            />
          </label>
        )}
        {["qe", "estimate_ram"].includes(kind) && (
          <label>
            Processes per job{" "}
            <input
              aria-label="Processes per job"
              type="number"
              min="1"
              step="1"
              disabled={groupedEstimation}
              value={groupedEstimation ? String(requestedProcesses) : processes}
              onChange={(e) => setProcesses(e.target.value)}
            />
          </label>
        )}
        {["qe", "estimate_ram"].includes(kind) && (
          <label>
            Execution location{" "}
            <select
              value={location}
              onChange={(e) => setLocation(e.target.value)}
            >
              <option value="local">Dashboard allocation / local</option>
              <option value="slurm" disabled={!slurm}>
                Separate Slurm allocation
              </option>
            </select>
          </label>
        )}
        {remote && (
          <SlurmSettings
            value={scheduler}
            kind={kind}
            onChange={setScheduler}
          />
        )}
        {remote && kind === "estimate_ram" && (
          <label>
            <input
              type="checkbox"
              checked={rankGroup}
              onChange={(e) => setRankGroup(e.target.checked)}
            />{" "}
            Measure {estimationProcesses.join(", ")} processes in one{" "}
            {Math.max(...estimationProcesses)}-CPU allocation
          </label>
        )}
        <label>
          Timeout (seconds){" "}
          <input
            aria-label="Batch timeout (seconds)"
            type="number"
            min="1"
            value={timeout}
            placeholder={
              kind === "prepare"
                ? "900"
                : kind === "diagram"
                  ? "60"
                  : kind === "estimate_ram"
                    ? "120"
                    : "No timeout"
            }
            onChange={(e) => setTimeout(e.target.value)}
          />
        </label>
        {kind === "estimate_ram" && (
          <label className="rerun-option">
            <input
              type="checkbox"
              checked={rerunCompleted}
              onChange={(e) => setRerunCompleted(e.target.checked)}
            />
            Rerun existing estimates
          </label>
        )}
        <button disabled={busy || !ids.length || !valid} onClick={review}>
          Preview selected jobs
        </button>
      </div>
      {kind === "estimate_ram" && (
        <p>
          Existing successful estimates are skipped unless you select Rerun
          existing estimates. Results appear in the queue and task history.
        </p>
      )}
      {remote && (
        <p>
          Jobs continue after the dashboard stops. RAM checks run sequentially
          in one allocation; stopping that job cancels its remaining checks.
          Completed results are preserved.
        </p>
      )}
      {runtime === "native" && !remote && (
        <p>
          Native jobs run one at a time. Their memory shares the dashboard’s
          resources.
        </p>
      )}
      {!runtimes?.apptainer.available && (
        <p className="filter-help">
          For container execution, install Apptainer, build the chemistry image,
          and set PFAS_APPTAINER_IMAGE. See the dashboard documentation.
        </p>
      )}
      {busy && (
        <p role="status">
          {operation} · {elapsed}s elapsed…
        </p>
      )}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {current && (
        <div>
          <p role="status">
            {counts?.eligible || 0} eligible · {counts?.completed || 0}{" "}
            completed · {counts?.active || 0} already active ·{" "}
            {counts?.unavailable || 0} unavailable
          </p>
          <p>
            {runtime} · {current.resources.cpus} CPU(s) per job ·{" "}
            {current.resources.hard_memory_limit
              ? `${(current.resources.memory_bytes / 1024 ** 3).toFixed(2)} GiB cap`
              : remote
                ? `${(current.resources.memory_bytes / 1024 ** 3).toFixed(2)} GiB reservation (Slurm)`
                : "No separate task memory cap"}{" "}
            ·{" "}
            {current.resources.timeout
              ? `${current.resources.timeout}s timeout`
              : "No timeout"}
          </p>
          <p>
            Showing {page * 50 + 1}–
            {Math.min((page + 1) * 50, current.entries.length)} of{" "}
            {current.entries.length} jobs
          </p>
          <button disabled={page === 0} onClick={() => setPage(page - 1)}>
            Previous jobs
          </button>
          <button
            disabled={(page + 1) * 50 >= current.entries.length}
            onClick={() => setPage(page + 1)}
          >
            Next jobs
          </button>
          <ol className="batch-preview" start={page * 50 + 1}>
            {current.entries.slice(page * 50, (page + 1) * 50).map((entry) => (
              <li key={`${entry.candidate}-${entry.processes}`}>
                <a href={`#candidate/${entry.candidate}`}>
                  Cluster {entry.candidate}
                </a>{" "}
                · {entry.processes || 1} processes · {entry.status}
                {entry.resources &&
                  ` · ${(entry.resources.memory_bytes / 1024 ** 3).toFixed(2)} GiB reserved`}
                {entry.memory_estimate && ` · ${entry.memory_estimate.basis}`}
                {entry.reason && ` · ${entry.reason}`}
                {entry.input && (
                  <JobInput input={entry.input} command={entry.command} />
                )}
              </li>
            ))}
          </ol>
          <button
            disabled={busy || !counts?.eligible || !valid}
            onClick={submit}
          >
            Queue {counts?.eligible || 0} jobs
          </button>
        </div>
      )}
    </section>
  );
}

function JobInput({
  input,
  command,
}: {
  input: string;
  command?: string[];
}): ReactElement {
  const [open, setOpen] = useState(false);
  return (
    <details onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary>Command and exact input</summary>
      {open && (
        <>
          <code>{command?.join(" ")}</code>
          <pre>{input}</pre>
        </>
      )}
    </details>
  );
}
