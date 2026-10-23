import type { ReactElement } from "react";
import * as execution from "./execution";
import { useState } from "react";
import {
  Action,
  post,
  Preview,
  RuntimeInfo,
  InputVersion,
  SchedulerSettings,
} from "./data";
import { SlurmSettings } from "./SlurmSettings";
import { GeometryViewer } from "./GeometryViewer";
export function RunControls({
  candidate,
  system,
  action,
  runtimes,
  versions,
}: {
  runtimes?: RuntimeInfo;
  versions?: InputVersion[];
  candidate: string;
  system: string;
  action: (path: string, body?: Action) => Promise<void>;
}): ReactElement {
  const slurm = runtimes?.execution?.mode === execution.SLURM_MODE;
  const [scheduler, setScheduler] = useState<SchedulerSettings>({});
  const [reservation, setReservation] = useState("");
  const [processes, setProcesses] = useState(
      String(
        slurm
          ? execution.DEFAULT_SLURM_PROCESSES
          : execution.DEFAULT_LOCAL_PROCESSES,
      ),
    ),
    [target, setTarget] = useState(slurm ? "slurm" : "local"),
    [timeout, setTimeout] = useState("");
  const [runtime, setRuntime] = useState(
    runtimes?.apptainer.available ? "apptainer" : "native",
  );
  const [inputId, setInputId] = useState("");
  const [kind, setKind] = useState("qe");
  const [memory, setMemory] = useState("4");
  const [preview, setPreview] = useState<Preview | null>(null),
    [error, setError] = useState("");
  const automaticMemory =
    target === execution.SLURM_MODE &&
    kind === "qe" &&
    scheduler.memory_mode === "estimate";
  const body: Action = {
    kind,
    ...(inputId ? { input_id: inputId } : {}),
    runtime,
    memory_gib:
      target === "slurm"
        ? Number(
            reservation ||
              (automaticMemory ? execution.DEFAULT_LOCAL_PROCESSES : 0),
          )
        : Number(memory),
    scheduler:
      target === "slurm"
        ? {
            ...scheduler,
            memory_mode:
              kind === "estimate_ram"
                ? "manual"
                : (scheduler.memory_mode ?? "manual"),
          }
        : undefined,
    candidate,
    system,
    processes: Number(processes),
    target,
    ...(timeout ? { timeout: Number(timeout) } : {}),
  };
  const valid =
    (target === "slurm"
      ? automaticMemory || Number(reservation) > 0
      : Number(memory) > 0) &&
    Number.isInteger(Number(processes)) &&
    Number(processes) >= 1 &&
    (!timeout || Number(timeout) > 0);
  const changed = () => {
    setPreview(null);
    setError("");
  };
  return (
    <article>
      <h3>Run QE · {system}</h3>
      <div className="controls">
        <label>
          Input version{" "}
          <select
            value={inputId}
            onChange={(e) => {
              setInputId(e.target.value);
              changed();
            }}
          >
            <option value="">Prepared default</option>
            {versions
              ?.filter(
                (v) =>
                  v.system === system &&
                  v.candidate === (system === "tfa" ? "tfa" : candidate),
              )
              .map((v) => (
                <option key={v.input_id} value={v.input_id}>
                  {v.label}
                </option>
              ))}
          </select>
        </label>
        <label>
          Task{" "}
          <select
            value={kind}
            onChange={(e) => {
              setKind(e.target.value);
              changed();
            }}
          >
            <option value="qe">Run QE</option>
            <option value="estimate_ram">Estimate RAM</option>
          </select>
        </label>
        <label>
          Execution runtime{" "}
          <select
            value={runtime}
            onChange={(e) => {
              setRuntime(e.target.value);
              changed();
            }}
          >
            <option value="native">Native · installed tools</option>
            <option value="apptainer" disabled={!runtimes?.apptainer.available}>
              Apptainer ·{" "}
              {runtimes?.execution?.allocation || target === "slurm"
                ? "Slurm resources"
                : "hard memory cap"}
            </option>
          </select>
        </label>
        {(runtime === "apptainer" || target === execution.SLURM_MODE) &&
          !automaticMemory && (
            <label>
              Memory {target === "slurm" ? "reservation" : "per job"} (GiB){" "}
              <input
                type="number"
                min="0.125"
                step="0.125"
                value={target === "slurm" ? reservation : memory}
                onChange={(e) => {
                  (target === "slurm" ? setReservation : setMemory)(
                    e.target.value,
                  );
                  changed();
                }}
              />
            </label>
          )}
        {["qe", "estimate_ram"].includes(kind) && (
          <label>
            Processes{" "}
            <input
              type="number"
              min="1"
              step="1"
              value={processes}
              onChange={(e) => {
                setProcesses(e.target.value);
                changed();
              }}
            />
          </label>
        )}
        <label>
          Target{" "}
          <select
            value={target}
            onChange={(e) => {
              setTarget(e.target.value);
              changed();
            }}
          >
            <option value="local">Dashboard allocation / local</option>
            <option value="slurm" disabled={!slurm}>
              Separate Slurm allocation
            </option>
          </select>
        </label>
        {target === "slurm" && (
          <SlurmSettings
            value={scheduler}
            kind={kind}
            onChange={(value) => {
              setScheduler(value);
              changed();
            }}
          />
        )}
        <label>
          Timeout in seconds (
          {kind === "estimate_ram" ? "default 120" : "optional"}){" "}
          <input
            type="number"
            min="1"
            value={timeout}
            onChange={(e) => {
              setTimeout(e.target.value);
              changed();
            }}
          />
        </label>
        <button
          disabled={!valid}
          onClick={async () => {
            try {
              setPreview(await post<Preview>("preview", body));
              setError("");
            } catch (e) {
              setError(String(e));
            }
          }}
        >
          Preview command and input
        </button>
      </div>
      {error && <p className="error">{error}</p>}
      {preview && (
        <>
          <pre>{preview.command.join(" ")}</pre>
          {preview.resources && (
            <p>
              Separate Slurm job · {preview.resources.cpus} CPUs ·{" "}
              {(preview.resources.memory_bytes / 1024 ** 3).toFixed(2)} GiB
              reservation · {preview.memory_estimate?.basis || "Entered memory"}
              . Jobs continue when the dashboard stops.
            </p>
          )}
          <GeometryViewer
            key={preview.input_hash}
            geometries={{ [system]: preview.geometry }}
          />
          <details open>
            <summary>Exact input to be submitted</summary>
            <pre>{preview.input}</pre>
          </details>
          <button
            disabled={!valid}
            onClick={async () => {
              await action("tasks", {
                ...body,
                expected_hash: preview.input_hash,
                preview_id: preview.preview_id,
                expected_pseudos: preview.pseudopotentials,
              });
              setPreview(null);
            }}
          >
            {kind === "estimate_ram" ? "Estimate RAM" : "Run QE"}
          </button>
        </>
      )}
    </article>
  );
}
