import { useState } from "react";
import { post } from "./data";

type Review = {
  settings: {
    memory_mib: number;
    memory_mode: string;
    walltime_hours: number;
    processes: number;
  };
  entries: {
    candidate: string;
    system: string;
    status: string;
    reason?: string;
    input?: string;
    command?: string[];
    memory_mib?: number;
    memory_estimate?: { basis: string; base_bytes: number; processes: number };
  }[];
};

export function SlurmExportControls({ ids }: { ids: string[] }) {
  const [memory, setMemory] = useState("16");
  const [memoryMode, setMemoryMode] = useState("estimate");
  const [headroom, setHeadroom] = useState("50");
  const [hours, setHours] = useState("18");
  const [processes, setProcesses] = useState("8");
  const [partition, setPartition] = useState("ice-cpu");
  const [account, setAccount] = useState("coc");
  const [qos, setQos] = useState("coc-ice");
  const [includeTfa, setIncludeTfa] = useState(false);
  const [review, setReview] = useState<Review | null>(null);
  const [reviewKey, setReviewKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [page, setPage] = useState(0);
  const request = {
    candidates: ids,
    include_tfa: includeTfa,
    processes: Number(processes),
    memory_gib: Number(memory),
    memory_mode: memoryMode,
    headroom_percent: Number(headroom),
    walltime_hours: Number(hours),
    partition,
    account,
    qos,
  };
  const key = JSON.stringify(request);
  const current = key === reviewKey ? review : null;
  const valid =
    ids.length > 0 &&
    ids.length <= 256 &&
    Number(memory) > 0 &&
    Number(memory) <= 65536 &&
    Number(headroom) >= 0 &&
    Number(headroom) <= 1000 &&
    headroom !== "" &&
    Number.isInteger(Number(processes)) &&
    Number(processes) >= 1 &&
    Number(processes) <= 256 &&
    Number.isInteger(Number(hours)) &&
    Number(hours) >= 1 &&
    Number(hours) <= 168 &&
    [partition, account, qos].every((v) => !v || /^[A-Za-z0-9_.-]+$/.test(v));
  const eligible =
    current?.entries.filter((e) => e.status === "eligible").length || 0;
  const unavailable =
    current?.entries.filter((e) => e.status !== "eligible") || [];

  async function preview() {
    setBusy(true);
    setError("");
    setReview(null);
    try {
      setReview(await post<Review>("slurm/preview", request));
      setReviewKey(key);
      setPage(0);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function download() {
    if (!current) return;
    setBusy(true);
    setError("");
    try {
      const response = await fetch("/api/slurm/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...request, expected: current }),
      });
      if (!response.ok) {
        const result = await response.json();
        throw new Error(result.detail || "Export failed");
      }
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = url;
      link.download = "pfas-slurm-batch.zip";
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 10000);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <p>
        {ids.length} molecules · {ids.length * 2 + Number(includeTfa)}{" "}
        calculations. Export isolated candidates and TFA complexes using their
        latest prepared inputs. Each job runs on one node with the selected
        number of MPI processes and one CPU per process.
      </p>
      <div className="controls">
        <label>
          MPI processes per job{" "}
          <input
            type="number"
            min="1"
            max="256"
            step="1"
            value={processes}
            onChange={(e) => setProcesses(e.target.value)}
          />
        </label>
        <label>
          Memory sizing{" "}
          <select
            value={memoryMode}
            onChange={(e) => setMemoryMode(e.target.value)}
          >
            <option value="estimate">
              Automatic · RAM estimate + headroom
            </option>
            <option value="manual">Manual · same memory for every job</option>
          </select>
        </label>
        {memoryMode === "estimate" ? (
          <label>
            RAM headroom (%){" "}
            <input
              type="number"
              min="0"
              max="1000"
              step="5"
              value={headroom}
              onChange={(e) => setHeadroom(e.target.value)}
            />
          </label>
        ) : (
          <label>
            SLURM memory per job (GiB){" "}
            <input
              type="number"
              min="0.125"
              step="0.125"
              value={memory}
              onChange={(e) => setMemory(e.target.value)}
            />
          </label>
        )}
        <label>
          Walltime per job (hours){" "}
          <input
            type="number"
            min="1"
            max="168"
            step="1"
            value={hours}
            onChange={(e) => setHours(e.target.value)}
          />
        </label>
        <label>
          Partition{" "}
          <input
            value={partition}
            onChange={(e) => setPartition(e.target.value)}
          />
        </label>
        <label>
          Account{" "}
          <input value={account} onChange={(e) => setAccount(e.target.value)} />
        </label>
        <label>
          QOS <input value={qos} onChange={(e) => setQos(e.target.value)} />
        </label>
        <label>
          <input
            type="checkbox"
            checked={includeTfa}
            onChange={(e) => setIncludeTfa(e.target.checked)}
          />
          Include one isolated TFA reference
        </label>
        <button disabled={busy || !valid} onClick={preview}>
          Preview SLURM batch
        </button>
      </div>
      <p>
        Automatic sizing uses each calculation's matching RAM estimate plus
        headroom, rounded up to whole GiB. With only a one-process estimate, it
        is used directly as the total job baseline, without multiplying by the
        MPI count. Missing or stale estimates require a fresh estimate or manual
        memory. Confirm the PACE scheduling values for your allocation. Blank
        partition, account, or QOS uses the cluster default.
      </p>
      <p>
        The ZIP includes inputs, pseudopotentials, and submission scripts.
        Upload the extracted folder and your chemistry.sif to PACE, then run
        bash submit.sh. The image is transferred separately. Exporting creates
        no local or remote jobs.
      </p>
      {busy && <p role="status">Preparing SLURM export…</p>}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {current && (
        <div>
          <p role="status">
            {eligible} ready · {current.entries.length - eligible} unavailable ·{" "}
            {current.settings.memory_mode === "estimate"
              ? "Memory sized per calculation"
              : `${(current.settings.memory_mib / 1024).toFixed(2)} GiB per job`}{" "}
            · {current.settings.walltime_hours} hours ·{" "}
            {current.settings.processes} MPI processes
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
              <li key={`${entry.candidate}-${entry.system}`}>
                {entry.candidate} · {entry.system} · {entry.status}
                {entry.memory_mib !== undefined &&
                  ` · ${(entry.memory_mib / 1024).toFixed(2)} GiB reserved`}
                {entry.memory_estimate && (
                  <p>
                    {entry.memory_estimate.basis}:{" "}
                    {(entry.memory_estimate.base_bytes / 1024 ** 3).toFixed(2)}{" "}
                    GiB + {headroom}% headroom
                  </p>
                )}
                {entry.reason && ` · ${entry.reason}`}
                {entry.input && (
                  <details>
                    <summary>Command and exact input</summary>
                    <code>{entry.command?.join(" ")}</code>
                    <pre>{entry.input}</pre>
                  </details>
                )}
              </li>
            ))}
          </ol>
          {unavailable.length > 0 && (
            <div role="status">
              <p>Download is unavailable until these calculations are ready:</p>
              <ul>
                {unavailable.slice(0, 50).map((entry) => (
                  <li key={`${entry.candidate}-${entry.system}`}>
                    {entry.candidate} · {entry.system}: {entry.reason}
                    {entry.system === "tfa" && (
                      <>
                        {" "}
                        Uncheck Include one isolated TFA reference, or{" "}
                        <a href="#candidate/tfa">open TFA to estimate RAM</a>.
                      </>
                    )}
                  </li>
                ))}
              </ul>
              {unavailable.length > 50 && (
                <p>
                  See the preview pages for the remaining unavailable
                  calculations.
                </p>
              )}
            </div>
          )}
          <button
            disabled={busy || !valid || eligible !== current.entries.length}
            onClick={download}
          >
            Download SLURM batch ZIP
          </button>
        </div>
      )}
    </div>
  );
}
