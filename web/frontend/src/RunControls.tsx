import { useState } from "react";
import { Action, post, Preview } from "./data";
export function RunControls({
  candidate,
  system,
  action,
}: {
  candidate: string;
  system: string;
  action: (path: string, body?: Action) => Promise<void>;
}) {
  const [processes, setProcesses] = useState(1),
    [target, setTarget] = useState("local"),
    [timeout, setTimeout] = useState("");
  const [preview, setPreview] = useState<Preview | null>(null),
    [error, setError] = useState("");
  const body: Action = {
    kind: "qe",
    candidate,
    system,
    processes,
    target,
    ...(timeout ? { timeout: Number(timeout) } : {}),
  };
  const changed = () => {
    setPreview(null);
    setError("");
  };
  return (
    <article>
      <h3>Run QE · {system}</h3>
      <div className="controls">
        <label>
          Processes{" "}
          <input
            type="number"
            min="1"
            step="1"
            value={processes}
            onChange={(e) => {
              setProcesses(Number(e.target.value));
              changed();
            }}
          />
        </label>
        <label>
          Target{" "}
          <select
            value={target}
            onChange={(e) => {
              setTarget(e.target.value);
              changed();
            }}
          >
            <option value="local">Local</option>
            <option value="slurm">Slurm (not implemented)</option>
            <option value="slurm-array">Slurm array (not implemented)</option>
          </select>
        </label>
        <label>
          Timeout in seconds (optional){" "}
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
          <details open>
            <summary>Exact input to be submitted</summary>
            <pre>{preview.input}</pre>
          </details>
          <button
            onClick={async () => {
              await action("tasks", {
                ...body,
                expected_hash: preview.input_hash,
              });
              setPreview(null);
            }}
          >
            Run QE
          </button>
        </>
      )}
    </article>
  );
}
