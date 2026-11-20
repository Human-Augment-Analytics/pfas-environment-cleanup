import { useState, type ReactElement } from "react";
import type { Task } from "./data";
import * as execution from "./execution";

export function ContinuationControls({ task }: { task: Task }): ReactElement {
  const [limit, setLimit] = useState(
    task.continuation_policy?.remaining ?? execution.DEFAULT_CONTINUATIONS,
  );
  const [message, setMessage] = useState("");
  const [pending, setPending] = useState(false);
  const save = async (remaining: number): Promise<void> => {
    setPending(true);
    try {
      const response = await fetch(
        `/api/tasks/${encodeURIComponent(task.id)}/continuation`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ limit: remaining }),
        },
      );
      if (!response.ok) throw new Error(await response.text());
      setMessage(
        remaining
          ? `Authorized up to ${remaining} further jobs under the current pipeline. Resume or recover the sequential chain to apply.`
          : "Automatic continuation disabled for this task.",
      );
    } catch (error) {
      setMessage(String(error));
    } finally {
      setPending(false);
    }
  };
  return (
    <fieldset>
      <legend>Automatic coordinate continuation</legend>
      <p>
        Opt in after reviewing this job's input and resources. Requires the
        sequential Slurm chain. Known timeouts and clean, incomplete relaxations
        can create fresh jobs; checkpoints are never reused.
      </p>
      <label>
        Maximum further jobs{" "}
        <input
          type="number"
          min={1}
          max={execution.MAX_CONTINUATIONS}
          value={limit}
          onChange={(event) => setLimit(Number(event.target.value))}
        />
      </label>
      <button
        disabled={
          pending ||
          !Number.isInteger(limit) ||
          limit < 1 ||
          limit > execution.MAX_CONTINUATIONS
        }
        onClick={() => void save(limit)}
      >
        Authorize continuation
      </button>
      <button disabled={pending} onClick={() => void save(0)}>
        Disable for this task
      </button>
      <p>
        Saved remaining allowance: {task.continuation_policy?.remaining ?? 0}
      </p>
      {task.continuation_parent && (
        <p>Continues task {task.continuation_parent}</p>
      )}
      {task.continuation_error && (
        <p className="error">{task.continuation_error}</p>
      )}
      {message && <p role="status">{message}</p>}
    </fieldset>
  );
}
