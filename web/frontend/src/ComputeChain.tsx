import { useState, type ReactElement } from "react";
import type { QueueData, Task } from "./data";
import * as execution from "./execution";

const AUDIT_HISTORY_LIMIT = 10;
const CHAIN_PATH = "queue/chain";
const RECOVER_PATH = "queue/recover";
const ORDER_PATH = "queue/order";
const DISPATCHER_REJECTED_PATH = "queue/dispatcher/rejected";
const SUBMISSION_UNKNOWN_STATE = "SUBMISSION_UNKNOWN";
const TERMINAL_SLURM_STATES: readonly string[] = [
  "COMPLETED",
  "FAILED",
  "CANCELLED",
  "TIMEOUT",
  "OUT_OF_MEMORY",
  "NODE_FAIL",
  "BOOT_FAIL",
  "PREEMPTED",
  "DEADLINE",
  "REVOKED",
];

type Mutate = (path: string, body?: unknown, message?: string) => Promise<void>;
type ChainProps = { data: QueueData; pending: boolean; mutate: Mutate };

function taskLabel(task: Task | undefined, id: string): string {
  return task
    ? `${task.kind} · ${task.system} · Cluster ${task.candidate} (${id})`
    : id;
}

export function ComputeChain({
  data,
  pending,
  mutate,
}: ChainProps): ReactElement {
  const { chain, ordered_ids: ids } = data;
  const tasks = new Map(data.tasks.map((task) => [task.id, task]));
  const activeId = chain.active_task_id;
  const nextId = ids[0];
  const dispatcherOutstanding =
    chain.dispatcher !== null &&
    !TERMINAL_SLURM_STATES.includes(chain.dispatcher.state);
  const activeTask = activeId ? tasks.get(activeId) : undefined;
  const activeOutstanding =
    Boolean(activeId) &&
    !TERMINAL_SLURM_STATES.includes(activeTask?.scheduler?.state ?? "");
  const computeOutstanding = data.tasks.some(
    (task) =>
      task.target === "slurm" &&
      task.scheduler &&
      !TERMINAL_SLURM_STATES.includes(task.scheduler.state),
  );
  const canDisable =
    !activeOutstanding && !computeOutstanding && !dispatcherOutstanding;
  return (
    <section aria-labelledby="compute-chain-heading">
      <h2 id="compute-chain-heading">Sequential Slurm chain</h2>
      <p role="status">
        {chain.enabled ? (chain.paused ? "Paused" : "Started") : "Disabled"} ·{" "}
        {ids.length} waiting
      </p>
      <p>
        Run one computation at a time. Chain pause is independent of the local
        queue pause and lets the active computation finish. Disabling restores
        independent Slurm submission.
      </p>
      <p>
        Enable a paused chain before adding tasks to arrange their order before
        they start. When the chain is disabled, Slurm tasks submit
        independently.
      </p>
      <div className="controls">
        <button
          disabled={pending || chain.enabled}
          onClick={() => mutate(CHAIN_PATH, { enabled: true, paused: true })}
        >
          Enable paused chain
        </button>
        <button
          disabled={pending || chain.enabled}
          onClick={() => mutate(CHAIN_PATH, { enabled: true, paused: false })}
        >
          Enable and start chain
        </button>
        <button
          disabled={pending || !chain.enabled}
          onClick={() => mutate(CHAIN_PATH, { paused: !chain.paused })}
        >
          {chain.paused ? "Resume chain" : "Pause chain"}
        </button>
        <button
          disabled={pending || !chain.enabled || !canDisable}
          aria-describedby="chain-disable-help"
          onClick={() => mutate(CHAIN_PATH, { enabled: false })}
        >
          Disable chain
        </button>
        <button
          disabled={pending}
          onClick={() =>
            mutate(RECOVER_PATH, undefined, "Chain recovery requested.")
          }
        >
          Recover chain
        </button>
      </div>
      <p id="chain-disable-help">
        Disable is available only when the chain is idle, with no active
        computation or pending chain advancement. Pause and wait for current
        work to finish.
      </p>
      <ChainDiagnostics data={data} pending={pending} mutate={mutate} />
      <p>
        Active: {activeId ? taskLabel(tasks.get(activeId), activeId) : "None"}
      </p>
      <p>Next: {nextId ? taskLabel(tasks.get(nextId), nextId) : "None"}</p>
      <p>
        Add tasks using the candidate or batch controls with Slurm selected. GPU
        scheduling is unsupported.
      </p>
      <h3>Waiting order</h3>
      <p>
        Reorder or remove waiting tasks before they are submitted to Slurm.
        Refresh and try again if the queue changes while you edit it.
      </p>
      {!ids.length && <p>No Slurm tasks waiting.</p>}
      <ol>
        {ids.map((id, index) => (
          <WaitingTask
            key={id}
            id={id}
            task={tasks.get(id)}
            position={index + 1}
            ids={ids}
            pending={pending}
            mutate={mutate}
          />
        ))}
      </ol>
    </section>
  );
}

function ChainDiagnostics({ data, pending, mutate }: ChainProps): ReactElement {
  const { chain } = data;
  const dispatcher = chain.dispatcher;
  return (
    <div>
      <p>
        Dispatcher:{" "}
        {dispatcher
          ? `Job ${dispatcher.job_id || "submission unresolved"} · ${dispatcher.state} · ${dispatcher.reason}`
          : "None"}
      </p>
      {dispatcher?.job_id === null &&
        dispatcher.state === SUBMISSION_UNKNOWN_STATE && (
          <div>
            <p>
              Check Slurm for this dispatcher before confirming rejection. A
              delayed reply can mean it was accepted. After confirming
              rejection, use Recover chain to continue.
            </p>
            <button
              disabled={pending}
              onClick={() =>
                mutate(
                  DISPATCHER_REJECTED_PATH,
                  { token: dispatcher.token },
                  "Dispatcher rejection recorded. Use Recover chain to continue.",
                )
              }
            >
              I confirmed Slurm rejected this dispatcher
            </button>
          </div>
        )}
      <p>Last attempt: {chain.last_attempt || "Never"}</p>
      <ChainAudit events={chain.events} />
      {chain.error && (
        <p className="error" role="alert">
          Chain error: {chain.error}
        </p>
      )}
      <p>
        Computation failures appear in job history. Use Recover chain to check
        for interrupted progress and continue when possible. Review failed
        computations in history to retry them.
      </p>
    </div>
  );
}

type WaitingTaskProps = {
  id: string;
  task: Task | undefined;
  position: number;
  ids: string[];
  pending: boolean;
  mutate: Mutate;
};

function WaitingTask({
  id,
  task,
  position,
  ids,
  pending,
  mutate,
}: WaitingTaskProps): ReactElement {
  const [destination, setDestination] = useState("");
  const targetPosition = Number(destination);
  const validPosition =
    Number.isInteger(targetPosition) &&
    targetPosition >= 1 &&
    targetPosition <= ids.length;
  async function move(to: number): Promise<void> {
    // Submit the complete displayed order as the optimistic concurrency check.
    const reordered = ids.filter((taskId) => taskId !== id);
    reordered.splice(to - 1, 0, id);
    await mutate(ORDER_PATH, { ids: reordered, expected: [...ids] });
    setDestination("");
  }
  return (
    <li>
      <p>
        <strong>
          Position {position}: {taskLabel(task, id)}
        </strong>
      </p>
      <p>
        Owner: {task?.owner || "Not recorded"} · Memory:{" "}
        {task?.resources
          ? `${(task.resources.memory_bytes / execution.BYTES_PER_GIB).toFixed(2)} GiB`
          : "Unavailable"}{" "}
        · CPUs: {task?.resources?.cpus ?? task?.processes ?? "Unavailable"} ·
        Walltime:{" "}
        {task?.scheduler_settings?.walltime_hours !== undefined
          ? `${task.scheduler_settings.walltime_hours} hours`
          : "Unavailable"}{" "}
        · GPU unsupported
      </p>
      <form
        className="controls"
        onSubmit={(event) => {
          event.preventDefault();
          if (validPosition && !pending) void move(targetPosition);
        }}
      >
        <button
          type="button"
          disabled={pending || position === 1}
          onClick={() => move(position - 1)}
        >
          Move up
        </button>
        <button
          type="button"
          disabled={pending || position === ids.length}
          onClick={() => move(position + 1)}
        >
          Move down
        </button>
        <label>
          Move {id} to position{" "}
          <input
            type="number"
            min={1}
            max={ids.length}
            step={1}
            value={destination}
            disabled={pending}
            onChange={(event) => setDestination(event.target.value)}
          />
        </label>
        <button
          type="submit"
          disabled={pending || !validPosition || targetPosition === position}
        >
          Move to position
        </button>
        <button
          type="button"
          disabled={pending || task?.stop_requested}
          onClick={() =>
            mutate(
              `queue/tasks/${id}/remove`,
              undefined,
              "Waiting task removed. History is retained.",
            )
          }
        >
          Remove from waiting queue
        </button>
      </form>
    </li>
  );
}

function ChainAudit({
  events,
}: {
  events: QueueData["chain"]["events"];
}): ReactElement {
  const recent = (events ?? []).slice(-AUDIT_HISTORY_LIMIT).reverse();
  return (
    <details>
      <summary>Recent chain actions</summary>
      {!recent.length && <p>No chain actions recorded.</p>}
      <ol>
        {recent.map((event, index) => (
          <li key={`${event.timestamp}-${event.action}-${index}`}>
            <p>
              <time>{event.timestamp}</time> · {event.action} · {event.outcome}
            </p>
            {event.detail && <p>{event.detail}</p>}
            {event.error && (
              <p className="error">Recorded error: {event.error}</p>
            )}
          </li>
        ))}
      </ol>
    </details>
  );
}
