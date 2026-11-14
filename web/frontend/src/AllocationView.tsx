import type { ReactElement } from "react";
import type { Task } from "./data";
import * as execution from "./execution";

const ESTIMATE_RAM = "estimate_ram";
const QUEUED = "queued";
const RUNNING = "running";

function allocationKey(task: Task, sharedWaitingEstimates: boolean): string {
  if (task.scheduler?.token) return `slurm:${task.scheduler.token}`;
  if (
    sharedWaitingEstimates &&
    task.kind === ESTIMATE_RAM &&
    task.batch_id &&
    task.status === QUEUED
  ) {
    return `estimate:${task.batch_id}`;
  }
  return `task:${task.id}`;
}

export function groupAllocations(
  tasks: Task[],
  sharedWaitingEstimates = false,
): Map<string, Task[]> {
  const groups = new Map<string, Task[]>();
  for (const task of tasks) {
    if (task.target !== execution.SLURM_MODE) continue;
    if (!task.scheduler && task.status !== QUEUED && task.status !== RUNNING)
      continue;
    const key = allocationKey(task, sharedWaitingEstimates);
    const checks = groups.get(key) || [];
    checks.push(task);
    groups.set(key, checks);
  }
  // Queue history arrives newest first; show checks in their execution order.
  for (const checks of groups.values()) checks.sort(compareChecks);
  return groups;
}

function compareChecks(first: Task, second: Task): number {
  if (first.position !== undefined && second.position !== undefined) {
    const positionOrder = first.position - second.position;
    if (positionOrder) return positionOrder;
  }
  return first.created.localeCompare(second.created);
}

function Check({ task }: { task: Task }): ReactElement {
  const estimate = task.ram_estimate;
  return (
    <li>
      <a
        href={
          task.candidate === "tfa" ? "#tfa" : `#candidate/${task.candidate}`
        }
      >
        Cluster {task.candidate}
      </a>{" "}
      · {task.system} · {task.kind} · {task.processes} processes ·{" "}
      {task.stop_requested ? `${task.status} (stop requested)` : task.status}
      {estimate && (
        <span>
          {" "}
          · RAM:{" "}
          {estimate.total
            ? `${estimate.total.value} ${estimate.total.unit} total`
            : "total unavailable"}
          {estimate.per_process &&
            ` · ${estimate.per_process.value} ${estimate.per_process.unit} per process`}
        </span>
      )}
      {!estimate && task.kind === ESTIMATE_RAM && " · RAM pending"}
    </li>
  );
}

export function allocationResources(checks: Task[]): {
  cpus: number;
  memoryBytes: number;
  walltimeHours: number;
} {
  // Sequential checks reuse resources; reservations must not be added together.
  const cpus = Math.max(
    ...checks.map((task) => task.resources?.cpus ?? task.processes),
  );
  const memoryBytes = Math.max(
    ...checks.map((task) => task.resources?.memory_bytes ?? 0),
  );
  const walltimeHours = Math.max(
    ...checks.map((task) => task.scheduler_settings?.walltime_hours ?? 0),
  );
  return { cpus, memoryBytes, walltimeHours };
}

function AllocationResources({ checks }: { checks: Task[] }): ReactElement {
  const { cpus, memoryBytes, walltimeHours } = allocationResources(checks);
  return (
    <span>
      {cpus} CPU(s) ·{" "}
      {memoryBytes
        ? `${(memoryBytes / execution.BYTES_PER_GIB).toFixed(2)} GiB reserved`
        : "Memory reservation unavailable"}{" "}
      · {walltimeHours ? `${walltimeHours}h walltime` : "Walltime unavailable"}
    </span>
  );
}

function Allocation({ checks }: { checks: Task[] }): ReactElement {
  const scheduler = checks.find((task) => task.scheduler)?.scheduler;
  const statuses = [...new Set(checks.map((task) => task.status))].join(", ");
  return (
    <article>
      <h3>
        One node allocation · Slurm {scheduler?.job_id || "awaiting submission"}
      </h3>
      <p>
        {scheduler?.state || statuses} · <AllocationResources checks={checks} />
        {scheduler?.reason && ` · ${scheduler.reason}`}
      </p>
      <ul>
        {checks.map((task) => (
          <Check key={task.id} task={task} />
        ))}
      </ul>
    </article>
  );
}

export function AllocationView({
  tasks,
  sharedWaitingEstimates = false,
}: {
  tasks: Task[];
  sharedWaitingEstimates?: boolean;
}): ReactElement {
  const groups = groupAllocations(tasks, sharedWaitingEstimates);
  return (
    <section aria-label="Slurm allocations">
      <h2>Slurm allocations</h2>
      <p>
        Shared RAM checks run sequentially in one allocation. CPU and memory
        reservations are the maximum required by a check, not the sum.
        Individual check history and controls remain below.
      </p>
      {!groups.size && <p>No Slurm allocations yet.</p>}
      {[...groups].map(([key, checks]) => (
        <Allocation key={key} checks={checks} />
      ))}
    </section>
  );
}
