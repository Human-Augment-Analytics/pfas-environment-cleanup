import type { Candidate, Task } from "./data";

export const ramFields: Record<string, string> = {
  candidate_ram_per_process_gib: "Candidate QE max RAM/process (GiB)",
  complex_ram_per_process_gib: "Complex QE max RAM/process (GiB)",
  candidate_ram_total_gib: "Candidate QE total RAM (GiB)",
  complex_ram_total_gib: "Complex QE total RAM (GiB)",
};

export function withTaskFields(
  candidates: Candidate[],
  tasks: Task[],
): Candidate[] {
  const summaries = new Map<string, NonNullable<Candidate["task_summary"]>>();
  // Do not depend on API or snapshot task ordering. A failed retry never erases a success.
  const newest = [...tasks].sort((a, b) => b.created.localeCompare(a.created));
  for (const task of newest) {
    const summary = summaries.get(task.candidate) || { ram: {} };
    if (task.kind === "prepare") {
      summary.latestPreparation ||= task;
      if (task.status === "succeeded") summary.prepared ||= task;
    }
    if (
      task.kind === "estimate_ram" &&
      task.status === "succeeded" &&
      ["candidate", "complex"].includes(task.system) &&
      task.ram_estimate?.per_process &&
      Number.isFinite(task.ram_estimate.per_process.bytes) &&
      task.ram_estimate.per_process.bytes >= 0
    )
      summary.ram[task.system] ||= task;
    summaries.set(task.candidate, summary);
  }
  return candidates.map((candidate) => {
    const summary = summaries.get(candidate.id) || { ram: {} };
    const fields = { ...candidate.fields };
    for (const system of ["candidate", "complex"]) {
      const estimate = summary.ram[system]?.ram_estimate;
      for (const [metric, value] of [
        ["per_process", estimate?.per_process],
        ["total", estimate?.total],
      ] as const) {
        fields[`${system}_ram_${metric}_gib`] =
          value && Number.isFinite(value.bytes) && value.bytes >= 0
            ? String(value.bytes / 1024 ** 3)
            : "";
      }
    }
    return { ...candidate, fields, task_summary: summary };
  });
}

export function preparationLabel(candidate: Candidate): string {
  if (candidate.task_summary?.prepared) return "Prepared";
  const latest = candidate.task_summary?.latestPreparation;
  if (!latest) return "Not run";
  return (
    (
      {
        queued: "Queued",
        running: "Running",
        failed: "Failed",
        canceled: "Canceled",
        interrupted: "Interrupted",
      } as Record<string, string>
    )[latest.status] || latest.status
  );
}

export function matchesPreparation(
  candidate: Candidate,
  selection: string,
): boolean {
  return (
    selection === "all" ||
    Boolean(candidate.task_summary?.prepared) === (selection === "prepared")
  );
}
