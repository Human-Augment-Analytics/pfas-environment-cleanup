import { useState, type ReactElement } from "react";
import type { Geometry, Task } from "./data";
import { GeometryViewer } from "./GeometryViewer";

type Step = {
  energy_ry: number;
  total_force_ry_bohr: number | null;
  scf_iterations: number | null;
};
type Report = {
  steps: Step[];
  calculation: string;
  failed_scf_cycles: number;
  force_threshold_ry_bohr: number | null;
  energy_threshold_ry: number | null;
  initial_geometry: Geometry;
  latest_geometry: Geometry | null;
  continuation_error: string;
};
const CHART_WIDTH = 600;
const CHART_HEIGHT = 180;
const CHART_PADDING = 25;

function ProgressChart({
  values,
  label,
}: {
  values: (number | null)[];
  label: string;
}): ReactElement | null {
  const points = values.flatMap((value, index) =>
    value === null ? [] : [{ value, index }],
  );
  if (!points.length) return null;
  const minimum = Math.min(...points.map((point) => point.value));
  const maximum = Math.max(...points.map((point) => point.value));
  const x = (index: number): number =>
    CHART_PADDING +
    (index * (CHART_WIDTH - 2 * CHART_PADDING)) /
      Math.max(values.length - 1, 1);
  const y = (value: number): number =>
    CHART_HEIGHT -
    CHART_PADDING -
    ((value - minimum) * (CHART_HEIGHT - 2 * CHART_PADDING)) /
      (maximum - minimum || 1);
  return (
    <figure className="optimization-chart">
      <figcaption>
        {label} · range {minimum} to {maximum}
      </figcaption>
      <svg
        viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
        role="img"
        aria-label={`${label} by completed SCF sample; exact values in the table below`}
      >
        <title>{label} by completed SCF sample</title>
        <path
          d={`M ${CHART_PADDING} ${CHART_PADDING} V ${CHART_HEIGHT - CHART_PADDING} H ${CHART_WIDTH - CHART_PADDING}`}
          fill="none"
          stroke="currentColor"
        />
        <polyline
          points={points
            .map((point) => `${x(point.index)},${y(point.value)}`)
            .join(" ")}
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        />
        {points.map((point) => (
          <circle
            key={point.index}
            cx={x(point.index)}
            cy={y(point.value)}
            r="3"
            fill="currentColor"
          >
            <title>
              Sample {point.index + 1}: {point.value}
            </title>
          </circle>
        ))}
      </svg>
      <small>
        Completed SCF sample 1–{values.length}; missing force values are
        omitted.
      </small>
    </figure>
  );
}

function RunStatistics({ report }: { report: Report }): ReactElement {
  const iterations = report.steps.flatMap((step) =>
    step.scf_iterations === null ? [] : [step.scf_iterations],
  );
  const first = report.steps[0];
  const last = report.steps[report.steps.length - 1];
  return (
    <>
      <p>
        {report.calculation} · {report.steps.length} completed SCF samples ·{" "}
        {iterations.reduce((sum, value) => sum + value, 0)} reported SCF
        iterations · {report.failed_scf_cycles} unconverged SCF cycles
      </p>
      {first && last && (
        <p>
          Energy change: {last.energy_ry - first.energy_ry} Ry · Latest energy:{" "}
          {last.energy_ry} Ry · Latest total force:{" "}
          {last.total_force_ry_bohr ?? "unavailable"} Ry/bohr
        </p>
      )}
      <p>
        Explicit input thresholds: energy{" "}
        {report.energy_threshold_ry ?? "QE default"} Ry · force{" "}
        {report.force_threshold_ry_bohr ?? "QE default"} Ry/bohr
      </p>
      <p>
        Samples are converged electronic calculations, including any final
        reevaluation. Total force alone does not certify geometry convergence.
      </p>
      <ProgressChart
        values={report.steps.map((step) => step.energy_ry)}
        label="Total energy (Ry)"
      />
      <ProgressChart
        values={report.steps.map((step) => step.total_force_ry_bohr)}
        label="Total force (Ry/bohr)"
      />
      <details>
        <summary>Exact sample values</summary>
        <div className="optimization-table">
          <table>
            <thead>
              <tr>
                <th>Sample</th>
                <th>Energy (Ry)</th>
                <th>Total force (Ry/bohr)</th>
                <th>SCF iterations</th>
              </tr>
            </thead>
            <tbody>
              {report.steps.map((step, index) => (
                <tr key={index}>
                  <td>{index + 1}</td>
                  <td>{step.energy_ry}</td>
                  <td>{step.total_force_ry_bohr ?? "unavailable"}</td>
                  <td>{step.scf_iterations ?? "unavailable"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </>
  );
}

export function OptimizationDetails({ task }: { task: Task }): ReactElement {
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const terminal = !["queued", "running"].includes(task.status);
  const load = async (): Promise<void> => {
    setLoading(true);
    try {
      const response = await fetch(
        `/api/tasks/${encodeURIComponent(task.id)}/optimization`,
      );
      if (!response.ok)
        throw new Error(
          `Cannot read optimization output: ${await response.text()}`,
        );
      setReport(await response.json());
      setError("");
    } catch (error) {
      setError(String(error));
    } finally {
      setLoading(false);
    }
  };
  return (
    <details className="optimization-details">
      <summary>Geometry optimization and run statistics</summary>
      <button disabled={loading} onClick={load}>
        {loading
          ? "Reading output…"
          : report
            ? "Refresh statistics"
            : "Load statistics"}
      </button>
      {error && <p className="error">{error}</p>}
      {report && (
        <>
          <RunStatistics report={report} />
          <GeometryViewer
            optimization
            geometries={{
              initial: report.initial_geometry,
              ...(report.latest_geometry
                ? { latest: report.latest_geometry }
                : {}),
            }}
          />
          {report.continuation_error ? (
            <p>Continuation unavailable: {report.continuation_error}</p>
          ) : terminal ? (
            <p>
              To continue from this geometry, select “Latest ATOMIC_POSITIONS
              from task {task.id}” under Input version in the run controls, then
              preview and submit. This starts a new run with the original
              coordinate constraints and fresh electronic state.
            </p>
          ) : (
            <p>
              Continuation becomes available when this task finishes or is
              stopped.
            </p>
          )}
        </>
      )}
    </details>
  );
}
