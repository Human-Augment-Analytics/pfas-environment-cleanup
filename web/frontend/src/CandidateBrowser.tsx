import { Fragment, useMemo, useState, useEffect } from "react";
import { Action, Candidate, Data, Task } from "./data";
import { PubChemLink } from "./PubChemLink";
import {
  Filter,
  matchesFilters,
  describeFilters,
  parseClusterIDs,
  matchesClusterIDs,
} from "./filters";
import {
  ramFields,
  preparationLabel,
  matchesPreparation,
} from "./candidateResults";
import { BatchControls } from "./BatchControls";
import { RunControls } from "./RunControls";

const numericFields = new Set([
  ...Object.keys(ramFields),
  "cluster",
  "n_points",
  "medoid_CID",
  "MolecularWeight",
  "ExactMass",
  "Charge",
  "XLogP",
  "TPSA",
  "HBondDonorCount",
  "HBondAcceptorCount",
  "RotatableBondCount",
]);

function sorted(candidates: Candidate[], field: string, direction: string) {
  if (!field) return candidates;
  return [...candidates].sort((a, b) => {
    const left = a.fields[field],
      right = b.fields[field];
    // Empty and nonnumeric values always appear last.
    const missingLeft =
      !left || (numericFields.has(field) && !Number.isFinite(Number(left)));
    const missingRight =
      !right || (numericFields.has(field) && !Number.isFinite(Number(right)));
    if (missingLeft || missingRight)
      return Number(missingLeft) - Number(missingRight);
    const comparison = numericFields.has(field)
      ? Number(left) - Number(right)
      : left.localeCompare(right);
    return direction === "desc" ? -comparison : comparison;
  });
}

function taskIndex(tasks: Task[]) {
  const diagrams = new Map<string, Task>();
  const prepared = new Map<string, Task>();
  const active = new Set<string>();
  const latestDiagram = new Map<string, Task>();
  for (const task of tasks) {
    if (task.kind === "diagram" && !latestDiagram.has(task.candidate))
      latestDiagram.set(task.candidate, task);
    if (
      task.kind === "diagram" &&
      task.status === "succeeded" &&
      task.artifacts["diagram.png"] &&
      !diagrams.has(task.candidate)
    )
      diagrams.set(task.candidate, task);
    if (
      task.kind === "prepare" &&
      task.status === "succeeded" &&
      !prepared.has(task.candidate)
    )
      prepared.set(task.candidate, task);
    if (["queued", "running"].includes(task.status))
      active.add(`${task.candidate}/${task.kind}/${task.system}`);
  }
  return { diagrams, prepared, active, latestDiagram };
}

export function CandidateBrowser({
  data,
  busy,
  action,
  filters,
  clearFilters,
  refresh,
}: {
  filters: Filter[];
  clearFilters: () => void;
  refresh: () => Promise<void>;
  data: Data;
  busy: boolean;
  action: (path: string, body?: Action) => Promise<void>;
}) {
  const [view, setView] = useState("tiles"),
    [search, setSearch] = useState("");
  const [sort, setSort] = useState(""),
    [direction, setDirection] = useState("asc");
  const [preparationFilter, setPreparationFilter] = useState("all");
  const [ramField, setRamField] = useState("complex_ram_per_process_gib");
  const [minRAM, setMinRAM] = useState(""),
    [maxRAM, setMaxRAM] = useState("");
  const ramRangeInvalid =
    (minRAM !== "" &&
      (!Number.isFinite(Number(minRAM)) || Number(minRAM) < 0)) ||
    (maxRAM !== "" &&
      (!Number.isFinite(Number(maxRAM)) || Number(maxRAM) < 0)) ||
    (minRAM !== "" && maxRAM !== "" && Number(minRAM) > Number(maxRAM));
  const [run, setRun] = useState<{ candidate: string; system: string } | null>(
    null,
  );
  const index = useMemo(() => taskIndex(data.tasks), [data.tasks]);
  const idFilter = useMemo(() => parseClusterIDs(search), [search]);
  const candidates = useMemo(
    () =>
      sorted(
        data.candidates.filter(
          (c) =>
            !idFilter.error &&
            !ramRangeInvalid &&
            matchesPreparation(c, preparationFilter) &&
            matchesFilters(c, [
              ...(minRAM !== ""
                ? [{ field: ramField, operator: ">=", value: minRAM }]
                : []),
              ...(maxRAM !== ""
                ? [{ field: ramField, operator: "<=", value: maxRAM }]
                : []),
            ]) &&
            matchesClusterIDs(c.id, idFilter.ranges) &&
            matchesFilters(c, filters),
        ),
        sort,
        direction,
      ),
    [
      data.candidates,
      idFilter,
      sort,
      direction,
      filters,
      preparationFilter,
      ramField,
      minRAM,
      maxRAM,
      ramRangeInvalid,
    ],
  );
  const [selected, setSelected] = useState<Set<string>>(new Set());
  useEffect(() => {
    const visible = new Set(candidates.map((c) => c.id));
    setSelected(
      (previous) => new Set([...previous].filter((id) => visible.has(id))),
    );
  }, [candidates]);
  const selectedIDs = candidates
    .filter((c) => selected.has(c.id))
    .map((c) => c.id);
  const toggle = (id: string) =>
    setSelected((previous) => {
      const next = new Set(previous);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  const live = data.mode === "live";
  const diagramContent = (c: Candidate, tile: boolean) => {
    const diagram = index.diagrams.get(c.id);
    const attempt = index.latestDiagram.get(c.id);
    return diagram ? (
      <img
        loading="lazy"
        className={tile ? "tile-diagram" : "thumbnail"}
        src={diagram.artifacts["diagram.png"]}
        alt={`Representative molecule for cluster ${c.id}`}
      />
    ) : (
      <div
        className={tile ? "tile-placeholder" : "diagram-status"}
        title={attempt?.error || attempt?.stderr_tail}
      >
        {attempt ? `Diagram ${attempt.status}` : "Diagram not generated"}
      </div>
    );
  };
  return (
    <>
      <h1>All clusters</h1>
      <p>
        {data.candidates.length} candidates. Descriptors are cluster averages;
        CID and SMILES identify the representative molecule.
      </p>
      <div className="controls browser-controls">
        <div className="view-switch" role="group" aria-label="Candidate view">
          <button
            aria-pressed={view === "tiles"}
            onClick={() => setView("tiles")}
          >
            Tiles
          </button>
          <button
            aria-pressed={view === "rows"}
            onClick={() => setView("rows")}
          >
            Rows
          </button>
        </div>
        <label>
          Filter cluster ID{" "}
          <input
            type="search"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="e.g. 20-50 or 1,3,5 or 2-10, 15"
            aria-invalid={Boolean(idFilter.error)}
            aria-describedby="cluster-filter-help"
            aria-label="Filter cluster ID"
          />
        </label>
        <label>
          Preparation{" "}
          <select
            aria-label="Filter preparation"
            value={preparationFilter}
            onChange={(e) => setPreparationFilter(e.target.value)}
          >
            <option value="all">All candidates</option>
            <option value="prepared">Prepared successfully</option>
            <option value="not_prepared">No successful preparation</option>
          </select>
        </label>
        <label>
          RAM field{" "}
          <select
            aria-label="RAM field"
            value={ramField}
            onChange={(e) => setRamField(e.target.value)}
          >
            {Object.entries(ramFields).map(([key, label]) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Min RAM (GiB){" "}
          <input
            aria-label="Minimum RAM (GiB)"
            type="number"
            min="0"
            step="any"
            value={minRAM}
            onChange={(e) => setMinRAM(e.target.value)}
          />
        </label>
        <label>
          Max RAM (GiB){" "}
          <input
            aria-label="Maximum RAM (GiB)"
            type="number"
            min="0"
            step="any"
            value={maxRAM}
            onChange={(e) => setMaxRAM(e.target.value)}
          />
        </label>
        <button
          disabled={!minRAM && !maxRAM && preparationFilter === "all"}
          onClick={() => {
            setMinRAM("");
            setMaxRAM("");
            setPreparationFilter("all");
          }}
        >
          Clear RAM/preparation filters
        </button>
        <label>
          Sort by{" "}
          <select
            aria-label="Sort by"
            value={sort}
            onChange={(e) => setSort(e.target.value)}
          >
            <option value="">CSV order</option>
            {Object.keys(data.candidates[0]?.fields || {}).map((field) => (
              <option key={field} value={field}>
                {ramFields[field] ||
                  (field === "cluster"
                    ? "Cluster ID"
                    : field === "n_points"
                      ? "Points"
                      : field === "medoid_CID"
                        ? "Representative CID"
                        : field === "medoid_SMILES"
                          ? "Representative SMILES"
                          : `Average ${field}`)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Sort direction{" "}
          <select
            aria-label="Sort direction"
            value={direction}
            disabled={!sort}
            onChange={(e) => setDirection(e.target.value)}
          >
            <option value="asc">Ascending</option>
            <option value="desc">Descending</option>
          </select>
        </label>
      </div>
      <p
        id="cluster-filter-help"
        className={idFilter.error ? "error" : "filter-help"}
        role={idFilter.error ? "alert" : undefined}
      >
        {idFilter.error ||
          "Use an ID, a range (20-50), or a comma-separated list (2-10, 15)."}
      </p>
      {ramRangeInvalid && (
        <p className="error" role="alert">
          Enter nonnegative RAM limits with the minimum no greater than the
          maximum.
        </p>
      )}
      <p className="filter-help">
        RAM values are latest successful QE estimates, separately for each
        system. Missing estimates do not match RAM limits. Use Advanced search
        to combine RAM conditions.
      </p>
      {filters.length > 0 && (
        <p className="banner">
          Advanced filters: {describeFilters(filters)}.{" "}
          <a href="#advanced-search">Edit filters</a>{" "}
          <button onClick={clearFilters}>Clear advanced filters</button>
        </p>
      )}
      <p role="status">
        Showing {candidates.length} of {data.candidates.length} clusters.
      </p>
      {!candidates.length && <p>No clusters match your search.</p>}
      {live && (
        <>
          <div className="controls selection-controls">
            <button
              disabled={!candidates.length || busy}
              onClick={() => setSelected(new Set(candidates.map((c) => c.id)))}
            >
              Select all matching ({candidates.length})
            </button>
            <button
              disabled={!selectedIDs.length}
              onClick={() => setSelected(new Set())}
            >
              Clear selection
            </button>
            <span role="status">{selectedIDs.length} selected</span>
          </div>
          <BatchControls
            ids={selectedIDs}
            runtimes={data.queue?.runtimes}
            onQueued={() => {
              void refresh();
              location.hash = "queue";
            }}
          />
        </>
      )}
      {view === "tiles" ? (
        <div className="candidate-grid">
          {candidates.map((c) => (
            <div
              className={`candidate-tile ${selected.has(c.id) ? "selected" : ""}`}
              key={c.id}
            >
              {live && (
                <label className="tile-selection">
                  <input
                    type="checkbox"
                    aria-label={`Select cluster ${c.id}`}
                    checked={selected.has(c.id)}
                    onChange={() => toggle(c.id)}
                  />{" "}
                  Select
                </label>
              )}
              <a
                className="tile-entry"
                href={`#candidate/${c.id}`}
                aria-label={`Open cluster ${c.id}`}
              >
                <strong className="tile-cluster">Cluster {c.id}</strong>
                {diagramContent(c, true)}
              </a>
              <span className="tile-representative">
                Representative CID <PubChemLink cid={c.cid} />
              </span>
            </div>
          ))}
        </div>
      ) : (
        <div
          className="table-wrap"
          role="region"
          aria-label="Cluster and representative table"
          tabIndex={0}
        >
          <table className="candidate-table">
            <colgroup>
              <col style={{ width: "6%" }} />
              <col style={{ width: "6%" }} />
              <col style={{ width: "6%" }} />
              <col style={{ width: "6%" }} />
            </colgroup>
            <colgroup>
              <col style={{ width: "10%" }} />
              <col style={{ width: "13%" }} />
              <col style={{ width: "13%" }} />
            </colgroup>
            <colgroup>
              <col style={{ width: live ? "9%" : "10%" }} />
              <col style={{ width: live ? "10%" : "30%" }} />
              {live && <col style={{ width: "21%" }} />}
            </colgroup>
            <thead>
              <tr className="group-headings">
                <th className="cluster-heading" scope="colgroup" colSpan={4}>
                  Cluster
                </th>
                <th className="task-heading" scope="colgroup" colSpan={3}>
                  Preparation and QE estimates
                </th>
                <th
                  className="representative-heading"
                  scope="colgroup"
                  colSpan={live ? 3 : 2}
                >
                  Representative
                </th>
              </tr>
              <tr>
                <th className="cluster-heading" scope="col">
                  ID / Select
                </th>
                <th className="cluster-heading" scope="col">
                  Points
                </th>
                <th className="cluster-heading" scope="col">
                  Avg MW
                </th>
                <th className="cluster-heading" scope="col">
                  Avg XLogP
                </th>
                <th className="task-heading" scope="col">
                  Inputs
                </th>
                <th className="task-heading" scope="col">
                  Candidate RAM / process (GiB)
                </th>
                <th className="task-heading" scope="col">
                  Complex RAM / process (GiB)
                </th>
                <th className="representative-heading" scope="col">
                  CID
                </th>
                <th className="representative-heading" scope="col">
                  Diagram
                </th>
                {live && (
                  <th className="representative-heading" scope="col">
                    Actions
                  </th>
                )}
              </tr>
            </thead>
            <tbody>
              {candidates.map((c) => {
                const preparation = index.prepared.get(c.id);
                const preparing = index.active.has(`${c.id}/prepare/candidate`);
                return (
                  <Fragment key={c.id}>
                    <tr data-cluster={c.id}>
                      <td className="cluster-cell">
                        {live && (
                          <input
                            type="checkbox"
                            aria-label={`Select cluster ${c.id}`}
                            checked={selected.has(c.id)}
                            onChange={() => toggle(c.id)}
                          />
                        )}
                        <a href={`#candidate/${c.id}`}>{c.id}</a>
                      </td>
                      <td className="cluster-cell">{c.fields.n_points}</td>
                      <td className="cluster-cell">
                        {Number(c.fields.MolecularWeight).toFixed(2)}
                      </td>
                      <td className="cluster-cell">
                        {Number(c.fields.XLogP).toFixed(2)}
                      </td>
                      <td className="task-cell">
                        <strong>{preparationLabel(c)}</strong>
                        {c.task_summary?.prepared &&
                          c.task_summary.latestPreparation?.id !==
                            c.task_summary.prepared.id && (
                            <small>
                              Latest attempt:{" "}
                              {c.task_summary.latestPreparation?.status}
                            </small>
                          )}
                      </td>
                      {["candidate", "complex"].map((system) => {
                        const estimate = c.task_summary?.ram[system];
                        const report = estimate?.ram_estimate;
                        return (
                          <td
                            key={system}
                            className="task-cell"
                            title={
                              estimate
                                ? `${estimate.created} · ${estimate.runtime || "native"} · QE reported ${report?.per_process?.value} ${report?.per_process?.unit} per process`
                                : "No successful QE estimate"
                            }
                          >
                            {report?.per_process ? (
                              <>
                                <strong>
                                  {(
                                    report.per_process.bytes /
                                    1024 ** 3
                                  ).toFixed(3)}
                                </strong>
                                <small>
                                  Total:{" "}
                                  {report.total
                                    ? `${(report.total.bytes / 1024 ** 3).toFixed(3)} GiB`
                                    : "unavailable"}
                                </small>
                                <small>
                                  {estimate?.version || "Prepared default"} ·{" "}
                                  {estimate?.processes} process(es)
                                </small>
                              </>
                            ) : (
                              "Unavailable"
                            )}
                          </td>
                        );
                      })}
                      <td className="representative-cell">
                        <PubChemLink cid={c.cid} />
                      </td>
                      <td className="representative-cell">
                        <a href={`#candidate/${c.id}`}>
                          {diagramContent(c, false)}
                        </a>
                      </td>
                      {live && (
                        <td className="representative-cell actions-cell">
                          <div className="row-actions">
                            <button
                              disabled={busy || preparing}
                              onClick={() =>
                                action("tasks", {
                                  kind: "prepare",
                                  candidate: c.id,
                                  runtime: data.queue?.runtimes.apptainer
                                    .available
                                    ? "apptainer"
                                    : "native",
                                })
                              }
                            >
                              {preparing
                                ? "Preparing inputs…"
                                : "Prepare inputs"}
                            </button>
                            {[
                              {
                                system: "candidate",
                                label: "Run pw.x · single",
                              },
                              {
                                system: "complex",
                                label: "Run pw.x · TFA complex",
                              },
                            ].map(({ system, label }) => (
                              <button
                                key={system}
                                disabled={
                                  busy ||
                                  !preparation?.artifacts[`${system}.in`] ||
                                  index.active.has(`${c.id}/qe/${system}`)
                                }
                                title={
                                  !preparation?.artifacts[`${system}.in`]
                                    ? "Prepare inputs first"
                                    : "Review the command and input before submitting"
                                }
                                aria-expanded={
                                  run?.candidate === c.id &&
                                  run.system === system
                                }
                                onClick={() =>
                                  setRun({ candidate: c.id, system })
                                }
                              >
                                {label}
                              </button>
                            ))}
                          </div>
                          {preparing && (
                            <small>Use Refresh to update progress.</small>
                          )}
                        </td>
                      )}
                    </tr>
                    {live && run?.candidate === c.id && (
                      <tr className="run-row">
                        <td colSpan={10}>
                          <button onClick={() => setRun(null)}>
                            Close run controls
                          </button>
                          <RunControls
                            key={run.candidate + run.system}
                            candidate={run.candidate}
                            system={run.system}
                            runtimes={data.queue?.runtimes}
                            versions={data.input_versions}
                            action={action}
                          />
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
