import { Fragment, useMemo, useState } from "react";
import { Action, Candidate, Data, Task } from "./data";
import { PubChemLink } from "./PubChemLink";
import {
  Filter,
  matchesFilters,
  describeFilters,
  parseClusterIDs,
  matchesClusterIDs,
} from "./filters";
import { RunControls } from "./RunControls";

const numericFields = new Set([
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
}: {
  filters: Filter[];
  clearFilters: () => void;
  data: Data;
  busy: boolean;
  action: (path: string, body?: Action) => Promise<void>;
}) {
  const [view, setView] = useState("tiles"),
    [search, setSearch] = useState("");
  const [sort, setSort] = useState(""),
    [direction, setDirection] = useState("asc");
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
            matchesClusterIDs(c.id, idFilter.ranges) &&
            matchesFilters(c, filters),
        ),
        sort,
        direction,
      ),
    [data.candidates, idFilter, sort, direction, filters],
  );
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
          Sort by{" "}
          <select
            aria-label="Sort by"
            value={sort}
            onChange={(e) => setSort(e.target.value)}
          >
            <option value="">CSV order</option>
            {Object.keys(data.candidates[0]?.fields || {}).map((field) => (
              <option key={field} value={field}>
                {field === "cluster"
                  ? "Cluster ID"
                  : field === "n_points"
                    ? "Points"
                    : field === "medoid_CID"
                      ? "Representative CID"
                      : field === "medoid_SMILES"
                        ? "Representative SMILES"
                        : `Average ${field}`}
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
        {live && (
          <button disabled={busy} onClick={() => action("diagrams")}>
            Generate missing diagrams for all {data.candidates.length}
          </button>
        )}
      </div>
      <p
        id="cluster-filter-help"
        className={idFilter.error ? "error" : "filter-help"}
        role={idFilter.error ? "alert" : undefined}
      >
        {idFilter.error ||
          "Use an ID, a range (20-50), or a comma-separated list (2-10, 15)."}
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
      {view === "tiles" ? (
        <div className="candidate-grid">
          {candidates.map((c) => (
            <div className="candidate-tile" key={c.id}>
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
              <col style={{ width: "7%" }} />
              <col style={{ width: "7%" }} />
              <col style={{ width: "10%" }} />
              <col style={{ width: "10%" }} />
            </colgroup>
            <colgroup>
              <col style={{ width: "12%" }} />
              <col style={{ width: live ? "18%" : "54%" }} />
              {live && <col style={{ width: "36%" }} />}
            </colgroup>
            <thead>
              <tr className="group-headings">
                <th className="cluster-heading" scope="colgroup" colSpan={4}>
                  Cluster
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
                  ID
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
                        <a href={`#candidate/${c.id}`}>{c.id}</a>
                      </td>
                      <td className="cluster-cell">{c.fields.n_points}</td>
                      <td className="cluster-cell">
                        {Number(c.fields.MolecularWeight).toFixed(2)}
                      </td>
                      <td className="cluster-cell">
                        {Number(c.fields.XLogP).toFixed(2)}
                      </td>
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
                        <td colSpan={7}>
                          <button onClick={() => setRun(null)}>
                            Close run controls
                          </button>
                          <RunControls
                            key={run.candidate + run.system}
                            candidate={run.candidate}
                            system={run.system}
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
