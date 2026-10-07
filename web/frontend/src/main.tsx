import React, { useState, useEffect } from "react";
import { createRoot } from "react-dom/client";
import { read, post, Data, Action, Task } from "./data";
import { CandidateDetail } from "./CandidateDetail";
import { TaskHistory } from "./TaskHistory";
import "./style.css";
function App() {
  const [data, setData] = useState<Data | null>(null),
    [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [search, setSearch] = useState(""),
    [hash, setHash] = useState(location.hash.slice(1)),
    [busy, setBusy] = useState(false);
  const refresh = async () => {
    try {
      setData(await read());
      setError("");
    } catch (e) {
      setError(String(e));
    }
  };
  useEffect(() => {
    void refresh();
    const update = () => setHash(location.hash.slice(1));
    window.addEventListener("hashchange", update);
    return () => window.removeEventListener("hashchange", update);
  }, []);
  const action = async (path: string, body?: Action) => {
    setBusy(true);
    setNotice("");
    try {
      const result = await post<Task | Task[] | null>(path, body);
      if (Array.isArray(result)) {
        setNotice(
          `${result.length} diagram tasks queued. Open All tasks to view progress; use Refresh to update.`,
        );
      } else if (result?.kind) {
        setNotice(
          `${result.kind === "diagram" ? "Diagram" : result.kind === "prepare" ? "Preparation" : "QE"} task ${result.status}. Use Refresh to update progress.`,
        );
      } else {
        setNotice("Stop requested. Use Refresh to update task status.");
      }
      await refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };
  const candidate =
    data &&
    (hash === "tfa"
      ? data.reference
      : data.candidates.find((c) => `candidate/${c.id}` === hash));
  return (
    <>
      <header>
        <a href="#">PFAS / Candidate dashboard</a>
        <nav>
          <a href="#">Candidates</a>
          <a href="#tfa">TFA reference</a>
          <a href="#tasks">All tasks</a>
          <button disabled={busy} onClick={refresh}>
            Refresh
          </button>
        </nav>
      </header>
      <main aria-busy={busy}>
        {error && (
          <p role="alert" className="error">
            {error}
          </p>
        )}
        {notice && (
          <p role="status" className="banner">
            {notice}
          </p>
        )}
        {!data ? (
          <p>Loading candidates…</p>
        ) : (
          <>
            <p className="banner">
              {data.mode === "live"
                ? "Local live dashboard · One active task at a time"
                : `Read-only snapshot · Exported ${data.timestamp}`}
            </p>
            {data.mode === "snapshot" && (
              <p>
                To generate diagrams, prepare inputs, or run QE locally:{" "}
                <code>uv run --project web python -m dashboard</code> from the
                repository. See{" "}
                <a href="https://github.com/Human-Augment-Analytics/pfas-environment-cleanup">
                  repository documentation
                </a>
                .
              </p>
            )}
            {candidate ? (
              <CandidateDetail
                key={candidate.id}
                candidate={candidate}
                tasks={data.tasks.filter(
                  (t) =>
                    t.candidate === candidate.id ||
                    (candidate.id === "tfa" &&
                      t.kind === "prepare" &&
                      t.status === "succeeded" &&
                      Boolean(t.artifacts["tfa.in"])),
                )}
                live={data.mode === "live"}
                action={action}
              />
            ) : hash === "tasks" ? (
              <TaskHistory
                tasks={data.tasks}
                live={data.mode === "live"}
                action={action}
              />
            ) : (
              <>
                <h1>Clusters 500–667</h1>
                <p>
                  {data.candidates.length} candidates in CSV order. Descriptors
                  are cluster averages; CID and SMILES identify the
                  representative molecule.
                </p>
                <div className="controls">
                  <label>
                    Search candidates{" "}
                    <input
                      type="search"
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                      placeholder="Cluster, CID, SMILES or descriptor"
                    />
                  </label>
                  {data.mode === "live" && (
                    <button disabled={busy} onClick={() => action("diagrams")}>
                      Generate missing diagrams for all {data.candidates.length}
                    </button>
                  )}
                </div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Cluster</th>
                        <th>Points</th>
                        <th>Representative CID</th>
                        <th>Average molecular weight</th>
                        <th>Average XLogP</th>
                        <th>Diagram</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.candidates
                        .filter((c) =>
                          Object.values(c.fields)
                            .join(" ")
                            .toLowerCase()
                            .includes(search.toLowerCase()),
                        )
                        .map((c) => {
                          const diagram = data.tasks.find(
                            (t) =>
                              t.candidate === c.id &&
                              t.kind === "diagram" &&
                              t.status === "succeeded",
                          );
                          return (
                            <tr key={c.id}>
                              <td>
                                <a href={`#candidate/${c.id}`}>{c.id}</a>
                              </td>
                              <td>{c.fields.n_points}</td>
                              <td>{c.cid}</td>
                              <td>
                                {Number(c.fields.MolecularWeight).toFixed(2)}
                              </td>
                              <td>{Number(c.fields.XLogP).toFixed(2)}</td>
                              <td>
                                {diagram?.artifacts["diagram.png"] ? (
                                  <img
                                    className="thumbnail"
                                    src={diagram.artifacts["diagram.png"]}
                                    alt={`Cluster ${c.id}`}
                                  />
                                ) : (
                                  <span>Not generated</span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </>
        )}
      </main>
    </>
  );
}
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
