import React, { useState, useEffect } from "react";
import { createRoot } from "react-dom/client";
import { read, post, Data, Action, Task } from "./data";
import { AdvancedSearch } from "./AdvancedSearch";
import { Filter } from "./filters";
import { CandidateBrowser } from "./CandidateBrowser";
import { CandidateDetail } from "./CandidateDetail";
import { TaskHistory } from "./TaskHistory";
import "./style.css";
function App() {
  const [data, setData] = useState<Data | null>(null),
    [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [hash, setHash] = useState(location.hash.slice(1)),
    [busy, setBusy] = useState(false),
    [filters, setFilters] = useState<Filter[]>([]);
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
          <a href="#advanced-search">Advanced search</a>
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
            ) : hash === "advanced-search" ? (
              <AdvancedSearch candidates={data.candidates} filters={filters} apply={(next) => {setFilters(next); location.hash = "";}} />
            ) : null}
            <div hidden={Boolean(candidate) || hash === "tasks" || hash === "advanced-search"}>
              <CandidateBrowser data={data} busy={busy} action={action} filters={filters} clearFilters={() => setFilters([])} />
            </div>
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
