import { Task, Action } from "./data";
export function TaskHistory({
  tasks,
  live,
  action,
}: {
  tasks: Task[];
  live: boolean;
  action: (path: string, body?: Action) => Promise<void>;
}) {
  return (
    <section>
      <h2>Task history</h2>
      {!tasks.length && <p>No attempts yet.</p>}
      {tasks.map((t) => (
        <article key={t.id} className="attempt">
          <a
            href={t.candidate === "tfa" ? "#tfa" : `#candidate/${t.candidate}`}
          >
            {t.candidate === "tfa" ? "TFA" : `Cluster ${t.candidate}`}
          </a>
          {" · "}
          <strong>
            {t.kind} · {t.system} · {t.status}
          </strong>
          <p>
            {t.created}{" "}
            {t.elapsed_seconds !== undefined &&
              `· ${t.elapsed_seconds.toFixed(1)} seconds`}{" "}
            {t.exit_code !== undefined && `· exit ${t.exit_code}`}
          </p>
          {t.error && <p className="error">{t.error}</p>}
          {t.command && <code>{t.command.join(" ")}</code>}
          {t.evidence && (
            <p>
              Energy: {t.evidence.energy_ry ?? "unavailable"} Ry (
              {t.evidence.provisional ? "provisional" : "confirmed"}). JOB DONE:{" "}
              {String(t.evidence.job_done)} · SCF convergence:{" "}
              {String(t.evidence.scf_converged)} · Relaxation complete:{" "}
              {String(t.evidence.relaxation_completed)}
            </p>
          )}
          {live && ["queued", "running"].includes(t.status) && (
            <button onClick={() => action(`tasks/${t.id}/stop`)}>Stop</button>
          )}
          {live &&
            ["failed", "canceled", "interrupted"].includes(t.status) &&
            t.kind !== "qe" && (
              <button
                onClick={() =>
                  action("tasks", {
                    kind: t.kind,
                    candidate: t.candidate,
                    system: t.system,
                  })
                }
              >
                Retry
              </button>
            )}
          {live &&
            t.kind === "qe" &&
            !["queued", "running"].includes(t.status) && (
              <p>
                To retry, preview the current input and submit a new run below.
              </p>
            )}
          {Object.entries(t.artifacts).map(([name, url]) => (
            <a className="download" key={name} href={url} download>
              {name}
            </a>
          ))}
          {live && (
            <details>
              <summary>Bounded log tail</summary>
              <h3>stdout</h3>
              <pre>{t.stdout_tail || "No output yet."}</pre>
              <h3>stderr</h3>
              <pre>{t.stderr_tail || "No errors reported."}</pre>
            </details>
          )}
        </article>
      ))}
    </section>
  );
}
