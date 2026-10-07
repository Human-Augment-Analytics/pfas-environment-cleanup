import { useState } from "react";
import { Candidate, Task, Action } from "./data";
import { TaskHistory } from "./TaskHistory";
import { RunControls } from "./RunControls";
export function CandidateDetail({
  candidate,
  tasks,
  live,
  action,
}: {
  candidate: Candidate;
  tasks: Task[];
  live: boolean;
  action: (path: string, body?: Action) => Promise<void>;
}) {
  const [input, setInput] = useState(""),
    [error, setError] = useState("");
  const diagram = tasks.find(
    (t) =>
      t.kind === "diagram" &&
      t.status === "succeeded" &&
      t.artifacts["diagram.png"],
  );
  const prepared = tasks.find(
    (t) => t.kind === "prepare" && t.status === "succeeded",
  );
  return (
    <>
      <h2>
        {candidate.id === "tfa"
          ? "Shared neutral TFA reference"
          : `Cluster ${candidate.id}`}
      </h2>
      <p>Representative molecule · PubChem CID {candidate.cid}</p>
      <pre>{candidate.smiles}</pre>
      {diagram ? (
        <img
          className="diagram"
          src={diagram.artifacts["diagram.png"]}
          alt={`Representative molecule for cluster ${candidate.id}`}
        />
      ) : (
        <div className="placeholder">Diagram has not been generated.</div>
      )}
      {live && (
        <div className="controls">
          <button
            onClick={() =>
              action("tasks", { kind: "diagram", candidate: candidate.id })
            }
          >
            Generate diagram
          </button>
          <button
            onClick={() =>
              action("tasks", { kind: "prepare", candidate: candidate.id })
            }
          >
            Prepare inputs
          </button>
        </div>
      )}
      <h3>
        {candidate.id === "tfa"
          ? "Reference identity"
          : "Original CSV fields — descriptors are cluster averages; medoid fields identify the representative molecule"}
      </h3>
      <dl>
        {Object.entries(candidate.fields).map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
      {prepared && (
        <section>
          <h3>Prepared inputs</h3>
          {Object.entries(prepared.artifacts)
            .filter(
              ([n]) =>
                n.endsWith(".in") && (candidate.id !== "tfa" || n === "tfa.in"),
            )
            .map(([name, url]) => (
              <div key={name}>
                <a href={url} download>
                  {name}
                </a>{" "}
                <button
                  onClick={async () => {
                    try {
                      const r = await fetch(url);
                      if (!r.ok) throw Error("Input unavailable");
                      setInput(await r.text());
                    } catch (e) {
                      setError(String(e));
                    }
                  }}
                >
                  View input
                </button>
              </div>
            ))}
          {error && <p className="error">{error}</p>}
          {input && <pre>{input}</pre>}
        </section>
      )}
      {live &&
        (candidate.id === "tfa" ? ["tfa"] : ["candidate", "complex"]).map(
          (system) => (
            <RunControls
              key={candidate.id + system}
              candidate={candidate.id}
              system={system}
              action={action}
            />
          ),
        )}
      <TaskHistory tasks={tasks} live={live} action={action} />
    </>
  );
}
