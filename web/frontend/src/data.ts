export type Candidate = {
  id: string;
  cid: string;
  smiles: string;
  fields: Record<string, string>;
};
export type Geometry = {
  atoms?: { element: string; position: number[] }[];
  units?: string;
  input_hash?: string;
  error?: string;
};
export type Task = {
  geometries?: Record<string, Geometry>;
  id: string;
  kind: string;
  candidate: string;
  system: string;
  processes: number;
  status: string;
  created: string;
  started?: string;
  ended?: string;
  elapsed_seconds?: number;
  exit_code?: number;
  error?: string;
  command?: string[];
  artifacts: Record<string, string>;
  stdout_tail?: string;
  stderr_tail?: string;
  evidence?: {
    energy_ry: number | null;
    confirmed: boolean;
    provisional: boolean;
    job_done: boolean;
    scf_converged: boolean;
    relaxation_completed: boolean;
  };
};
export type Data = {
  mode: "live" | "snapshot";
  timestamp?: string;
  candidates: Candidate[];
  reference: Candidate;
  tasks: Task[];
};
export type Action = {
  kind: string;
  candidate: string;
  system?: string;
  processes?: number;
  target?: string;
  timeout?: number;
  expected_hash?: string;
};
export type Preview = { command: string[]; input: string; input_hash: string };
declare const __DATA_MODE__: Data["mode"];

export async function readLive(): Promise<Data> {
  const response = await fetch("/api/data");
  if (!response.ok) throw new Error("Cannot load dashboard data");
  return response.json();
}

export async function readSnapshot(): Promise<Data> {
  const response = await fetch(new URL("snapshot/data.json", document.baseURI));
  if (!response.ok) throw new Error("Cannot load exported snapshot");
  return response.json();
}

export async function read(): Promise<Data> {
  return __DATA_MODE__ === "snapshot" ? readSnapshot() : readLive();
}
export async function post<T>(path: string, body?: Action): Promise<T> {
  const response = await fetch("/api/" + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || response.statusText);
  }
  return response.json();
}
