import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";
import ts from "typescript";

const require = createRequire(import.meta.url);

// Match the existing source-transpilation tests, resolving TSX runtime imports.
function sourceURL(name, replacements = {}) {
  let code = ts.transpileModule(
    readFileSync(new URL(`../src/${name}`, import.meta.url), "utf8"),
    {
      compilerOptions: {
        module: ts.ModuleKind.ES2022,
        target: ts.ScriptTarget.ES2020,
        jsx: ts.JsxEmit.ReactJSX,
      },
    },
  ).outputText;
  for (const [original, replacement] of Object.entries(replacements)) {
    code = code.replace(original, JSON.stringify(replacement));
  }
  return `data:text/javascript;base64,${Buffer.from(code).toString("base64")}`;
}

const { groupAllocations, allocationResources } = await import(
  sourceURL("AllocationView.tsx", {
    '"./execution"': sourceURL("execution.ts"),
    '"react/jsx-runtime"': new URL(
      `file://${require.resolve("react/jsx-runtime")}`,
    ).href,
  })
);
const GIB = 1024 ** 3;

function task(id, overrides = {}) {
  return {
    id,
    target: "slurm",
    kind: "estimate_ram",
    batch_id: "batch",
    candidate: "500",
    system: "candidate",
    status: "queued",
    created: "2026-11-14T00:00:00Z",
    processes: 4,
    ...overrides,
  };
}

test("queued estimates share a batch allocation only when explicitly enabled", () => {
  const tasks = [task("candidate"), task("complex", { system: "complex" })];
  assert.equal(groupAllocations(tasks, true).size, 1);
  assert.equal(groupAllocations(tasks, false).size, 2);
  assert.equal(groupAllocations(tasks).size, 2);
  assert.equal(
    groupAllocations([...tasks, task("other", { batch_id: "other" })], true)
      .size,
    2,
  );
});

test("scheduler tokens override batch inference in both queue modes", () => {
  const tasks = [
    task("a", { scheduler: { token: "shared" } }),
    task("b", { batch_id: "other", scheduler: { token: "shared" } }),
    task("c", { scheduler: { token: "separate" } }),
  ];
  for (const shared of [true, false]) {
    const groups = groupAllocations(tasks, shared);
    assert.equal(groups.size, 2);
    assert.deepEqual(
      groups.get("slurm:shared").map((check) => check.id),
      ["a", "b"],
    );
  }
});

test("nested checks follow batch position rather than reversed task history", () => {
  const candidate = task("candidate", {
    position: 0,
    created: "2026-11-14T00:00:02Z",
  });
  const complex = task("complex", { position: 1, system: "complex" });
  const tasks = [complex, candidate];
  const [checks] = groupAllocations(tasks, true).values();
  assert.deepEqual(
    checks.map((check) => check.id),
    ["candidate", "complex"],
  );
  assert.deepEqual(tasks, [complex, candidate]);
});

test("checks without positions use creation order", () => {
  const tasks = [task("new", { created: "2026-11-14T00:00:02Z" }), task("old")];
  const [checks] = groupAllocations(tasks, true).values();
  assert.deepEqual(
    checks.map((check) => check.id),
    ["old", "new"],
  );
});

test("sequential allocation resources use each maximum, never the sum", () => {
  const checks = [
    task("a", {
      resources: { cpus: 16, memory_bytes: 2 * GIB },
      scheduler_settings: { walltime_hours: 4 },
    }),
    task("b", {
      resources: { cpus: 4, memory_bytes: 8 * GIB },
      scheduler_settings: { walltime_hours: 2 },
    }),
  ];
  assert.deepEqual(allocationResources(checks), {
    cpus: 16,
    memoryBytes: 8 * GIB,
    walltimeHours: 4,
  });
  assert.deepEqual(allocationResources([task("fallback")]), {
    cpus: 4,
    memoryBytes: 0,
    walltimeHours: 0,
  });
});

test("unscheduled terminal tasks and local jobs are excluded; submitted history remains", () => {
  const tasks = [
    task("canceled", { status: "canceled" }),
    task("failed", { status: "failed" }),
    task("local", { target: "local" }),
    task("submitted", {
      status: "succeeded",
      scheduler: { token: "history", job_id: "123" },
    }),
  ];
  const groups = groupAllocations(tasks, true);
  assert.equal(groups.size, 1);
  assert.deepEqual(
    groups.get("slurm:history").map((check) => check.id),
    ["submitted"],
  );
});
