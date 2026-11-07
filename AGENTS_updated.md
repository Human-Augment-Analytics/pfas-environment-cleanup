# Running project coding preferences

Updated 2026-10-23. These preferences supplement the project's AGENTS.md and
were confirmed during the PACE dashboard refactor.

- Use module namespaces for constants, in Python and TypeScript. Avoid long
  `from ... import CONSTANT_A, CONSTANT_B, ...` lists.
- Organize dashboard code under singular `app`, `core`, and `test` directories.
  Group core modules by responsibility: chemistry, jobs, runtimes, and Slurm.
- Make job kinds visible as separate named modules in the file tree. Keep
  non-Python frontend source and tooling outside Python `app` and `core`;
  the proposed frontend location is `dashboard/web`.
- Write architecture walkthroughs from the inside out: explain each module's
  core work before the dependencies, wrappers, and interfaces supporting it.
- Build every command-line parser in `app.get_parser()`; app entry points pass
  parsed options as keyword arguments to keyword-only `core.main(*, ...)`.
  Keep argument parsing and `sys.argv` out of core.
- Keep module imports free of workflow execution and environment-dependent
  initialization. In particular, importing diagram code must not generate files.
- Keep the dashboard a thin wrapper around the existing chemistry pipeline.
  Reuse preparation/input-generation functions. Prefer ordinary Slurm scripts
  launching Apptainer and pw.x directly to a compute-side Python worker or a
  second pipeline framework.
- Preserve behavior incrementally, add comprehensive type hints, and explain
  why remaining wrappers are necessary. Treat compatibility code as temporary
  and state its removal condition.
- Maintain this running preferences file as more preferences are learned and
  periodically remind the user that it is being updated.
- Show the full dashboard task ID in task history so users can copy it and
  locate the corresponding run folder.
- Expose QE geometry optimization progress, total-force history, run statistics,
  and a previewed option to continue from a previous run's atomic positions.

- Keep functions around 20 lines or fewer and explain logical blocks every
  5–10 lines. Split orchestration, I/O, validation, and result handling into
  focused helpers. Document a concrete reason in the docstring only for a
  genuine exception that cannot be sensibly split.
