Code with **readability, maintainability, explicitness, and simplicity as the highest priorities**.

Assume the code will be read and modified by developers unfamiliar with its implementation.

Follow these requirements:

1. **Eliminate magic strings and magic numbers.** Move repeated or meaningful values, environment variable names, configuration keys, file names, status strings, and other domain-specific literals into appropriately named constants, enums, or configuration modules. For example, replace `os.getenv("SLURM_JOB_ID")` with `os.getenv(SLURM_JOB_ID_ENV)` where `SLURM_JOB_ID_ENV` is defined in a constants module.

2. **No module-level executable logic.** Module scope should contain imports, constants, type definitions, classes, and function definitions only. Do not perform I/O, read environment variables, initialize services, start jobs, or execute workflows at import time. Put entry-point execution under `if __name__ == "__main__":` and move its implementation into a `main()` function.

3. **Require comprehensive type hints.** Every function and method must have parameter and return annotations, including `-> None` where appropriate. Prefer precise types over `Any`. Use explicit data models or typed structures when they improve clarity.

4. **Keep functions short and focused.** Break long functions into smaller, well-named functions with clear responsibilities. Separate orchestration, business logic, validation, data transformation, and I/O when practical. Avoid deeply nested conditional logic.

5. **Use descriptive names.** Names should communicate purpose and domain meaning. Avoid obscure abbreviations, generic names, and unnecessary indirection.

6. **Explain logical sections.** In longer functions, add short leading comments describing each logical phase. Comments should explain intent, assumptions, or non-obvious reasoning rather than merely restating the code.

7. **Apply DRY pragmatically.** Extract repeated logic when doing so genuinely improves maintainability. Prefer a small amount of straightforward duplication over complicated abstractions, excessive helper functions, or tightly coupled code. WET is acceptable when it makes the code easier to follow.

8. **Avoid overengineering.** Do not introduce unnecessary classes, inheritance, design patterns, configuration layers, factories, or generic abstractions. Prefer simple functions and explicit control flow.

9. **Make dependencies and side effects explicit.** Prefer passing dependencies as arguments over relying on hidden global state. Keep external operations isolated and testable.

10. **Preserve existing behavior.** Refactor incrementally, update tests as needed, and run available linting, formatting, type checking, and test commands.

When reviewing the codebase, look for violations of these principles and improve them systematically. Do not make changes merely to reduce line count or eliminate all duplication.

**When competing principles conflict, prioritize:**
1. Correctness and preservation of behavior
2. Readability and clarity
3. Maintainability and ease of modification
4. Simplicity and explicitness
5. Testability
6. DRY and reuse
7. Brevity and cleverness

Make the code as straightforward as possible without sacrificing correctness.

## Updating AGENTS.md

- In each project, keep a running `AGENTS_updated.md` file and update it as new coding preferences are learned. Periodically remind the user that it is being updated.

## Module organization and entry points

- Import constants through their owning module (for example, `from dashboard.core import execution` and `execution.SLURM_JOB_ID_ENV`), rather than importing lists of constants into the caller's module namespace. Use namespace imports for frontend constants as well.
- Organize dashboard code under `dashboard/app`, `dashboard/core`, and `dashboard/test`. Subdivide `core` by responsibility, such as chemistry, jobs, runtimes, and Slurm. Keep top-level package files limited to entry points and necessary compatibility shims.
- Put all command-line parsers in `app`. Construct each parser in a `get_parser() -> argparse.ArgumentParser` function. Parse arguments at the app entry point and pass them explicitly as keyword arguments into a `main(*, ...)` function in `core`. Core functions must not parse arguments or read `sys.argv`.
- Keep the dashboard a thin interface to the existing chemistry pipeline. Reuse its preparation and input-generation functions. Prefer direct `apptainer exec ... pw.x` Slurm jobs over compute-side Python workers, manifests, or additional execution frameworks. Retain wrappers only when they provide a concrete capability, and document that reason.


## Function size and explanatory comments

- Aim for functions of no more than about 20 lines. Break longer functions into focused, well-named subfunctions rather than leaving large orchestration blocks intact.
- Add a short leading comment for each logical block of roughly 5–10 lines, explaining its purpose or non-obvious assumptions.
- If a function genuinely cannot be broken up safely or clearly, explain the concrete reason in its docstring. Do not use a docstring exception to avoid a straightforward refactor.
