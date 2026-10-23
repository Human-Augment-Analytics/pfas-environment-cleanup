import type { ReactElement } from "react";
import {
  DEFAULT_PARTITION,
  DEFAULT_ACCOUNT,
  DEFAULT_QOS,
  DEFAULT_ESTIMATION_HOURS,
  DEFAULT_QE_HOURS,
  MAX_SLURM_HOURS,
  DEFAULT_HEADROOM_PERCENT,
  MAX_HEADROOM_PERCENT,
} from "./execution";
import { SchedulerSettings } from "./data";

const SCHEDULING_FIELDS = [
  { key: "partition", label: "Partition", defaultValue: DEFAULT_PARTITION },
  { key: "account", label: "Account", defaultValue: DEFAULT_ACCOUNT },
  { key: "qos", label: "QOS", defaultValue: DEFAULT_QOS },
] as const;

export function SlurmSettings({
  value,
  kind,
  onChange,
}: {
  value: SchedulerSettings;
  kind: string;
  onChange: (value: SchedulerSettings) => void;
}): ReactElement {
  return (
    <>
      {SCHEDULING_FIELDS.map((field) => (
        <label key={field.key}>
          {field.label}{" "}
          <input
            value={value[field.key] ?? field.defaultValue}
            onChange={(event) =>
              onChange({ ...value, [field.key]: event.target.value })
            }
          />
        </label>
      ))}
      <label>
        Walltime (hours){" "}
        <input
          type="number"
          min="1"
          max={MAX_SLURM_HOURS}
          value={
            value.walltime_hours ??
            (kind === "estimate_ram"
              ? DEFAULT_ESTIMATION_HOURS
              : DEFAULT_QE_HOURS)
          }
          onChange={(e) =>
            onChange({ ...value, walltime_hours: Number(e.target.value) })
          }
        />
      </label>
      {kind === "qe" && (
        <label>
          Memory sizing{" "}
          <select
            value={value.memory_mode ?? "manual"}
            onChange={(e) =>
              onChange({
                ...value,
                memory_mode: e.target.value as "manual" | "estimate",
              })
            }
          >
            <option value="manual">Entered reservation</option>
            <option value="estimate">RAM estimate + headroom</option>
          </select>
        </label>
      )}
      {kind === "qe" && value.memory_mode === "estimate" && (
        <label>
          Memory headroom (%){" "}
          <input
            type="number"
            min="0"
            max={MAX_HEADROOM_PERCENT}
            value={value.headroom_percent ?? DEFAULT_HEADROOM_PERCENT}
            onChange={(event) =>
              onChange({
                ...value,
                headroom_percent: Number(event.target.value),
              })
            }
          />
        </label>
      )}
    </>
  );
}
