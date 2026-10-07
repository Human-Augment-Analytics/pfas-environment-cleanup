import { Candidate } from "./data";

export const filterFields: Record<string, string> = {
  cluster: "Cluster ID",
  n_points: "Points",
  MolecularWeight: "Avg MW",
  XLogP: "Avg XLogP",
  ExactMass: "Avg exact mass",
  Charge: "Avg charge",
  TPSA: "Avg TPSA",
  HBondDonorCount: "Avg H-bond donors",
  HBondAcceptorCount: "Avg H-bond acceptors",
  RotatableBondCount: "Avg rotatable bonds",
  medoid_CID: "Representative CID",
};
export type Filter = { field: string; operator: string; value: string };
export function matchesFilters(candidate: Candidate, filters: Filter[]) {
  return filters.every(({ field, operator, value }) => {
    const raw = candidate.fields[field];
    const actual = Number(raw),
      expected = Number(value);
    if (
      !raw?.trim() ||
      !value.trim() ||
      !Number.isFinite(actual) ||
      !Number.isFinite(expected)
    )
      return false;
    switch (operator) {
      case "<":
        return actual < expected;
      case "<=":
        return actual <= expected;
      case ">":
        return actual > expected;
      case ">=":
        return actual >= expected;
      case "=":
        return actual === expected;
      case "!=":
        return actual !== expected;
      default:
        return false;
    }
  });
}
export function describeFilters(filters: Filter[]) {
  return filters
    .map((f) => `${filterFields[f.field]} ${f.operator} ${f.value}`)
    .join(" AND ");
}
