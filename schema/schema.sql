-- PFAS adsorption database: simple PostgreSQL / Neon starter schema.
-- Apply ONCE to a database where these objects do not already exist.
-- No credentials, example data, DROP statements, or repository changes.
BEGIN;

-- One row = one PFAS-only, adsorbent-only, or complex calculation.
CREATE TABLE calculations (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL
        CHECK (kind IN ('pfas', 'adsorbent', 'complex')),
    smiles TEXT,
    structure_path TEXT,
    charge INTEGER NOT NULL,

    -- engine: e.g. QE, PySCF, ORCA.
    -- settings: versions, functional, basis/cutoffs, spin, cell,
    -- pseudopotentials, convergence evidence, original energy/unit, etc.
    engine TEXT NOT NULL,
    settings JSONB NOT NULL DEFAULT '{}'
        CHECK (jsonb_typeof(settings) = 'object'),

    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'converged', 'failed')),
    -- Always eV. Convert source units during ingestion.
    energy_ev DOUBLE PRECISION,
    failure_reason TEXT,
    output_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT calculation_energy_requires_convergence CHECK (
        (status = 'converged' AND energy_ev IS NOT NULL)
        OR (status <> 'converged' AND energy_ev IS NULL)
    ),
    CONSTRAINT calculation_energy_is_finite CHECK (
        energy_ev IS NULL
        OR (energy_ev > '-Infinity'::DOUBLE PRECISION
            AND energy_ev < 'Infinity'::DOUBLE PRECISION)
    )
);

-- One row = one adsorption setup linking three calculation records.
CREATE TABLE adsorption_cases (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    pfas_calculation_id BIGINT NOT NULL REFERENCES calculations(id),
    adsorbent_calculation_id BIGINT NOT NULL REFERENCES calculations(id),
    complex_calculation_id BIGINT NOT NULL REFERENCES calculations(id),
    validation_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (validation_status IN ('pending', 'accepted', 'rejected')),
    validation_notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT case_has_three_distinct_calculations CHECK (
        pfas_calculation_id <> adsorbent_calculation_id
        AND pfas_calculation_id <> complex_calculation_id
        AND adsorbent_calculation_id <> complex_calculation_id
    )
);

CREATE INDEX adsorption_cases_pfas_idx
    ON adsorption_cases (pfas_calculation_id);
CREATE INDEX adsorption_cases_adsorbent_idx
    ON adsorption_cases (adsorbent_calculation_id);
CREATE INDEX adsorption_cases_complex_idx
    ON adsorption_cases (complex_calculation_id);

-- Saved query, not a stored training dataset or a scientific validator.
-- Compute the target from the linked energies instead of storing a duplicate.
CREATE VIEW ml_adsorption_data AS
SELECT
    c.id AS case_id,
    c.name AS case_name,
    p.id AS pfas_calculation_id,
    a.id AS adsorbent_calculation_id,
    x.id AS complex_calculation_id,
    p.name AS pfas_name,
    p.smiles AS pfas_smiles,
    p.charge AS pfas_charge,
    p.engine AS pfas_engine,
    p.settings AS pfas_settings,
    a.name AS adsorbent_name,
    a.smiles AS adsorbent_smiles,
    a.charge AS adsorbent_charge,
    a.structure_path AS adsorbent_structure_path,
    a.engine AS adsorbent_engine,
    a.settings AS adsorbent_settings,
    x.engine AS complex_engine,
    x.settings AS complex_settings,
    x.energy_ev - a.energy_ev - p.energy_ev AS adsorption_energy_ev
FROM adsorption_cases c
JOIN calculations p ON p.id = c.pfas_calculation_id
JOIN calculations a ON a.id = c.adsorbent_calculation_id
JOIN calculations x ON x.id = c.complex_calculation_id
WHERE c.validation_status = 'accepted'
  AND p.status = 'converged'
  AND a.status = 'converged'
  AND x.status = 'converged'
  AND p.kind = 'pfas'
  AND a.kind = 'adsorbent'
  AND x.kind = 'complex';

COMMIT;
