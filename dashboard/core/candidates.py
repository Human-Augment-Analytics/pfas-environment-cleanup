import csv

from dashboard.core import config as configuration
from dashboard.core import models


def load() -> list[models.CandidateRecord]:
    with (configuration.ROOT / "shivani_ml_models/cluster_centers.csv").open(
        newline=""
    ) as stream:
        return [
            {
                "id": row["cluster"],
                "fields": row,
                "cid": row["medoid_CID"],
                "smiles": row["medoid_SMILES"],
            }
            for row in csv.DictReader(stream)
        ]


TFA: models.CandidateRecord = {
    "id": "tfa",
    "cid": "8442",
    "smiles": "O=C(O)C(F)(F)F",
    "fields": {"name": "Neutral trifluoroacetic acid", "formula": "C2HF3O2"},
}
