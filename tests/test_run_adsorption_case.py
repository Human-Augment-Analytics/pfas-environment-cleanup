import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "qespresso_pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

import run_adsorption_case as runner


VALID_ARGUMENTS = [
    "--case-name", "case",
    "--adsorbent-name", "adsorbent",
    "--pfas-name", "pfas",
    "--adsorbent-source", "smiles",
    "--pseudo-dir", "pseudos",
]


class RunAdsorptionCaseTests(unittest.TestCase):
    def test_get_parser_preserves_defaults(self):
        args = runner.get_parser().parse_args(VALID_ARGUMENTS)

        self.assertEqual(args.system_type, "molecule")
        self.assertEqual(args.mode, "cluster")
        self.assertEqual(args.workdir, "dft_cases")
        self.assertFalse(args.prepare_only)

    def test_main_forwards_parsed_values_as_keywords(self):
        expected_args = runner.get_parser().parse_args(VALID_ARGUMENTS)
        parser = unittest.mock.Mock()
        parser.parse_args.return_value = expected_args

        with patch.object(runner, "get_parser", return_value=parser), patch.object(
            runner, "main_from_kwargs"
        ) as main_from_kwargs:
            runner.main()

        main_from_kwargs.assert_called_once_with(**vars(expected_args))

    def test_main_from_kwargs_runs_without_cli_arguments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            kwargs = vars(runner.get_parser().parse_args(VALID_ARGUMENTS))
            kwargs.update(
                workdir=str(temp_path / "cases"),
                compound_root=str(temp_path / "compounds"),
                pseudo_dir=str(temp_path / "pseudos"),
                prepare_only=True,
                skip_ads=True,
                skip_pfas=True,
                skip_complex=True,
            )
            runner.main_from_kwargs(**kwargs)

            self.assertTrue((temp_path / "cases" / "case" / "complex").is_dir())


if __name__ == "__main__":
    unittest.main()
