from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from webspell.demo import build_demo_model
from webspell.pipeline import WebSpellModel
from webspell.serialization.bundle import BundleValidationError


class SerializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = build_demo_model()

    def test_prediction_round_trip_is_identical(self) -> None:
        text = "we learn form clean text"
        before = self.model.predict(text)
        with tempfile.TemporaryDirectory() as directory:
            self.model.save(directory)
            loaded = WebSpellModel.load(directory)
            after = loaded.predict(text)
        self.assertEqual(before, after)
        self.assertEqual(
            self.model.corrected_text(text, before),
            loaded.corrected_text(text, after),
        )

    def test_checksum_detects_model_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.model.save(directory)
            model_path = Path(directory) / "model.json"
            model_path.write_bytes(model_path.read_bytes() + b" ")
            with self.assertRaisesRegex(BundleValidationError, "size|checksum"):
                WebSpellModel.load(directory)

    def test_schema_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.model.save(directory)
            manifest_path = Path(directory) / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = 999
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(BundleValidationError, "schema"):
                WebSpellModel.load(directory)


if __name__ == "__main__":
    unittest.main()
