from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from made_reproduction.cli import export_best_checkpoint


class CheckpointExportTests(unittest.TestCase):
    def test_export_best_checkpoint_writes_stable_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "epoch-0012.ckpt"
            source.write_bytes(b"ckpt")
            destination = Path(directory) / "drive"
            named = export_best_checkpoint(source, destination, score=87.4)
            self.assertEqual(named.name, "epoch-0012.ckpt")
            self.assertEqual((destination / "best.ckpt").read_bytes(), b"ckpt")
            payload = json.loads((destination / "best_checkpoint.json").read_text())
            self.assertEqual(payload["checkpoint"], "epoch-0012.ckpt")
            self.assertEqual(payload["best_model_score"], 87.4)
