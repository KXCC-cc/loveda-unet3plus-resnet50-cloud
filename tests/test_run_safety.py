import tempfile
import unittest
from pathlib import Path

from engine.trainer import _prepare_output_directory


class RunDirectorySafetyTests(unittest.TestCase):
    def test_new_run_rejects_non_empty_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            (output_dir / "best_model.pth").touch()

            with self.assertRaises(FileExistsError):
                _prepare_output_directory(output_dir, resume=None)

    def test_resume_checkpoint_must_belong_to_output_directory(self):
        with tempfile.TemporaryDirectory() as first_directory:
            with tempfile.TemporaryDirectory() as second_directory:
                output_dir = Path(first_directory)
                checkpoint = Path(second_directory) / "last_checkpoint.pth"
                checkpoint.touch()

                with self.assertRaises(ValueError):
                    _prepare_output_directory(output_dir, resume=checkpoint)


if __name__ == "__main__":
    unittest.main()
