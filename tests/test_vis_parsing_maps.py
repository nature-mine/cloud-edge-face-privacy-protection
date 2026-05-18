import importlib.util
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_MODULE_PATH = PROJECT_ROOT / "test.py"


def load_test_module():
    spec = importlib.util.spec_from_file_location("face_parsing_test_module", TEST_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class VisParsingMapsTest(unittest.TestCase):
    def test_saves_only_core_facial_parts(self):
        module = load_test_module()
        image = Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8))
        parsing_anno = np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 10, 11],
                [12, 13, 16, 17],
                [7, 8, 14, 15],
            ],
            dtype=np.uint8,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            save_path = Path(temporary_directory) / "example.jpg"

            module.vis_parsing_maps(image, parsing_anno, stride=1, save_im=True, save_path=str(save_path))

            saved_mask = cv2.imread(str(save_path.with_suffix(".png")), cv2.IMREAD_UNCHANGED)
            with np.load(str(save_path.with_name("example_dct_blocks.npz"))) as dct_result:
                dct_keys = set(dct_result.files)
                coefficients_shape = dct_result["coefficients"].shape
                block_size = int(dct_result["block_size"])
            blocks_image_exists = save_path.with_name("example_blocks.jpg").exists()
            dct_image_exists = save_path.with_name("example_dct.jpg").exists()

        expected_values = {0, 2, 3, 4, 5, 10, 11, 12, 13}
        self.assertTrue(set(np.unique(saved_mask)).issubset(expected_values))
        self.assertEqual(dct_keys, {"coefficients", "block_mask", "bbox", "padded_shape", "block_size"})
        self.assertEqual(coefficients_shape[-2:], (8, 8))
        self.assertEqual(block_size, 8)
        self.assertTrue(blocks_image_exists)
        self.assertTrue(dct_image_exists)


if __name__ == "__main__":
    unittest.main()
