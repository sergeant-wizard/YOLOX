import unittest

import cv2
import numpy as np

from yolox.data.data_augment import preproc


class TestDataAugment(unittest.TestCase):
    def test_preproc(self) -> None:
        input_shape = (640, 640)
        padded_color = np.array([114, 114, 114], dtype=np.float32)

        original_img = cv2.imread("assets/dog.jpg")

        # HWC, BGR
        self.assertEquals(original_img.shape, (576, 768, 3))
        self.assertEquals(original_img.min(), 0)
        self.assertEquals(original_img.max(), 255)

        img, ratio = preproc(original_img, input_shape)

        self.assertEquals(img.shape, (3, input_shape[0], input_shape[1]))
        self.assertAlmostEquals(ratio, min(input_shape[0] / original_img.shape[0], input_shape[1] / original_img.shape[1]))
        self.assertAlmostEquals(img.min(), 0.0)
        self.assertAlmostEquals(img.max(), 255.0)
        self.assertAlmostEquals(img.dtype, np.float32)

        # Color Channel is preserved.
        np.testing.assert_allclose(img[:, 0, 0], original_img[0, 0, :])

        # transposed to CHW
        # The right top corner should not be padded.
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_allclose(img[:, 0, -1], padded_color)

        # The bottom corners should be padded.
        np.testing.assert_allclose(img[:, -1, 0], padded_color)
        np.testing.assert_allclose(img[:, -1, -1], padded_color)
