import unittest

import numpy as np
import torch
import torchvision

from yolox.utils.boxes import postprocess


num_classes = 10
prediction_dim = 8400


def select_high_confidence_prediction(image_pred: torch.Tensor, conf_thre = 0.7) -> np.ndarray:
    assert image_pred.shape == (prediction_dim, 5 + num_classes)
    class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
    conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre)

    # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
    assert detections.shape == (prediction_dim, 7)
    return detections[conf_mask]


def convert_bbox_format(image_pred: torch.Tensor) -> torch.Tensor:
    ret = torch.empty(image_pred.shape[0], 4)
    ret[:, 0] = image_pred[:, 0] - image_pred[:, 2] / 2
    ret[:, 1] = image_pred[:, 1] - image_pred[:, 3] / 2
    ret[:, 2] = image_pred[:, 0] + image_pred[:, 2] / 2
    ret[:, 3] = image_pred[:, 1] + image_pred[:, 3] / 2
    return ret


class TestBoxes(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        return super().setUp()

    def test_postprocess(self) -> None:
        obj_conf_idx = 4
        class_conf_idx = 5
        class_pred_idx = 6
        conf_thre = 0.7
        nms_thre = 0.45

        prediction = torch.rand(1, prediction_dim, 5 + num_classes)

        # step 1: filter out low confidence predictions
        high_confidence_predictions = select_high_confidence_prediction(prediction[0], conf_thre=conf_thre)
        highest_class_conf, highest_class_idx = prediction[0, :, class_conf_idx:].max(dim=1)
        confidence = prediction[0, :, obj_conf_idx] * highest_class_conf
        confidence_mask = confidence >= conf_thre

        num_high_confidence = confidence_mask.sum().item()
        self.assertEqual(high_confidence_predictions.shape, (num_high_confidence, 7))

        torch.testing.assert_close(high_confidence_predictions[:, :obj_conf_idx], prediction[0, confidence_mask, :obj_conf_idx])
        torch.testing.assert_close(high_confidence_predictions[:, obj_conf_idx], prediction[0, confidence_mask, obj_conf_idx])
        torch.testing.assert_close(high_confidence_predictions[:, class_conf_idx], highest_class_conf[confidence_mask])
        torch.testing.assert_close(high_confidence_predictions[:, class_pred_idx], highest_class_idx[confidence_mask].float())

        # step 2: bbox format conversion from (center, size) to (left, top, right, bottom)
        high_confidence_predictions[:, :obj_conf_idx] = convert_bbox_format(high_confidence_predictions[:, :obj_conf_idx])

        # step 3: NMS filter
        overall_confidence = high_confidence_predictions[:, obj_conf_idx] * high_confidence_predictions[:, class_conf_idx]
        nms_filter_idx = torchvision.ops.batched_nms(
            high_confidence_predictions[:, :obj_conf_idx],
            overall_confidence,
            high_confidence_predictions[:, class_pred_idx],
            nms_thre,
        )
        nms_filtered_predictions = high_confidence_predictions[nms_filter_idx]

        processed = postprocess(prediction, num_classes=num_classes)[0]
        torch.testing.assert_close(processed, nms_filtered_predictions)
