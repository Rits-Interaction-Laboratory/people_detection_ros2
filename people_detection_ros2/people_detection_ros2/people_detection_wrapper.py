from typing import Any, List, Tuple

import numpy as np
import torch
from torch.backends import cudnn
from yolact.data import set_cfg, cfg
from yolact.layers.output_utils import postprocess
from yolact.utils.augmentations import FastBaseTransform
from yolact.utils.functions import SavePath
from yolact.yolact import Yolact


class PeopleDetectionWrapper:
    config: str
    net: Any

    def __init__(self, trained_model_path: str, score_threshold: float, is_debug_mode: bool):
        self.trained_model_path = trained_model_path
        self.score_threshold = score_threshold
        self.is_debug_mode = is_debug_mode

        self.load_model()

    def load_model(self):
        model_path = SavePath.from_str(self.trained_model_path)
        self.config = model_path.model_name + "_config"
        print("Config not specified. Parsed %s from the file name.\n" % self.config)
        set_cfg(self.config)

        with torch.no_grad():
            cudnn.benchmark = True
            cudnn.fastest = True
            torch.set_default_tensor_type("torch.cuda.FloatTensor")

            print("Loading model...")
            self.net = Yolact()
            self.net.load_weights(self.trained_model_path)
            self.net.eval()
            self.net = self.net.cuda()
            print("Loading model done.")

            self.net.detect.use_fast_nms = True
            cfg.mask_proto_debug = False

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Note: If 'undo_transform=False' then 'im_h' and 'im_w' are allowed to be 'None'.
        """
        frame = torch.from_numpy(image).cuda().float()
        batch = FastBaseTransform().forward(frame.unsqueeze(0))
        preds = self.net(batch)

        h, w, _ = frame.shape

        t = postprocess(preds, w, h,
                        crop_masks=True,
                        score_threshold=self.score_threshold)
        torch.cuda.synchronize(torch.cuda.current_device())

        masks = None
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy.
            masks = t[3][:]
        classes, scores, boxes = [x[:].cpu().detach().numpy() for x in t[:3]]

        num_dets_to_consider = classes.shape[0]
        for j in range(num_dets_to_consider):
            if scores[j] < self.score_threshold:
                num_dets_to_consider = j
                break

        masked_img = np.zeros(frame.shape[:2], np.uint8)
        result_boxes = []
        if num_dets_to_consider == 0:
            # maskが見つからない
            return masked_img, result_boxes

        if masks is None:
            return masked_img, result_boxes

        for i in range(num_dets_to_consider):
            # 人間以外ははじく
            if cfg.dataset.class_names[classes[i]] != 'person':
                continue
            m = masks.byte().cpu().numpy()
            masked_img = np.where(m[i] > 0, 255, masked_img)
            result_boxes.append(boxes[i])

        return masked_img, result_boxes
