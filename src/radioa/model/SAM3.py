import sys
import types
import torch
import numpy as np
from pathlib import Path
from loguru import logger
import nibabel as nib
import cv2
from torchvision.transforms import v2

# --- 路徑設定 ---
current_file_path = Path(__file__).resolve()
sam3_root = str(current_file_path.parent.parent / "utils" / "sam3")
if sam3_root not in sys.path:
    sys.path.append(sam3_root)

from radioa.model.inferer import Inferer
from radioa.prompts.prompt import PromptStep
from radioa.datasets_preprocessing.conversion_utils import load_any_to_nib
from radioa.utils.transforms import (
    ResizeLongestSide,
    orig_to_SAR_dense,
    orig_to_canonical_sparse_coords,
)

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    logger.info("成功從 sys.path 載入 SAM 3 官方模組")
except ImportError:
    from radioa.utils.sam3.sam3.model_builder import build_sam3_image_model
    from radioa.utils.sam3.sam3.model.sam3_image_processor import Sam3Processor


class SAM3Inferer(Inferer):
    supported_prompts = ("box", "point", "mask", "text")
    dim = 2
    transform_reverses_order = True
    _POINT_PROMPT_SUPPORTED = False  # SAM3 PCS 模式不支援 point prompt

    def __init__(self, checkpoint_path, device):
        super().__init__(checkpoint_path, device)
        self.device = device

        model_dtype = next(self.model.parameters()).dtype
        self._model_dtype = model_dtype
        self.processor = Sam3Processor(self.model, device=str(device))

        # --- Monkey Patch set_image ---
        # 讓 transform 用 float32 處理 Normalize，避免 bfloat16 的效能或相容性問題
        def patched_set_image(proc_self, image, state=None):
            import PIL
            if state is None:
                state = {}

            if isinstance(image, PIL.Image.Image):
                width, height = image.size
            elif isinstance(image, (torch.Tensor, np.ndarray)):
                height, width = image.shape[-2:]
            else:
                raise ValueError("Image must be a PIL image or a tensor")

            _transform = v2.Compose([
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(proc_self.resolution, proc_self.resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

            image = v2.functional.to_image(image).to(proc_self.device)
            image = _transform(image).unsqueeze(0)  # float32

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                state["backbone_out"] = proc_self.model.backbone.forward_image(image)
                inst_interactivity_en = proc_self.model.inst_interactive_predictor is not None
                if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
                    sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
                    sam2_backbone_out["backbone_fpn"][0] = (
                        proc_self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                            sam2_backbone_out["backbone_fpn"][0]
                        )
                    )
                    sam2_backbone_out["backbone_fpn"][1] = (
                        proc_self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                            sam2_backbone_out["backbone_fpn"][1]
                        )
                    )

            state["original_height"] = height
            state["original_width"] = width
            return state

        self.processor.set_image = types.MethodType(patched_set_image, self.processor)
        logger.info(f"模型 dtype={model_dtype}，已 patch set_image 確保精度與對齊")

        self.img_size = 1008
        self.transform = ResizeLongestSide(self.img_size)
        self.mask_threshold = 0.5
        # self.mask_threshold = 0.0
        self.image_embeddings_dict = {}
        self._point_warning_shown = False

    def load_model(self, checkpoint_path, device):
        logger.info(f"Loading SAM 3 from {checkpoint_path}")
        correct_bpe_path = (
            "/data/gas/tsao_data/radioactive/src/radioa/utils/sam3/sam3/assets/"
            "bpe_simple_vocab_16e6.txt.gz"
        )
        model = build_sam3_image_model(
            checkpoint_path=str(checkpoint_path),
            bpe_path=correct_bpe_path,
        )
        model.to(device)
        model.eval()
        return model

    # --- 影像處理相關 ---
    def preprocess_img(self, img: np.ndarray, slices_to_process: list) -> dict:
        slices_processed = {}
        for slice_idx in slices_to_process:
            slice_data = img[slice_idx, ...]
            slice_uint8 = self._normalize_slice_to_uint8(slice_data)
            slice_rgb = np.repeat(slice_uint8[..., None], 3, axis=-1)  # (H, W, 3)
            slices_processed[slice_idx] = slice_rgb
        return slices_processed

    def set_image(self, img_path: Path | str):
        img_path = Path(img_path)
        if self._image_already_loaded(img_path=img_path):
            return

        self.image_embeddings_dict = {}
        img_nib = load_any_to_nib(img_path)
        self.orig_affine = img_nib.affine
        self.orig_shape = img_nib.shape

        self.img, self.inv_trans_dense = self.transform_to_model_coords_dense(img_nib, is_seg=False)
        self.loaded_image = img_path
        self.new_shape = self.img.shape

        if self.img.ndim != 3:
            raise ValueError(f"SAM3Inferer 只支援 3D volume，收到 shape={self.img.shape}")

        self.D, self.H, self.W = self.img.shape
        self.global_min = np.percentile(self.img, 0.5)
        self.global_max = np.percentile(self.img, 99.5)


        self.img = np.clip(self.img, self.global_min, self.global_max)

        logger.info(f"影像載入完成。Model-space shape (D, H, W): {self.img.shape}")

    # --- Prompt 處理 ---
    def preprocess_prompt(self, prompt: PromptStep) -> dict:
        slices_raw = prompt.get_slices_to_infer()
        slices = [int(s) for s in slices_raw] if slices_raw is not None and len(slices_raw) > 0 else []
        preprocessed = {s: {"box": None, "points": None, "point_labels": None} for s in slices}

        # Box 處理
        if prompt.has_boxes:
            for slice_idx, box in prompt.boxes.items():
                y_min, x_min, y_max, x_max = box
                cx = float(np.clip((x_min + x_max) / 2.0 / self.W, 0.0, 1.0))
                cy = float(np.clip((y_min + y_max) / 2.0 / self.H, 0.0, 1.0))
                w = float(np.clip((x_max - x_min) / self.W, 0.0, 1.0))
                h = float(np.clip((y_max - y_min) / self.H, 0.0, 1.0))
                preprocessed[int(slice_idx)]["box"] = [cx, cy, w, h]
                #preprocessed[int(slice_idx)]["box"] = [float(x_min), float(y_min), float(x_max), float(y_max)] 

        # Point 處理 (SAM3 PCS 限制提醒)
        if prompt.has_points and prompt.coords is not None:
            if not self._point_warning_shown:
                logger.warning("SAM3 在 PCS 模式下 point prompt 可能無效，Dice 預期較低。")
                self._point_warning_shown = True

            coords = prompt.coords  # (N, 3): (x, y, z)
            labels = prompt.labels  # (N,)
            for i in range(len(coords)):
                x_px, y_px, z_idx = coords[i, 0], coords[i, 1], int(coords[i, 2])
                if z_idx not in preprocessed: continue

                x_norm = float(np.clip(x_px / self.W, 0.0, 1.0))
                y_norm = float(np.clip(y_px / self.H, 0.0, 1.0))
                if preprocessed[z_idx]["points"] is None:
                    preprocessed[z_idx]["points"] = []
                    preprocessed[z_idx]["point_labels"] = []
                preprocessed[z_idx]["points"].append([x_norm, y_norm])
                preprocessed[z_idx]["point_labels"].append(int(labels[i]))

        return preprocessed

    # --- 座標轉換與後處理 ---
    def transform_to_model_coords_dense(self, nifti, **kwargs):
        return orig_to_SAR_dense(nifti)

    def transform_to_model_coords_sparse(self, coords, type=None, **kwargs):
        if coords is None or len(coords) == 0: return coords
        return orig_to_canonical_sparse_coords(coords, self.orig_affine, self.orig_shape)

    def _normalize_slice_to_uint8(self, slice_data: np.ndarray) -> np.ndarray:
        denom = self.global_max - self.global_min + 1e-10
        return np.round((slice_data - self.global_min) / denom * 255.0).astype(np.uint8)

    def _resize_mask_to_hw(self, mask_2d: np.ndarray) -> np.ndarray:
        if mask_2d.shape == (self.H, self.W): return mask_2d.astype(np.uint8)
        if mask_2d.shape == (self.W, self.H): mask_2d = mask_2d.T
        if mask_2d.shape != (self.H, self.W):
            mask_2d = cv2.resize(mask_2d.astype(np.uint8), (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        return mask_2d.astype(np.uint8)

    def _extract_best_mask_from_output(self, output: dict) -> torch.Tensor:
        if "masks" not in output:
            raise ValueError("output 裡沒有 'masks'，請檢查 _forward_grounding")
        masks = output["masks"]
        if masks.shape[0] == 0:
            return torch.zeros(1, 1, self.H, self.W, dtype=torch.bool, device=self.device)
        return masks

    def _combine_masks(self, masks) -> np.ndarray:
        if not isinstance(masks, torch.Tensor): masks = torch.as_tensor(masks)
        if masks.ndim == 4: masks = masks.squeeze(1)
        combined = torch.any(masks, dim=0) if masks.ndim == 3 else masks
        return self._resize_mask_to_hw(combined.to(torch.uint8).cpu().numpy())

    def postprocess_slices(self, slice_mask_dict: dict, return_logits: bool) -> np.ndarray:
        dtype = np.float32 if return_logits else np.uint8
        segmentation = np.zeros((self.D, self.H, self.W), dtype)
        for z, mask in slice_mask_dict.items():
            segmentation[z, :, :] = mask
        return segmentation

    def _add_point_prompt_to_state(self, state: dict, points: list, point_labels: list):
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()
        pts_tensor = torch.tensor(points, device=self.device, dtype=torch.float32).view(len(points), 1, 2)
        lbl_tensor = torch.tensor(point_labels, device=self.device, dtype=torch.long).view(len(point_labels), 1)
        state["geometric_prompt"].append_points(pts_tensor, lbl_tensor)

    # --- 主推論方法 ---
    def predict(
        self,
        prompt: PromptStep,
        return_logits: bool = False,
        prev_seg=None,
        promptstep_in_model_coord_system: bool = False,
    ):
        if not isinstance(prompt, PromptStep):
            raise TypeError("prompt 必須是 PromptStep instance")

        if not promptstep_in_model_coord_system:
            prompt = self.transform_promptstep_to_model_coords(prompt)

        self.D, self.H, self.W = self.img.shape
        slices_to_infer = prompt.get_slices_to_infer()
        slices_to_infer = [int(z) for z in slices_to_infer] if slices_to_infer is not None else []

        if not slices_to_infer:
            segmentation = np.zeros((self.D, self.H, self.W), dtype=np.uint8)
            return self.inv_trans_dense(segmentation), {}, segmentation

        # 預處理
        slices_to_process = [s for s in slices_to_infer if s not in self.image_embeddings_dict]
        slices_processed = self.preprocess_img(self.img, slices_to_process)
        preprocessed_prompt_dict = self.preprocess_prompt(prompt)

        masks_dict, aux_outputs = {}, {}
        use_cuda = "cuda" in str(self.device)

        with torch.no_grad():
            # 確保 autocast 在 loop 外部或正確包覆
            with torch.amp.autocast("cuda" if use_cuda else "cpu", dtype=torch.bfloat16, enabled=use_cuda):
                for slice_idx in slices_to_infer:
                    if slice_idx < 0 or slice_idx >= self.D: continue

                    if slice_idx not in self.image_embeddings_dict:
                        img_np = slices_processed[slice_idx]
                        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(self.device)
                        self.image_embeddings_dict[slice_idx] = self.processor.set_image(img_tensor)

                    state = self.image_embeddings_dict[slice_idx]
                    p = preprocessed_prompt_dict[slice_idx]
                    has_box = p["box"] is not None
                    has_point = p["points"] is not None and len(p["points"]) > 0

                    if not has_box and not has_point: continue

                    self.processor.reset_all_prompts(state)
                    if has_box:
                        output = self.processor.add_geometric_prompt(state=state, box=p["box"], label=True)
                    elif has_point:
                        if "language_features" not in state["backbone_out"]:
                            dummy_text = self.model.backbone.forward_text(["visual"], device=self.device)
                            state["backbone_out"].update(dummy_text)
                        self._add_point_prompt_to_state(state, p["points"], p["point_labels"])
                        output = self.processor._forward_grounding(state)

                    masks = self._extract_best_mask_from_output(output)
                    masks_dict[slice_idx] = self._combine_masks(masks)
                    aux_outputs[slice_idx] = output

        # 組合與返回
        segmentation = self.postprocess_slices(masks_dict, return_logits)
        if prev_seg is not None:
            segmentation = self.merge_seg_with_prev_seg(segmentation, prev_seg, slices_to_infer)

        return self.inv_trans_dense(segmentation), aux_outputs, segmentation