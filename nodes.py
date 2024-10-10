import contextlib

import torch
import torchvision.ops
import PIL
import numpy as np

import folder_paths

import groundingdino.util.inference
import groundingdino.datasets.transforms
import sam2.sam2_image_predictor

@contextlib.contextmanager
def switch_device(model, device, offload_to_cpu=True):
	try:
		model.to(device)
		yield
	finally:
		if offload_to_cpu:
			model.to('cpu')

class Segment:
	def __init__(self):
		device = 'cpu'

		self.groundingdino_model = groundingdino.util.inference.load_model(
			f"/{folder_paths.models_dir}/GroundingDINO/GroundingDINO_SwinB_cfg.py",
			f"/{folder_paths.models_dir}/GroundingDINO/groundingdino_swinb_cogcoor.pth",
			device=device,
		)
		self.sam2_predictor = sam2.sam2_image_predictor.SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large", device=device)

		self.transform = groundingdino.datasets.transforms.Compose(
			[
				groundingdino.datasets.transforms.RandomResize([800], max_size=1333),
				groundingdino.datasets.transforms.ToTensor(),
				groundingdino.datasets.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
			]
		)

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"images": ("IMAGE", {}),
				"label": ("STRING", {}),
			}
		}

	RETURN_TYPES = ("MASK", )
	CATEGORY = "mask"
	FUNCTION = "segment"

	def segment(self, images, label):
		image = images[0]

		with switch_device(self.groundingdino_model, 'cuda'):
			boxes, confidences, labels = groundingdino.util.inference.predict(
				self.groundingdino_model,
				self.transform(
					PIL.Image.fromarray(
						np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8)
					).convert('RGB'),
					None
				)[0],
				label,
				0.3,
				0.25,
			)

		h, w = image.shape[:2]

		with switch_device(self.sam2_predictor.model, 'cuda'):
			self.sam2_predictor.set_image(np.array(image)[..., :3])
			masks, scores, logits = self.sam2_predictor.predict(
				box=torchvision.ops.box_convert(
					boxes=boxes[max(enumerate(confidences), key=lambda x: x[1])[0]] * torch.Tensor([w, h, w, h]),
					in_fmt="cxcywh",
					out_fmt="xyxy",
				).numpy(),
			)

		return (
			torch.from_numpy(masks[max(enumerate(scores), key=lambda x: x[1])[0]]),
		)
