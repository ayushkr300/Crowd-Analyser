"""
crowd_analyzer.py
-----------------
Crowd density analysis using the P2PNet model.
Provides a singleton-safe CrowdAnalyzer class with prediction,
density classification, heatmap generation, and image annotation.
"""

import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as standard_transforms
from PIL import Image

from models import build_model

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Risk classification constants
# ---------------------------------------------------------------------------
RISK_THRESHOLDS = [
    (2.0, "LOW"),
    (4.0, "MODERATE"),
    (6.0, "HIGH"),
]
RISK_CRITICAL = "CRITICAL"

RISK_COLORS = {
    "LOW": "success",
    "MODERATE": "info",
    "HIGH": "warning",
    "CRITICAL": "error",
}

RISK_EMOJIS = {
    "LOW": "safe",
    "MODERATE": "moderate",
    "HIGH": "high",
    "CRITICAL": "critical",
}

# Image normalisation parameters (ImageNet)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# P2PNet requires image dimensions to be multiples of this value
_DIM_MULTIPLE = 128


class CrowdAnalyzer:
    """
    Wraps a P2PNet model to provide crowd counting and density analysis.

    Usage
    -----
    analyzer = CrowdAnalyzer()
    analyzer.load_model()
    results = analyzer.predict(image_path, area_m2)
    """

    def __init__(
        self,
        weight_path: str = "weights/SHTechA.pth",
        backbone: str = "vgg16_bn",
        row: int = 2,
        line: int = 2,
        gpu_id: int = 0,
        threshold: float = 0.5,
    ) -> None:
        self.weight_path = weight_path
        self.backbone    = backbone
        self.row         = row
        self.line        = line
        self.gpu_id      = gpu_id
        self.threshold   = threshold

        # Resolved at load_model() time
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform: Optional[standard_transforms.Compose] = None

        # Minimal namespace expected by build_model()
        class _Args:
            def __init__(self, backbone, row, line):
                self.backbone = backbone
                self.row      = row
                self.line     = line

        self._args = _Args(backbone, row, line)

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Build the P2PNet model and load pre-trained weights.
        Call once at application startup; the instance is then reusable.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        self.model = build_model(self._args)
        self.model.to(self.device)

        weight_path = Path(self.weight_path)
        if weight_path.exists():
            checkpoint = torch.load(str(weight_path), map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            print(f"[CrowdAnalyzer] Weights loaded from '{weight_path}'.")
        else:
            print(
                f"[CrowdAnalyzer] WARNING: Weights not found at '{weight_path}'. "
                "Running with randomly initialised parameters."
            )

        self.model.eval()

        self.transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image_path: str, area: float) -> Dict:
        """
        Run crowd detection on a single image.

        Parameters
        ----------
        image_path : str
            Path to the input image (JPEG or PNG).
        area : float
            Approximate real-world area visible in the image, in m².

        Returns
        -------
        dict with keys:
            count            : int   – number of people detected
            points           : list  – [[x, y], ...] head coordinates
            density          : float – persons per m²
            risk             : str   – LOW | MODERATE | HIGH | CRITICAL
            inference_time   : float – seconds taken for inference
            annotated_image  : ndarray (BGR) – image with head dots
            original_image   : PIL.Image – resized original
            image_resolution : tuple – (width, height) after resizing
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")
        if area <= 0:
            raise ValueError(f"Area must be positive; received {area}.")

        start = time.perf_counter()

        # --- Load and pre-process ------------------------------------------
        img_raw = Image.open(image_path).convert("RGB")
        img_raw = self._resize_to_multiple(img_raw, _DIM_MULTIPLE)
        new_width, new_height = img_raw.size

        tensor = self.transform(img_raw).unsqueeze(0).to(self.device)

        # --- Inference --------------------------------------------------------
        with torch.no_grad():
            outputs = self.model(tensor)

        scores = torch.nn.functional.softmax(outputs["pred_logits"], dim=-1)[:, :, 1][0]
        pred_points = outputs["pred_points"][0]

        mask   = scores > self.threshold
        points = pred_points[mask].detach().cpu().numpy().tolist()
        count  = int(mask.sum())

        # --- Metrics ----------------------------------------------------------
        density = count / area
        risk    = self.classify_risk(density)

        # --- Visualisation ----------------------------------------------------
        annotated = self._generate_annotated_image(img_raw, points)

        return {
            "count":            count,
            "points":           points,
            "density":          density,
            "risk":             risk,
            "inference_time":   time.perf_counter() - start,
            "annotated_image":  annotated,
            "original_image":   img_raw,
            "image_resolution": (new_width, new_height),
        }

    def generate_heatmap(
        self,
        points: List[List[float]],
        image_shape: Tuple[int, int],
        sigma: int = 15,
    ) -> np.ndarray:
        """
        Produce a JET-colourised density heatmap from detected head points.

        Parameters
        ----------
        points      : [[x, y], ...] from predict()
        image_shape : (height, width)
        sigma       : Gaussian spread in pixels

        Returns
        -------
        np.ndarray (BGR, uint8)
        """
        height, width = image_shape[:2]
        heatmap = np.zeros((height, width), dtype=np.float32)

        for x, y in ((int(p[0]), int(p[1])) for p in points):
            if not (0 <= x < width and 0 <= y < height):
                continue
            y_grid, x_grid = np.ogrid[-y : height - y, -x : width - x]
            dist_sq = x_grid ** 2 + y_grid ** 2
            within  = dist_sq <= (3 * sigma) ** 2
            heatmap[within] += np.exp(-dist_sq[within] / (2 * sigma ** 2))

        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=sigma, sigmaY=sigma)

        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        else:
            heatmap = heatmap.astype(np.uint8)

        return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def classify_risk(density: float) -> str:
        """Return the risk level string for a given density (persons/m²)."""
        for threshold, label in RISK_THRESHOLDS:
            if density < threshold:
                return label
        return RISK_CRITICAL

    @staticmethod
    def get_risk_color(risk: str) -> str:
        """Map a risk label to a Streamlit colour keyword."""
        return RISK_COLORS.get(risk, "info")

    # ------------------------------------------------------------------
    # Image utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_to_multiple(img: Image.Image, multiple: int) -> Image.Image:
        """Crop the image so both dimensions are multiples of *multiple*."""
        w, h = img.size
        new_w = (w // multiple) * multiple
        new_h = (h // multiple) * multiple
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    @staticmethod
    def _generate_annotated_image(
        img_raw: Image.Image,
        points: List[List[float]],
        point_radius: int = 3,
        point_color: Tuple[int, int, int] = (0, 0, 255),  # BGR red
    ) -> np.ndarray:
        """
        Draw a filled circle at each detected head location.

        Returns a BGR ndarray ready for cv2 or Streamlit display.
        """
        # Ensure the image is converted to CPU numpy array before OpenCV operations
        img_array = np.array(img_raw)
        # Ensure it's a contiguous array on CPU
        if torch.is_tensor(img_array):
            img_array = img_array.cpu().numpy()
        canvas = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        for p in points:
            x, y = int(p[0]), int(p[1])
            cv2.circle(canvas, (x, y), point_radius, point_color, -1)
        return canvas

    def save_image(self, image: np.ndarray, output_path: str) -> None:
        """Write a BGR ndarray to disk, creating parent directories if needed."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, image)


# ---------------------------------------------------------------------------
# Singleton accessor (Streamlit-friendly)
# ---------------------------------------------------------------------------

_analyzer_instance: Optional[CrowdAnalyzer] = None


def get_analyzer(
    weight_path: str = "weights/SHTechA.pth",
    backbone: str = "vgg16_bn",
    row: int = 2,
    line: int = 2,
    gpu_id: int = 0,
    threshold: float = 0.5,
) -> CrowdAnalyzer:
    """
    Return the shared CrowdAnalyzer instance, creating and loading it on first call.
    Using a module-level singleton avoids reloading the model on every Streamlit rerun.
    """
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = CrowdAnalyzer(weight_path, backbone, row, line, gpu_id, threshold)
        _analyzer_instance.load_model()
    return _analyzer_instance
