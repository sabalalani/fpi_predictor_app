import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import tempfile
import os
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from skimage import morphology, measure
# ---- Safe compatibility patch for scikit-learn ----
import sklearn.utils
# Patch sklearn.utils._tags._safe_tags if missing
try:
    from sklearn.utils._tags import _safe_tags
except (ImportError, AttributeError):
    # Define a fallback version of _safe_tags
    def _safe_tags(estimator, key=None, fallback=None):
        """Fallback safe_tags for newer sklearn versions."""
        if hasattr(estimator, "_get_tags"):
            tags = estimator._get_tags()
        else:
            tags = {}
        return tags if key is None else tags.get(key, fallback)

    # Recreate sklearn.utils._tags module if needed
    import sys
    if 'sklearn.utils._tags' not in sys.modules:
        sys.modules['sklearn.utils._tags'] = types.ModuleType('sklearn.utils._tags')
    sys.modules['sklearn.utils._tags']._safe_tags = _safe_tags
# Define stubs for deprecated private functions if needed
def _noop(*args, **kwargs):
    return None

# Try to import _get_column_indices (only if really required by your model)
try:
    from sklearn.compose._column_transformer import _get_column_indices
except ImportError:
    def _get_column_indices(*args, **kwargs):
        raise ImportError(
            "_get_column_indices is not available in this scikit-learn version. "
            "Please update your model or preprocessing pipeline."
        )

# _print_elapsed_time was only used internally for timing, safe to stub
try:
    from sklearn.utils._estimator_html_repr import _print_elapsed_time
except ImportError:
    _print_elapsed_time = _noop

# Assign for backward compatibility
sklearn.utils._get_column_indices = _get_column_indices
sklearn.utils._print_elapsed_time = _print_elapsed_time
# ------------------------------------------------------

# ============== CONFIGURATION ==============

MODEL_PATH = "Ridge_Regression_pipeline.joblib"  # Your single model file

CRITERIA_VIEWS = {
    1: {"name": "Talar Head Palpation", "view": "Front View", "description": "Palpation of talar head position"},
    2: {"name": "Supra/Infra Lateral Malleolar Curvature", "view": "Posterior View",
        "description": "Curvature above and below lateral malleolus"},
    3: {"name": "Calcaneus Inversion/Eversion", "view": "Posterior View",
        "description": "Heel bone alignment and tilt"},
    4: {"name": "Bulge in Talonavicular Region", "view": "45-degree Back View",
        "description": "Prominence in talonavicular joint area"},
    5: {"name": "Medial Longitudinal Arch", "view": "Lateral Internal View",
        "description": "Height and shape of medial foot arch"},
    6: {"name": "Forefoot Abduction/Adduction", "view": "Posterior View", "description": "Forefoot alignment and splay"}
}


# ============== INTEGRATE YOUR SEGMENTATION CODE ==============

class OptimalLegPreprocessor:
    """
    OPTIMAL preprocessing - stops at the best stage (Step 4: Background Removed)
    """

    def __init__(self):
        pass

    def preprocess_image(self, img, visualize=False):
        """
        Optimal preprocessing pipeline for uploaded images
        """
        try:
            # Convert PIL to OpenCV if needed
            if isinstance(img, Image.Image):
                img_pil = img.convert("RGB")
                img_pil = ImageOps.exif_transpose(img_pil)
                img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            original = img.copy()
            height, width = img.shape[:2]

            # Step 2: Segmentation
            leg_mask = self.segment_person(img)

            # Step 4: Background removal (FINAL OUTPUT)
            bg_removed = self.remove_background_advanced(original, leg_mask)

            # Step 5: Standardize size to consistent dimensions
            standardized = self.standardize_size(bg_removed, target_height=400)  # Smaller consistent size

            return standardized

        except Exception as e:
            st.error(f"Segmentation error: {e}")
            return None

    def segment_person(self, img):
        """Segment person from background using improved GrabCut"""
        mask_grabcut = self.grabcut_segmentation(img, rect=None)
        combined_mask = self.cleanup_mask(mask_grabcut)
        return combined_mask

    def grabcut_segmentation(self, img, rect=None):
        """GrabCut with large initialization rectangle"""
        mask = np.zeros(img.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        height, width = img.shape[:2]

        if rect is None:
            rect = (
                int(width * 0.05),
                int(height * 0.01),
                int(width * 0.90),
                int(height * 0.98)
            )

        try:
            cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 7, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where(
                (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                255,
                0
            ).astype('uint8')
        except Exception as e:
            st.error(f"GrabCut failed: {e}")
            return np.zeros(img.shape[:2], np.uint8)

        return mask2

    def cleanup_mask(self, mask):
        """Gentle mask cleanup"""
        kernel_small = np.ones((4, 4), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)

        kernel_large = np.ones((12, 12), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)

        # Keep largest component
        labels = measure.label(mask, connectivity=2, background=0)
        if labels.max() == 0:
            return mask

        bincount = np.bincount(labels.ravel())
        if len(bincount) > 1:
            largest_label = np.argmax(bincount[1:]) + 1
        else:
            return mask

        cleaned_mask = np.zeros_like(mask, dtype=np.uint8)
        cleaned_mask[labels == largest_label] = 255

        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(cleaned_mask, contours, -1, 255, -1)
        return cleaned_mask

    def remove_background_advanced(self, img, mask):
        """Clean background removal with gray background"""
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        result = np.where(mask_3channel > 0, img, 0)
        gray_bg = np.full_like(img, 200)  # Light gray background
        result = np.where(result == 0, gray_bg, result)
        return result

    def standardize_size(self, img, target_height=400):
        """Standardize image size while preserving aspect ratio - smaller size for display"""
        height, width = img.shape[:2]

        if height == 0:
            return np.full((target_height, target_height, 3), 200, dtype=np.uint8)

        scale = target_height / height
        new_width = int(width * scale)
        new_height = target_height

        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        if new_width > target_height:
            center = new_width // 2
            start = center - target_height // 2
            end = start + target_height
            final = resized[:, start:end]
        elif new_width < target_height:
            padding = (target_height - new_width) // 2
            final = cv2.copyMakeBorder(resized, 0, 0, padding,
                                       target_height - new_width - padding,
                                       cv2.BORDER_CONSTANT, value=[200, 200, 200])
        else:
            final = resized
        return final


# ============== FEATURE EXTRACTOR ==============

class OptimizedFPIFeatureExtractor:
    def __init__(self):
        self.n_features = 8

    def preprocess_image(self, img):
        if img is None:
            return None, None

        gray_mask = cv2.inRange(img, 0, 190)
        kernel = np.ones((3, 3), np.uint8)
        gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel)
        gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel)

        binary_img = gray_mask
        edges = cv2.Canny(img, 30, 100)
        edges_masked = cv2.bitwise_and(edges, edges, mask=gray_mask)

        return binary_img, edges_masked

    def extract_features(self, processed_img, criteria_num):
        """Extract features for specific criteria from appropriate view"""
        if processed_img is None:
            return np.zeros(self.n_features)

        binary_img, edges = self.preprocess_image(processed_img)
        if binary_img is None:
            return np.zeros(self.n_features)

        h, w = processed_img.shape

        # Route to appropriate feature extractor based on criteria
        if criteria_num == 2:
            return self._get_criteria_2_features(binary_img, edges, h, w)
        elif criteria_num == 3:
            return self._get_criteria_3_features(binary_img, edges, h, w)
        elif criteria_num == 6:
            return self._get_criteria_6_features(binary_img, edges, h, w)
        elif criteria_num == 5:
            return self._get_criteria_5_features(binary_img, edges, h, w)
        else:
            return self._get_fallback_features(processed_img, binary_img, h, w)

    def _ensure_scalar_features(self, features):
        scalar_features = []
        for f in features:
            if isinstance(f, (list, np.ndarray)):
                if len(f) > 0:
                    scalar_features.append(float(f[0]))
                else:
                    scalar_features.append(0.0)
            else:
                scalar_features.append(float(f))

        while len(scalar_features) < self.n_features:
            scalar_features.append(0.0)

        return np.array(scalar_features[:self.n_features], dtype=np.float32)

    def _get_fallback_features(self, processed_img, binary_img, h, w):
        mean_val = float(np.mean(processed_img[binary_img > 0])) if np.any(binary_img > 0) else 0.0
        std_val = float(np.std(processed_img[binary_img > 0])) if np.any(binary_img > 0) else 0.0

        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(main_contour))
            perimeter = float(cv2.arcLength(main_contour, True))
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)

            x, y, w_box, h_box = cv2.boundingRect(main_contour)
            aspect_ratio = float(w_box) / (float(h_box) + 1e-6)

            hull = cv2.convexHull(main_contour)
            hull_area = float(cv2.contourArea(hull))
            solidity = area / (hull_area + 1e-6)

            return [mean_val, std_val, area / (h * w), circularity, aspect_ratio, solidity, perimeter, area]
        else:
            return [mean_val, std_val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def _get_criteria_2_features(self, binary_img, edges, h, w):
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return [0.0] * 8

        main_contour = max(contours, key=cv2.contourArea)
        points = main_contour.reshape(-1, 2)

        y_mid = h // 2
        supra_points = points[points[:, 1] < y_mid]
        infra_points = points[points[:, 1] >= y_mid]

        if supra_points.size == 0 or infra_points.size == 0:
            return [0.0] * 8

        x_center = w // 2
        supra_lateral = supra_points[supra_points[:, 0] > x_center]
        infra_lateral = infra_points[infra_points[:, 0] > x_center]

        if supra_lateral.size == 0 or infra_lateral.size == 0:
            supra_lateral = supra_points[supra_points[:, 0] < x_center]
            infra_lateral = infra_points[infra_points[:, 0] < x_center]
            if supra_lateral.size == 0 or infra_lateral.size == 0:
                return [0.0] * 8
            supra_x = float(np.min(supra_lateral[:, 0]))
            infra_x = float(np.min(infra_lateral[:, 0]))
        else:
            supra_x = float(np.max(supra_lateral[:, 0]))
            infra_x = float(np.max(infra_lateral[:, 0]))

        horizontal_distance = abs(infra_x - supra_x)
        normalized_distance = horizontal_distance / w
        supra_prominence = abs(supra_x - x_center) / w
        infra_prominence = abs(infra_x - x_center) / w

        supra_y = float(np.mean(supra_lateral[:, 1])) if supra_lateral.size > 0 else 0.0
        infra_y = float(np.mean(infra_lateral[:, 1])) if infra_lateral.size > 0 else 0.0
        vertical_distance = abs(infra_y - supra_y)

        return [
            float(horizontal_distance),
            float(normalized_distance),
            float(supra_prominence),
            float(infra_prominence),
            float(supra_x),
            float(infra_x),
            float(vertical_distance),
            float(infra_x - supra_x)
        ]

    def _get_criteria_3_features(self, binary_img, edges, h, w):
        heel_region = binary_img[h // 2:, :]
        midpoints_x = []
        midpoints_y = []
        widths = []

        for y_row in range(heel_region.shape[0]):
            row = heel_region[y_row, :]
            nz_indices = np.where(row > 0)[0]
            if nz_indices.size >= 2:
                x_left = nz_indices[0]
                x_right = nz_indices[-1]
                mid_x = (x_left + x_right) / 2.0
                midpoints_x.append(mid_x)
                midpoints_y.append(y_row + h // 2)
                widths.append(x_right - x_left)

        if len(midpoints_x) < 2:
            return [0.0] * 8

        points = np.array(list(zip(midpoints_x, midpoints_y)), dtype=np.float32)
        [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

        vx_scalar = float(vx[0])
        vy_scalar = float(vy[0])
        angle_rad = np.arctan2(vy_scalar, vx_scalar)
        angle_deg = np.degrees(angle_rad)
        deviation_from_vertical = angle_deg - 90

        midpoint_std = float(np.std(midpoints_x))
        width_std = float(np.std(widths)) if widths else 0.0
        width_mean = float(np.mean(widths)) if widths else 0.0

        top_midpoint = float(midpoints_x[0]) if midpoints_x else 0.0
        bottom_midpoint = float(midpoints_x[-1]) if midpoints_x else 0.0
        lateral_shift = bottom_midpoint - top_midpoint

        return [
            float(deviation_from_vertical),
            float(angle_deg),
            float(midpoint_std),
            float(width_std),
            float(lateral_shift),
            float(width_mean),
            float(len(midpoints_x)),
            float(top_midpoint - w / 2)
        ]

    def _get_criteria_5_features(self, binary_img, edges, h, w):
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return [0.0] * 8

        main_contour = max(contours, key=cv2.contourArea)
        points = main_contour.reshape(-1, 2)

        ground_level = float(np.max(points[:, 1]))
        x_min, x_max = int(np.min(points[:, 0])), int(np.max(points[:, 0]))

        bottom_edge_y = []
        bottom_edge_x = []
        for x in range(x_min, x_max + 1):
            x_points = points[points[:, 0] == x]
            if x_points.size > 0:
                bottom_y = np.max(x_points[:, 1])
                bottom_edge_y.append(bottom_y)
                bottom_edge_x.append(x)

        if not bottom_edge_y:
            return [0.0] * 8

        arch_peak_y = float(np.min(bottom_edge_y))
        arch_height = ground_level - arch_peak_y

        peak_idx = bottom_edge_y.index(min(bottom_edge_y))
        arch_position_x = float(bottom_edge_x[peak_idx])
        arch_position_normalized = (arch_position_x - x_min) / (x_max - x_min + 1e-6)

        arch_sharpness = 0.0
        if peak_idx > 0 and peak_idx < len(bottom_edge_y) - 1:
            left_slope = bottom_edge_y[peak_idx] - bottom_edge_y[max(0, peak_idx - 10)]
            right_slope = bottom_edge_y[peak_idx] - bottom_edge_y[min(len(bottom_edge_y) - 1, peak_idx + 10)]
            arch_sharpness = (left_slope + right_slope) / 2

        arch_width = float(x_max - x_min)
        normalized_arch_height = arch_height / h
        arch_depth_std = float(np.std([ground_level - y for y in bottom_edge_y]))

        return [
            float(arch_height),
            float(normalized_arch_height),
            float(arch_position_normalized),
            float(arch_sharpness),
            float(arch_width),
            float(arch_depth_std),
            float(ground_level),
            float(arch_peak_y)
        ]

    def _get_criteria_6_features(self, binary_img, edges, h, w):
        heel_region = binary_img[h // 2:, :]
        heel_midpoints_x = []

        for y_row in range(heel_region.shape[0]):
            row = heel_region[y_row, :]
            nz_indices = np.where(row > 0)[0]
            if nz_indices.size >= 2:
                x_left = nz_indices[0]
                x_right = nz_indices[-1]
                mid_x = (x_left + x_right) / 2.0
                heel_midpoints_x.append(mid_x)

        if not heel_midpoints_x:
            return [0.0] * 8

        heel_center_x = int(np.mean(heel_midpoints_x))
        forefoot_region = binary_img[:h // 2, :]

        lateral_area = float(np.sum(forefoot_region[:, heel_center_x:] > 0))
        medial_area = float(np.sum(forefoot_region[:, :heel_center_x] > 0))

        area_difference = lateral_area - medial_area
        total_area = lateral_area + medial_area
        lateral_ratio = lateral_area / (total_area + 1e-6)

        forefoot_points = np.argwhere(forefoot_region > 0)
        if forefoot_points.size > 0:
            com_y, com_x = np.mean(forefoot_points, axis=0)
            angle_rad = np.arctan2(com_y, com_x - heel_center_x)
            forefoot_angle = float(np.degrees(angle_rad))
            distance_to_com = float(np.sqrt((com_x - heel_center_x) ** 2 + com_y ** 2))
        else:
            forefoot_angle = 0.0
            distance_to_com = 0.0

        lateral_normalized = lateral_area / (h * w)
        medial_normalized = medial_area / (h * w)
        asymmetry = abs(lateral_ratio - 0.5)

        return [
            float(area_difference),
            float(lateral_ratio),
            float(forefoot_angle),
            float(distance_to_com),
            float(lateral_normalized),
            float(medial_normalized),
            float(asymmetry),
            float(heel_center_x / w)
        ]


# ============== MAIN FPI ANALYZER ==============

class FPIAnalyzer:
    def __init__(self):
        self.feature_extractor = OptimizedFPIFeatureExtractor()
        self.segmenter = OptimalLegPreprocessor()
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the single trained joblib model"""
        try:
            if os.path.exists(MODEL_PATH):
                self.model = joblib.load(MODEL_PATH)
                st.success("✅ Loaded trained FPI model")
            else:
                st.warning(f"⚠ Model not found: {MODEL_PATH}")
                st.info("Using demo mode with heuristic predictions")
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            self.model = None

    def analyze_all_criteria(self, images_dict):
        """Analyze all criteria using images from different views"""
        try:
            features_dict = {}
            segmentation_results = {}

            # Extract features for each criteria from appropriate image
            for criteria, image in images_dict.items():
                if image is not None:
                    # Segment the image first - get only final segmented image
                    segmented_img = self.segmenter.preprocess_image(image, visualize=False)
                    segmentation_results[criteria] = segmented_img

                    if segmented_img is not None:
                        # Convert to grayscale for feature extraction
                        gray_image = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)

                        # Extract features for this criteria
                        features = self.feature_extractor.extract_features(gray_image, criteria)
                        features_dict[criteria] = features
                    else:
                        features_dict[criteria] = np.zeros(8)
                else:
                    features_dict[criteria] = np.zeros(8)
                    segmentation_results[criteria] = None

            # Since your model expects 8 features total, we need to select which features to use
            # Let's use features from the most important criteria (posterior view - criteria 2,3,6)
            posterior_features = []
            for criteria in [2, 3, 6]:  # Use posterior view criteria
                if criteria in features_dict:
                    posterior_features.extend(features_dict[criteria][:3])  # Use first 3 features from each

            # If we don't have enough features, pad with zeros
            while len(posterior_features) < 8:
                posterior_features.append(0.0)

            # Take only first 8 features
            final_features = np.array(posterior_features[:8]).reshape(1, -1)

            # Predict scores using the single model
            if self.model is not None:
                try:
                    prediction = self.model.predict(final_features)[0]

                    # Handle different prediction formats
                    if isinstance(prediction, (int, float, np.number)):
                        # Single prediction - distribute across criteria
                        base_score = int(np.clip(np.round(prediction), -2, 2))
                        scores = {criteria: base_score for criteria in range(1, 7)}
                    else:
                        # Multiple predictions
                        scores = {}
                        for i, criteria in enumerate(range(1, 7)):
                            if i < len(prediction):
                                score = int(np.clip(np.round(prediction[i]), -2, 2))
                                scores[criteria] = score
                            else:
                                scores[criteria] = 0
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    scores = self._demo_predict_scores(features_dict)
            else:
                # Demo mode - use feature-based heuristics
                scores = self._demo_predict_scores(features_dict)

            return scores, features_dict, segmentation_results

        except Exception as e:
            st.error(f"Analysis error: {e}")
            import traceback
            st.error(f"Full error: {traceback.format_exc()}")
            return None, None, None

    def _demo_predict_scores(self, features_dict):
        """Demo prediction fallback"""
        scores = {}
        for criteria in range(1, 7):
            features = features_dict.get(criteria, np.zeros(8))

            if criteria == 2:
                curvature_feature = features[0] if len(features) > 0 else 0
                score = np.clip((curvature_feature - 50) / 25, -2, 2)
            elif criteria == 3:
                angle_feature = features[0] if len(features) > 0 else 0
                score = np.clip(angle_feature / 15, -2, 2)
            elif criteria == 5:
                arch_feature = features[0] if len(features) > 0 else 0
                score = np.clip((arch_feature - 100) / 50, -2, 2)
            elif criteria == 6:
                abduction_feature = features[0] if len(features) > 0 else 0
                score = np.clip(abduction_feature / 1000, -2, 2)
            else:
                if len(features) > 0:
                    score = np.clip(features[0] / 100, -2, 2)
                else:
                    score = 0.0

            scores[criteria] = int(round(score))

        return scores

    def calculate_total_fpi(self, scores):
        return sum(scores.values())

    def classify_foot_type(self, total_score):
        if total_score <= -1:
            return "Supinated", "red"
        elif total_score <= 5:
            return "Neutral", "green"
        else:
            return "Pronated", "orange"


# ============== PASSWORD PROTECTION ==============
#def check_password():
#    if "password_correct" not in st.session_state:
#        st.session_state["password_correct"] = False

#    if not st.session_state["password_correct"]:
#        st.title("🔒 FPI Analyzer - Private Access")
#        password = st.text_input("Enter access password:", type="password")
#        if st.button("Submit"):
#            # FIXED: Only accept the actual secret password, not "default123"
#            if "APP_PASSWORD" in st.secrets and password == st.secrets["APP_PASSWORD"]:
#                st.session_state["password_correct"] = True
#                st.rerun()
#            else:
#                st.error("Incorrect password")
#        st.stop()
#    return True


# ============== STREAMLIT APP ==============

def main():
    st.set_page_config(
        page_title="FPI Multi-View Analyzer",
        page_icon="🦶",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .view-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .criteria-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }
    .score-badge {
        font-size: 1.1rem;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
    }
    .total-score-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .upload-section {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .segmented-image {
        max-width: 100%;
        border-radius: 10px;
        border: 2px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">🦶 FPI Multi-View Analyzer</div>', unsafe_allow_html=True)
    st.markdown("**Upload different foot views for complete FPI assessment**")

    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = FPIAnalyzer()

    if 'uploaded_images' not in st.session_state:
        st.session_state.uploaded_images = {}

    # Sidebar with instructions
    st.sidebar.title("📋 View Requirements")

    for criteria, info in CRITERIA_VIEWS.items():
        st.sidebar.markdown(f"""
        **Criteria {criteria}: {info['name']}**
        - **View:** {info['view']}
        - *{info['description']}*
        """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 🎯 Required Images:
    1. **Front View** → Criteria 1
    2. **Posterior View** → Criteria 2, 3, 6  
    3. **45-degree Back View** → Criteria 4
    4. **Lateral Internal View** → Criteria 5
    """)

    # Main content - Image upload sections
    st.subheader("📸 Upload Required Foot Views")

    # Create upload sections for each required view
    col1, col2 = st.columns(2)

    with col1:
        # Front View - Criteria 1
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("#### 🟦 Front View (Criteria 1)")
        front_file = st.file_uploader(
            "Upload FRONT view image",
            type=['jpg', 'jpeg', 'png'],
            key="front_view",
            help="Front view of the foot for Talar Head Palpation assessment"
        )
        if front_file is not None:
            front_image = Image.open(front_file)
            # Show segmented image instead of original
            segmented_img = st.session_state.analyzer.segmenter.preprocess_image(front_image)
            if segmented_img is not None:
                segmented_rgb = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)
                st.image(segmented_rgb, caption="Segmented Foot View", use_container_width=True)
                st.session_state.uploaded_images[1] = front_image
        st.markdown('</div>', unsafe_allow_html=True)

        # Posterior View - Criteria 2, 3, 6
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("#### 🟩 Posterior View (Criteria 2, 3, 6)")
        posterior_file = st.file_uploader(
            "Upload POSTERIOR view image",
            type=['jpg', 'jpeg', 'png'],
            key="posterior_view",
            help="Back view of foot for Curvature, Heel Alignment, and Forefoot assessment"
        )
        if posterior_file is not None:
            posterior_image = Image.open(posterior_file)
            # Show segmented image instead of original
            segmented_img = st.session_state.analyzer.segmenter.preprocess_image(posterior_image)
            if segmented_img is not None:
                segmented_rgb = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)
                st.image(segmented_rgb, caption="Segmented Foot View", use_container_width=True)
                # Use same posterior image for criteria 2, 3, 6
                st.session_state.uploaded_images[2] = posterior_image
                st.session_state.uploaded_images[3] = posterior_image
                st.session_state.uploaded_images[6] = posterior_image
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # 45-degree Back View - Criteria 4
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("#### 🟨 45-degree Back View (Criteria 4)")
        degree45_file = st.file_uploader(
            "Upload 45-DEGREE BACK view image",
            type=['jpg', 'jpeg', 'png'],
            key="45_degree_view",
            help="45-degree angle from back showing inner foot arch"
        )
        if degree45_file is not None:
            degree45_image = Image.open(degree45_file)
            # Show segmented image instead of original
            segmented_img = st.session_state.analyzer.segmenter.preprocess_image(degree45_image)
            if segmented_img is not None:
                segmented_rgb = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)
                st.image(segmented_rgb, caption="Segmented Foot View", use_container_width=True)
                st.session_state.uploaded_images[4] = degree45_image
        st.markdown('</div>', unsafe_allow_html=True)

        # Lateral Internal View - Criteria 5
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("#### 🟪 Lateral Internal View (Criteria 5)")
        lateral_file = st.file_uploader(
            "Upload LATERAL INTERNAL view image",
            type=['jpg', 'jpeg', 'png'],
            key="lateral_view",
            help="Side profile view showing medial foot arch"
        )
        if lateral_file is not None:
            lateral_image = Image.open(lateral_file)
            # Show segmented image instead of original
            segmented_img = st.session_state.analyzer.segmenter.preprocess_image(lateral_image)
            if segmented_img is not None:
                segmented_rgb = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)
                st.image(segmented_rgb, caption="Segmented Foot View", use_container_width=True)
                st.session_state.uploaded_images[5] = lateral_image
        st.markdown('</div>', unsafe_allow_html=True)

    # Analysis button
    st.markdown("---")
    if st.button("🚀 Analyze Complete FPI", type="primary", use_container_width=True):
        # Check if all required images are uploaded
        required_criteria = [1, 2, 3, 4, 5, 6]
        missing = [criteria for criteria in required_criteria if criteria not in st.session_state.uploaded_images]

        if missing:
            st.error(f"❌ Missing images for criteria: {', '.join(map(str, missing))}")
        else:
            with st.spinner("Analyzing all 6 FPI criteria from multiple views..."):
                scores, features_dict, segmentation_results = st.session_state.analyzer.analyze_all_criteria(
                    st.session_state.uploaded_images)

                if scores is not None:
                    # Store results
                    st.session_state.scores = scores
                    st.session_state.features_dict = features_dict
                    st.session_state.segmentation_results = segmentation_results
                    st.session_state.total_score = st.session_state.analyzer.calculate_total_fpi(scores)
                    st.session_state.foot_type, st.session_state.color = st.session_state.analyzer.classify_foot_type(
                        st.session_state.total_score)

                    st.success("✅ Complete FPI analysis finished!")

    # Results section
    if 'scores' in st.session_state:
        st.markdown("---")
        st.subheader("📊 FPI Analysis Results")

        # Total score
        st.markdown("### 🎯 Total FPI Score")
        st.markdown(f"""
        <div class="total-score-card">
            <h1 style="margin: 0; font-size: 3rem;">{st.session_state.total_score}/12</h1>
            <h2 style="margin: 0; color: {st.session_state.color};">{st.session_state.foot_type} Foot</h2>
        </div>
        """, unsafe_allow_html=True)

        # Individual criteria scores
        st.markdown("### 📈 Individual Criteria Scores")

        # Create two columns for criteria display
        col_results1, col_results2 = st.columns(2)

        with col_results1:
            for criteria in [1, 2, 3]:
                score = st.session_state.scores[criteria]
                info = CRITERIA_VIEWS[criteria]
                score_color = "blue" if score < 0 else "red" if score > 0 else "gray"

                st.markdown(f"""
                <div class="criteria-card">
                    <strong>Criteria {criteria}: {info['name']}</strong><br>
                    <small>View: {info['view']}</small><br>
                    <div class="score-badge" style="background-color: {score_color}; color: white;">
                        Score: {score}
                    </div>
                    <br><small>{info['description']}</small>
                </div>
                """, unsafe_allow_html=True)

        with col_results2:
            for criteria in [4, 5, 6]:
                score = st.session_state.scores[criteria]
                info = CRITERIA_VIEWS[criteria]
                score_color = "blue" if score < 0 else "red" if score > 0 else "gray"

                st.markdown(f"""
                <div class="criteria-card">
                    <strong>Criteria {criteria}: {info['name']}</strong><br>
                    <small>View: {info['view']}</small><br>
                    <div class="score-badge" style="background-color: {score_color}; color: white;">
                        Score: {score}
                    </div>
                    <br><small>{info['description']}</small>
                </div>
                """, unsafe_allow_html=True)

        # Clinical interpretation
        st.markdown("### 📋 Clinical Interpretation")
        if st.session_state.foot_type == "Supinated":
            st.warning("""
            **Supinated Foot Characteristics:**
            - High arch structure
            - Foot rolls outward (underpronation)
            - Rigid foot type
            - Poor shock absorption
            - **Recommendation:** Cushioned, flexible shoes
            """)
        elif st.session_state.foot_type == "Neutral":
            st.success("""
            **Neutral Foot Characteristics:**
            - Normal arch height
            - Efficient shock absorption
            - Balanced pronation
            - Biomechanically efficient
            - **Recommendation:** Stability shoes
            """)
        else:
            st.error("""
            **Pronated Foot Characteristics:**
            - Low arch or flat foot
            - Foot rolls inward excessively (overpronation)
            - Flexible foot type
            - Potential stability issues
            - **Recommendation:** Motion control shoes
            """)


if __name__ == "__main__":
   # if check_password():
    main()
