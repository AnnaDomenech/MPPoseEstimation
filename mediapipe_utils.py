import math

import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import landmarks2list

POSE = solutions.pose.PoseLandmark


class PoseEstimator:
    def __init__(self, model_asset: str, segmentation_mask: bool = False) -> None:

        base_options = python.BaseOptions(model_asset_path=model_asset)

        options = vision.PoseLandmarkerOptions(
            base_options=base_options, output_segmentation_masks=segmentation_mask
        )

        self.detector = vision.PoseLandmarker.create_from_options(options)

    def pose_detection(self, image_path: str) -> vision.PoseLandmarker:
        """
        Run Inference on Mediapipe Model.
        """
        image = mp.Image.create_from_file(image_path)
        detection_result = self.detector.detect(image)
        return detection_result

    def draw_landmarks_on_image(
        self, detection_result, rgb_image: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Draw Pose Landmarks(joints) and Connections(bones) on image.
        Parameters:
            detection_result: Output of the model
            rgb_image: image to overlay.
        Returns:
            np.ndarray: Annotated Image.

        """
        # Get pose landmarks
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # Get desired landmark format else default
        landmark_format = kwargs.get("landmark_format", {})
        # Subset of landmarks to plot
        landmark_set = (
            landmark_format["landmarks"] if "landmarks" in landmark_format else False
        )
        # Format to plot landmarks
        landmark_style = (
            landmark_format["style"]
            if "style" in landmark_format
            else solutions.drawing_styles.get_default_pose_landmarks_style()
        )

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            # Get idx-th detected_pose
            pose_landmarks = pose_landmarks_list[idx]

            # Fill NormalizedLandmarkList with landmark information.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            if landmark_set:
                pose_landmarks_proto.landmark.extend(
                    [
                        landmark_pb2.NormalizedLandmark(
                            x=landmark.x,
                            y=landmark.y,
                            z=landmark.z,
                            visibility=(
                                landmark.visibility if i in landmark_set else 0.0
                            ),
                        )
                        for i, landmark in enumerate(pose_landmarks)
                    ]
                )

            else:
                pose_landmarks_proto.landmark.extend(
                    [
                        landmark_pb2.NormalizedLandmark(
                            x=landmark.x, y=landmark.y, z=landmark.z
                        )
                        for landmark in pose_landmarks
                    ]
                )

            # Draw
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                landmark_style,
            )

        return annotated_image


def compute_angle(pose_sample, **kwargs) -> float:
    """
    Compute angle from 3 landmarks.
    """

    landmarks = pose_sample.pose_landmarks[0]
    idx = kwargs.get("landmarks", [])
    if len(idx):
        # Generate points from data and compute the angle
        points, _, _ = landmarks2list(landmarks, idx)
        return getAngle(points[0], points[1], points[2])
    else:
        print("Cannot compute angle. No specific landmarks.")
        return -1


def getAngle(a: list, b: list, c: list, side: str = "right") -> float:
    """
    Compute angle between segment A-B and B-C.
    """
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    # Normalize the angle to the range [-180, 180]
    if angle > 180:
        angle -= 360

    elif angle < -180:
        angle += 360

    # Adjust angle sign based on the side of the body
    if side == "right":
        angle = -angle

    return angle


def compute_distance(landmarks, idx: list) -> float:
    """
    Compute 2D Euclidean distance between two landmarks.
    """
    p, _, _ = landmarks2list(landmarks)
    points = np.array([p[i] for i in idx])
    return np.linalg.norm(points[0] - points[1])


def check_visibility(pose_sample, **kwargs):
    """
    Checks if the person is in front of the camera
    by computing the Euclidean Distance between
    hips and shoulder landmarks.
    Parameters:
        pose_sample: Pose to analyze.
    Returns:
        bool: True if the pose is visible, False otherwises.
    """
    landmarks = pose_sample.pose_landmarks[0]
    thr = kwargs.get("thr", 0.1)

    hip_idx = [POSE.RIGHT_HIP, POSE.LEFT_HIP]
    shoulder_idx = [POSE.RIGHT_SHOULDER, POSE.LEFT_SHOULDER]

    hip_dist = compute_distance(landmarks, hip_idx)
    shoulder_dist = compute_distance(landmarks, shoulder_idx)

    return False if hip_dist < thr or shoulder_dist < thr else True
