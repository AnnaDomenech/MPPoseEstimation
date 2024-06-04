import argparse

import mediapipe as mp
from mediapipe import solutions

from mediapipe_utils import PoseEstimator
from scenarios import framework
from utils import bottom, get_data, top

DEFAULT_DATA_PATH = "./Data/A"
MODEL_PATH = "./models/pose_landmarker_lite.task"
POSE = solutions.pose.PoseLandmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to run the proposed tasks.")
    parser.add_argument(
        "--scenario", type=str, default="A", required=False, help="The scenario to run."
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=DEFAULT_DATA_PATH,
        required=False,
        help="The folder to use.",
    )
    parser.add_argument(
        "--show", type=str, required=False, default=True, help="Show result."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
        required=False,
        help="Pose Model Path.",
    )

    args = parser.parse_args()
    scenario_str = args.scenario
    data_path = args.folder
    show = args.show
    model_path = args.model_path

    # Read files
    data_list = get_data(data_path)
    if not len(data_list):
        print("No images found. Review the directory.")
        quit()

    detector = PoseEstimator(model_path)
    config = {
        "show": show,
        "scenario": scenario_str,
        "config": {
            # Style for Plotting Landmarks
            "style": {
                "landmark_format": {
                    "landmarks": [
                        POSE.RIGHT_ELBOW,
                        POSE.RIGHT_SHOULDER,
                        POSE.RIGHT_HIP,
                        POSE.RIGHT_WRIST,
                    ],
                    "style": {
                        POSE.RIGHT_ELBOW: solutions.drawing_utils.DrawingSpec(
                            color=(0, 255, 0), thickness=-1, circle_radius=12
                        ),
                        POSE.RIGHT_SHOULDER: solutions.drawing_utils.DrawingSpec(
                            color=(100, 100, 100), thickness=-1, circle_radius=12
                        ),
                        POSE.RIGHT_HIP: solutions.drawing_utils.DrawingSpec(
                            color=(0, 255, 0), thickness=-1, circle_radius=12
                        ),
                        POSE.RIGHT_WRIST: solutions.drawing_utils.DrawingSpec(
                            color=(0, 255, 0), thickness=-1, circle_radius=12
                        ),
                    },
                },
                "warning_style": {
                    "position": bottom,
                    "text_color": (0, 0, 255),
                },
                "rectangle_color": (0, 0, 0, 0.5),
                "plain_style": {
                    "position": top,
                    "rectangle_color": (0, 0, 0, 0.5),
                    "text_color": (255, 255, 255),
                },
            },
            "shoulder": {
                "landmarks": [POSE.RIGHT_WRIST, POSE.RIGHT_SHOULDER, POSE.RIGHT_HIP]
            },
            "elbow": {
                "landmarks": [POSE.RIGHT_ELBOW, POSE.RIGHT_WRIST, POSE.RIGHT_SHOULDER],
            },
        },
    }

    framework(detector, data_list, **config)
