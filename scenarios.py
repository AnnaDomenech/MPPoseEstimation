import cv2

from mediapipe_utils import check_visibility, compute_angle
from utils import add_rectangle_and_text


def framework(model, data_list, **kwargs):
    show = kwargs.get("show", False)
    config = kwargs.get("config", False)
    scenario = kwargs.get("scenario", False)

    for sample in data_list:
        # Infere Pose
        pose_sample = model.pose_detection(sample)

        # Read Image
        image_sample = cv2.imread(sample, cv2.COLOR_BGR2RGB)

        # Draw keypoints and connections of interest
        annotated_image = model.draw_landmarks_on_image(
            pose_sample, image_sample, **config["style"]
        )

        if scenario:
            # Check if there is enough visibility to do any computation
            visible = True
            if scenario == "C":
                visible = check_visibility(pose_sample)

            if visible:
                # Compute shoulder abduction angle
                shoulder_angle = compute_angle(pose_sample, **config["shoulder"])
                # Draw shoulder abduction angle
                text = "Shoulder abduction angle: {:.2f} degrees.".format(
                    shoulder_angle
                )
                add_rectangle_and_text(
                    annotated_image, text, **config["style"]["plain_style"]
                )

                if scenario == "B":
                    # Compute elbow angle
                    elbow_angle = compute_angle(pose_sample, **config["elbow"])
                    # If its value is over 10 degrees, display a warning
                    if elbow_angle > 10:
                        text = "Your arm should be straight."
                        add_rectangle_and_text(
                            annotated_image, text, **config["style"]["warning_style"]
                        )
            else:
                text = "Stand in front of the camera."
                add_rectangle_and_text(
                    annotated_image, text, **config["style"]["warning_style"]
                )

        if show:
            cv2.imshow("Result", annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
