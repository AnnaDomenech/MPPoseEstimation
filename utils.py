import os

import cv2
import numpy as np


def get_data(folder: str, format: str = ".jpeg") -> list:
    """
    Reads all the files in the specified directory and returns a list of format files.

    Parameters:
    folder (str): The directory to read the files from.
    format (str): The type of file we want to extract.

    Returns:
    list: A list of 'format' file names.
    """
    filename_list = []
    if not os.path.exists(folder):
        print(f"Folder does not exist: {folder}")
        return filename_list

    for file in os.listdir(folder):
        if file.lower().endswith(format):
            filename_list.append(folder + "/" + file)

    return filename_list


def top(img: np.ndarray) -> tuple:
    """
    Returns proportional upper part of the image.
    """

    return (0, int(img.shape[0] * 0.05))


def bottom(img: np.ndarray) -> tuple:
    """
    Returns proportional lower part of the image.
    """

    return (0, int(img.shape[0] * 0.95))


def landmarks2list(landmarks, idx: list = []) -> list:
    """
    Convert mediapipe landmarks to list format.
    Parameters:
        landmarks: List of NormalizedLandmarkList() or LandmarkList().

    Returns:
        points, visibility, presence: Tuple of 2D landmark position, visibility and presence.
    """

    points, visibility, presence = [], [], []

    if len(idx):
        for i in idx:
            l = landmarks[i]
            points.append([l.x, l.y])
            visibility.append(l.visibility)
            presence.append(l.presence)
    else:
        for l in landmarks:
            points.append([l.x, l.y])
            visibility.append(l.visibility)
            presence.append(l.presence)

    return points, visibility, presence


def add_rectangle_and_text(img: np.ndarray, text: str, **kwargs) -> np.ndarray:
    """
    Add a rectangle with centered text to img.
    Parameters:
        img: Image to annotate.
        text: Text to add.

    Returns:
        img: Final image.
    """

    # Position
    position_func = kwargs.get("position", top)
    position = position_func(img)
    # Font Format
    text_color = kwargs.get("text_color", (255, 255, 255))
    font, font_scale, font_thickness = kwargs.get(
        "font_format", (cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
    )
    # Rectangle format
    rectangle_size = kwargs.get("rectangle_size", (img.shape[1], 100))
    rectangle_color = kwargs.get("rectangle_color", (0, 0, 0, 0.5))

    # Copy the image
    overlay = img.copy()

    # Define the rectangle coordinates.
    rect_x1, rect_y1 = position
    rect_x2, rect_y2 = rect_x1 + rectangle_size[0], rect_y1 + rectangle_size[1]

    # Draw the semi-transparent rectangle.
    alpha = rectangle_color[3]
    rectangle_color_rgb = (rectangle_color[0], rectangle_color[1], rectangle_color[2])
    cv2.rectangle(
        overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), rectangle_color_rgb, -1
    )
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Calculate text position to be in the center of rectangle.
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = rect_x1 + (rectangle_size[0] - text_size[0]) // 2
    text_y = rect_y1 + (rectangle_size[1] + text_size[1]) // 2

    # Draw the text
    cv2.putText(
        img,
        text,
        (text_x, text_y),
        font,
        font_scale,
        text_color,
        font_thickness,
        cv2.LINE_AA,
    )

    return img
