"""
Utility module providing functions for drawing shapes on video frames.

Includes tools for drawing triangles and ellipses to annotate player positions
or other objects in frame-by-frame processing.
"""

import cv2
import numpy as np

def get_center_of_bbox(bbox):
    """
    Returns the center (x, y) of a bounding box.
    """
    x_center = int((bbox[0] + bbox[2]) / 2)
    y_center = int((bbox[1] + bbox[3]) / 2)
    return x_center, y_center

def get_bbox_width(bbox):
    """
    Returns the width of a bounding box.
    """
    return int(bbox[2] - bbox[0])

def get_foot_position(bbox):
    """
    Returns the bottom center of the bounding box (player's foot position).
    """
    x_center = int((bbox[0] + bbox[2]) / 2)
    y_bottom = int(bbox[3])
    return x_center, y_bottom

def draw_rectangle(frame, bbox, color=(0, 0, 255), label=None):
    """
    Draws a rectangle on the frame using the given bounding box.

    Args:
        frame (np.ndarray): Image frame.
        bbox (list or tuple): Bounding box [x1, y1, x2, y2].
        color (tuple): BGR color for the rectangle.
        label (str, optional): Optional label text.

    Returns:
        np.ndarray: Annotated frame.
    """
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if label:
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame

def draw_triangle(frame, bbox, color):
    """
    Draws a filled triangle above the bounding box (e.g., above player's head).
    """
    y = int(bbox[1])
    x, _ = get_center_of_bbox(bbox)

    triangle_points = np.array([
        [x, y],
        [x - 10, y - 20],
        [x + 10, y - 20],
    ])
    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

    return frame

def draw_ellipse(frame, bbox, color, track_id=None):
    """
    Draws an ellipse below a bounding box and optionally displays a track ID.
    """
    y2 = int(bbox[3])
    x_center, _ = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)

    # Draw elliptical base under feet
    cv2.ellipse(
        frame,
        center=(x_center, y2),
        axes=(int(width), int(0.35 * width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4
    )

    # Optional rectangle with ID label
    if track_id is not None:
        rect_w, rect_h = 40, 20
        x1 = x_center - rect_w // 2
        x2 = x_center + rect_w // 2
        y1 = y2 - rect_h // 2 + 15
        y2_ = y2 + rect_h // 2 + 15

        cv2.rectangle(frame, (x1, y1), (x2, y2_), color, cv2.FILLED)

        x_text = x1 + 12
        if track_id > 99:
            x_text -= 10

        cv2.putText(
            frame,
            f"{track_id}",
            (x_text, y1 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    return frame
