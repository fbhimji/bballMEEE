from ultralytics import YOLO
import supervision as sv
import numpy as np

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)         # Load the YOLO model from the specified path
        self.model.to("mps")                  # gpu
        self.cls_names_inv = {v: k for k, v in self.model.names.items()}  # Create a map from class ID to class name
        self.ball_history = []                # Initialize list to store ball positions over time

    def get_object_track(self, frame):
        """
        Process a single frame and return ball detection.
        """
        results = self.model.predict(frame, conf=0.3, verbose=False)  # Run detection on the frame
        result = results[0]                                           # Extract first result (single image)

        detection_supervision = sv.Detections.from_ultralytics(result)  # Convert to Supervision Detections format so its itterable
        chosen_bbox = None            # Initialize best bounding box
        max_confidence = 0            # Track max confidence to select the best detection

        for det in detection_supervision:
            bbox = det[0].tolist()    # [x1, y1, x2, y2]
            cls_id = det[3]           # Extract class ID
            confidence = det[2]       # Extract confidence score

            if cls_id == self.cls_names_inv.get("Ball", -1):  # Check if detected object is the ball
                if confidence > max_confidence:               # Keep the most confident ball detection
                    chosen_bbox = bbox
                    max_confidence = confidence

        return chosen_bbox  # Return the best bounding box, or None if no ball found
    #idea - if best bounding box is found and has low cofnidence then to eliminate wrong detection we look at history with ocnfidence and also check distance

    def update_history(self, frame_number, bbox): # used in main.py maybe switch over to in function use
        """
        Store ball position with simple motion filtering.
        """
        if bbox is None:
            self.ball_history.append({})  # If no ball detected, append an empty record
            return

        last_good_idx = self._get_last_valid_index()  # Find most recent valid detection
        if last_good_idx is None:
            self.ball_history.append({1: {"bbox": bbox}})  # First detection — store as new record
            return

        last_bbox = self.ball_history[last_good_idx][1]["bbox"]  # Get last valid bounding box
        dx = np.linalg.norm(np.array(last_bbox[:2]) - np.array(bbox[:2]))  # Distance between current and last
        max_distance = 25 * (frame_number - last_good_idx)  # Allowable movement range over time

        if dx > max_distance:
            self.ball_history.append({})  # If jump is too big, discard it
        else:
            self.ball_history.append({1: {"bbox": bbox}})  # Otherwise, accept and record it

    def _get_last_valid_index(self):
        for i in range(len(self.ball_history) - 1, -1, -1):  # Iterate backward through history
            if 1 in self.ball_history[i]:                    # Look for last frame with a valid ball detection
                return i
        return None  # No valid detections found
