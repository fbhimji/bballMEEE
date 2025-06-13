from ultralytics import YOLO
import supervision as sv
import numpy as np

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)         # Load the YOLO model from the specified path
        self.model.to("mps")                # gpu
        self.cls_names_inv = {v: k for k, v in self.model.names.items()}  # Create a map from class ID to class name
        self.ball_history = []                # Initialize list to store ball positions over time
        self.last_velocity = None
        self.last_center = None

        self.last_rejection_confidences = []

    def get_object_track(self, frame):
        """
        Process a single frame and return ball detection.
        """
        results = self.model.predict(frame, conf=0.5, verbose=False)  # Run detection on the frame
        result = results[0]                                           # Extract first result (single image)

        detection_supervision = sv.Detections.from_ultralytics(result)  # Convert to Supervision Detections format so its itterable
        chosen_bbox = None            # Initialize best bounding box
        max_confidence = 0            # Track max confidence to select the best detection

        for det in detection_supervision:
            bbox = det[0].tolist()    # [x1, y1, x2, y2]
            cls_id = det[3]           # Extract class ID
            confidence = det[2]       # Extract confidence score

            if cls_id == self.cls_names_inv.get("Ball", -1):  # Check if detected object is the ball
                if confidence > max_confidence:
                    if self._is_reasonable_detection(bbox, confidence):               # Keep the most confident ball detection, and pass in cofidence of detection
                        chosen_bbox = bbox
                        max_confidence = confidence

        if chosen_bbox is None:
            chosen_bbox = self._get_predicted_position()
        

        if chosen_bbox is not None:
            self._update_velocity_tracking(chosen_bbox)

        return chosen_bbox  # Return the best bounding box, or None if no ball found
    #idea - if best bounding box is found and has low cofnidence then to eliminate wrong detection we look at history with ocnfidence and also check distance

    def _get_last_valid_index(self):
        for i in range(len(self.ball_history) - 1, -1, -1):  # Iterate backward through history
            if 1 in self.ball_history[i]:                    # Look for last frame with a valid ball detection
                return i
        return None  # No valid detections found
    
    def _get_predicted_position(self):
        """Get predicted ball position if missing for short time."""
        frames_missing = self._count_missing_frames()
        
        if frames_missing > 2 or self.last_velocity is None:
            return None  # Give up after 2 frames or no velocity data
            
        return self._calculate_predicted_bbox(frames_missing)
    
    def _count_missing_frames(self):
        """Count how many frames since last detection."""
        last_idx = self._get_last_valid_index()
        return len(self.ball_history) - last_idx - 1 if last_idx is not None else 999
    
    def update_history(self, frame_number, bbox):
        """Store ball position - simple version to make prediction work."""
        if bbox is None:
            self.ball_history.append({})  # Empty frame
        else:
            self.ball_history.append({1: {"bbox": bbox}})  # Ball detected
    
    def _update_velocity_tracking(self, bbox):
        """Track velocity from current detection."""
        center = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
        
        if self.last_center is not None:
            self.last_velocity = [center[0] - self.last_center[0], 
                                 center[1] - self.last_center[1]]
        
        self.last_center = center
    
    def _calculate_predicted_bbox(self, frames_missing):
        """
        Calculate predicted bbox position. 
        Change this method to switch between linear/physics prediction.
        """
        if self.last_center is None or self.last_velocity is None:
            return None
        
        # LINEAR PREDICTION (change this method for physics later)
        pred_x = self.last_center[0] + self.last_velocity[0] * frames_missing
        pred_y = self.last_center[1] + self.last_velocity[1] * frames_missing
        
        # Return bbox with standard ball size
        ball_size = 30
        return [pred_x - ball_size/2, pred_y - ball_size/2, 
                pred_x + ball_size/2, pred_y + ball_size/2]
    
    def _is_reasonable_detection(self, bbox, confidence):
        """Check if detection is reasonably close to expected position."""
        if self.last_center is None:
            return True  # First detection, accept it
        
        # Calculate center of new detection
        new_center = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
        
        # Calculate distance from last known position
        distance = np.sqrt((new_center[0] - self.last_center[0])**2 + 
                        (new_center[1] - self.last_center[1])**2)
        
        # Reasonable distance threshold (adjust as needed)
        max_reasonable_distance = 100  # pixels per frame
        
        if distance <= max_reasonable_distance:
            self.last_rejection_confidences.clear()
            return True  # Reasonable distance, accept it
        
        # maybe later lets look at the case that it Could jump on one high-confidence detection after 2 low-confidence rejections, maybe later we keep track of confidence and if its 3 high ocnfidence rejectiosn its it, Jump if multiple high-confidence rejections
        self.last_rejection_confidences.append(confidence) #  if a dtection is rejected, also keeps track of consecutive_rejections
        self.last_rejection_confidences = self.last_rejection_confidences[-2:]
        consecutive_rejections = len(self.last_rejection_confidences)

        # Allow jump if high confidence and multiple rejections
        if (all(conf > 0.5 for conf in self.last_rejection_confidences) and consecutive_rejections >= 2):
            self.last_rejection_confidences.clear()
            return True
        
        # Very high confidence gets more aggressive jump, most likely th eball so we need to go right away
        if confidence >= 0.80:
            self.last_rejection_confidences.clear()
            return True
        
        return False
