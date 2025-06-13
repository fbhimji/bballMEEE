from ultralytics import YOLO

class CourtKeypointDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.model.to("mps")
    
    def get_court_keypoints(self, frame):
        detections = self.model.predict(frame,conf=0.5,verbose=False)[0]
        return detections.keypoints