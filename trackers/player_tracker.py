from ultralytics import YOLO
import supervision as sv

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.model.to("mps")
        self.tracker = sv.ByteTrack()
        self.cls_names_inv = {v: k for k, v in self.model.names.items()}

    def detect_frame(self, frame):
        """
        Detect players in a single frame.
        """
        results = self.model.predict(frame, conf=0.5, verbose=False)
        return results[0]

    def get_object_tracks(self, frame):
        """
        Get tracking results for a single frame.

        Returns:
            tuple: (annotated frame, track_dict)
        """
        result = self.detect_frame(frame)
        detection_supervision = sv.Detections.from_ultralytics(result)
        detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

        track_dict = {}
        for det in detection_with_tracks:
            bbox = det[0].tolist()
            cls_id = det[3]
            track_id = det[4]

            if cls_id == self.cls_names_inv['Player']:
                track_dict[track_id] = {"bbox": bbox}

        return track_dict
