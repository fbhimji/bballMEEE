from ultralytics import YOLO
import supervision as sv
import numpy as np

class HoopTracker: # idea for hoop ball collision ball has to be above and right under hoop within frames of eachotehr?
    def __init__(self, model_path, max_dist=50, max_stale_frames=60):
        self.model = YOLO(model_path)
        self.model.to("mps")
        self.cls_names_inv = {v: k for k, v in self.model.names.items()}
        self.last_hoop_bbox = None
        self.last_confidence = 0
        self.last_seen_frame = -1
        self.frame_counter = 0
        self.max_dist = max_dist
        self.max_stale_frames = max_stale_frames

    def get_hoop_bbox(self, frame):
        """
        Detect the hoop in a given frame, return only if confidence >= 0.5.
        """
        self.frame_counter += 1
        results = self.model.predict(frame, conf=0.3, verbose=False)
        result = results[0]
        detection_supervision = sv.Detections.from_ultralytics(result)

        best_bbox = None
        best_conf = 0

        for det in detection_supervision:
            bbox = det[0].tolist()
            cls_id = det[3]
            conf = det[2]

            if cls_id == self.cls_names_inv.get("Hoop", -1):
                if self.last_hoop_bbox is not None:
                    dist = np.linalg.norm(np.array(bbox[:2]) - np.array(self.last_hoop_bbox[:2]))
                    if dist < self.max_dist and conf > best_conf:
                        best_bbox = bbox
                        best_conf = conf
                    elif dist >= self.max_dist and conf > 0.6:
                        best_bbox = bbox
                        best_conf = conf
                elif conf > best_conf:
                    best_bbox = bbox
                    best_conf = conf

            elif cls_id == self.cls_names_inv.get("made", -1): ## add hoop and ball colition logic too as a check
                print("finallyyyyyyyyy")
                return -1

        if best_bbox is not None:
            self.last_hoop_bbox = best_bbox
            self.last_confidence = best_conf
            self.last_seen_frame = self.frame_counter

        # Don't return stale results — only return current detection if conf >= 0.5
        if best_conf >= 0.5:
            return best_bbox, best_conf
        else:
            return None, 0
