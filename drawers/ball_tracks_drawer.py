from .utils import draw_triangle

class BallTracksDrawer:
    def __init__(self):
        self.ball_pointer_color = (0, 255, 0)  # Green (BGR)

    def draw_frame(self, frame, track_dict):
        """
        Draw ball pointer on a single frame.

        Args:
            frame (np.ndarray): Current video frame
            track_dict (dict): {1: {"bbox": [...]}} if ball is present

        Returns:
            np.ndarray: Frame with triangle annotation
        """
        frame = frame.copy()
        
        if not track_dict:
            return frame

        for _, track in track_dict.items():
            bbox = track.get("bbox")
            if bbox is None:
                continue
            frame = draw_triangle(frame, bbox, self.ball_pointer_color)

        return frame
