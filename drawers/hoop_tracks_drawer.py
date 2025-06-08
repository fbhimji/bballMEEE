from .utils import draw_rectangle

class HoopTracksDrawer:
    def __init__(self, color=(0, 0, 255)):
        self.color = color

    def draw_frame(self, frame, hoop_data):
        bbox, conf = hoop_data
        if bbox is not None and conf >= 0.5:
            frame = draw_rectangle(frame, bbox, color=self.color, label="Hoop")
        return frame
