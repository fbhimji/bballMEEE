import cv2
import os

class VideoStreamer:
    def __init__(self, video_path, output_path=None, save_output=False, fps=24):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps    = fps

        self.save_output = save_output
        self.out = None

        if self.save_output:
            if output_path is None:
                raise ValueError("Output path required for saving video.")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(output_path, fourcc, fps, (self.width, self.height))

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.release()
            raise StopIteration
        return frame

    def write(self, frame):
        if self.save_output and self.out:
            self.out.write(frame)

    def release(self):
        self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()
