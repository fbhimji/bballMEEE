import supervision as sv


class CourtKeypointsDrawer:
    def __init__(self):
        self.keypoint_color = '#ff2c2c'

        # Initialize annotators once for efficiency
        self.vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            radius=8)
        
        self.vertex_label_annotator = sv.VertexLabelAnnotator( # keypoint number to show where on court
            color=sv.Color.from_hex(self.keypoint_color),
            text_color=sv.Color.WHITE,
            text_scale=0.5,
            text_thickness=1
        )
    
    def draw_frame(self, frame, keypoints):
        """
        Draws court keypoints on a single frame.
        Args:
            frame: A single frame (as NumPy array or image object) on which to draw.
            keypoints: The keypoints for this frame - (x, y) coordinates of court keypoints.
        Returns:
            Annotated frame with keypoints drawn on it.
        """
        annotated_frame = frame.copy()
        
        if keypoints is not None:
            # Draw dots
            annotated_frame = self.vertex_annotator.annotate( #above
                scene=annotated_frame,
                key_points=keypoints)
            
            # Draw labels
            # Convert PyTorch tensor to numpy array
            keypoints_numpy = keypoints.cpu().numpy()
            annotated_frame = self.vertex_label_annotator.annotate( #above
                scene=annotated_frame,
                key_points=keypoints_numpy)
            
        return annotated_frame