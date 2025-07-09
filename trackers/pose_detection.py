import cv2
import mediapipe as mp
import numpy as np

class PoseDetector:
    def __init__(self, model_complexity=1, min_detection_confidence=0.5):
        """
        Initialize pose detector with MediaPipe
        
        Args:
            model_complexity: 0=Lite, 1=Full, 2=Heavy
            min_detection_confidence: Confidence threshold
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        
    def detect_poses_in_frame(self, frame):
        """
        Detect poses for all people in frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            pose_results: MediaPipe pose results
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.pose.process(rgb_frame)
        
        return results
    
    def extract_poses_for_players(self, frame, track_dict):
        """
        Extract pose keypoints for each tracked player
        
        Args:
            frame: Input frame
            track_dict: Player tracking results from PlayerTracker
            
        Returns:
            dict: {track_id: pose_keypoints} for each player
        """
        player_poses = {}
        
        for track_id, player_info in track_dict.items():
            bbox = player_info["bbox"]
            
            # Crop player region
            x1, y1, x2, y2 = map(int, bbox)
            player_crop = frame[y1:y2, x1:x2]
            
            if player_crop.size > 0:  # Valid crop
                # Get pose for this player
                pose_results = self.detect_poses_in_frame(player_crop)
                
                if pose_results.pose_landmarks:
                    # Convert relative coordinates back to full frame coordinates
                    keypoints = self._convert_to_frame_coordinates(
                        pose_results.pose_landmarks, 
                        x1, y1, x2, y2
                    )
                    player_poses[track_id] = keypoints
                else:
                    player_poses[track_id] = None
                    
        return player_poses
    
    def _convert_to_frame_coordinates(self, landmarks, x1, y1, x2, y2):
        """
        Convert relative landmark coordinates to full frame coordinates
        """
        keypoints = {}
        crop_width = x2 - x1
        crop_height = y2 - y1
        
        for idx, landmark in enumerate(landmarks.landmark):
            # Convert from relative crop coordinates to full frame coordinates
            x = int(landmark.x * crop_width + x1)
            y = int(landmark.y * crop_height + y1)
            confidence = landmark.visibility
            
            keypoints[idx] = {
                'x': x,
                'y': y,
                'confidence': confidence
            }
            
        return keypoints
    
    def get_hand_positions(self, pose_keypoints):
        """
        Extract hand positions from pose keypoints
        
        Args:
            pose_keypoints: Keypoints dict from extract_poses_for_players
            
        Returns:
            dict: {'left_wrist': (x, y), 'right_wrist': (x, y)} or None
        """
        if not pose_keypoints:
            return None
            
        # MediaPipe pose landmark indices
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        
        hands = {}
        
        if LEFT_WRIST in pose_keypoints and pose_keypoints[LEFT_WRIST]['confidence'] > 0.5:
            hands['left_wrist'] = (
                pose_keypoints[LEFT_WRIST]['x'], 
                pose_keypoints[LEFT_WRIST]['y']
            )
            
        if RIGHT_WRIST in pose_keypoints and pose_keypoints[RIGHT_WRIST]['confidence'] > 0.5:
            hands['right_wrist'] = (
                pose_keypoints[RIGHT_WRIST]['x'], 
                pose_keypoints[RIGHT_WRIST]['y']
            )
            
        return hands if hands else None