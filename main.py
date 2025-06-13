import argparse
import cv2

from utils import VideoStreamer
from trackers import PlayerTracker, BallTracker, HoopTracker
from team_assigner.team_assigner import TeamAssigner
from drawers.player_tracks_drawer import PlayerTracksDrawer
from drawers.ball_tracks_drawer import BallTracksDrawer
from drawers.hoop_tracks_drawer import HoopTracksDrawer
from ball_aquisition import BallAquisitionDetector
from court_keypoint_detector.court_keypoint_detector import CourtKeypointDetector
from tactical_view_converter.tactical_view_converter import TacticalViewConverter
from drawers.court_keypoints_drawer import CourtKeypointsDrawer
from drawers.tactical_view_drawer import TacticalViewDrawer
from drawers.team_scoreboard_drawer import TeamSoreboardDrawer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', required=True, help="Path to input video")
    parser.add_argument('--output_video', default="outputs/output.mp4", help="Path to save output")
    parser.add_argument('--save', action='store_true', help="Enable saving output video")
    args = parser.parse_args()

    # --- Initialize components ---
    tracker = PlayerTracker("models/player_detector.pt")
    ball_tracker = BallTracker("models/ball_detection_new.pt")
    hoop_tracker = HoopTracker("models/ball_detection_new.pt")
    court_keypoint_detector = CourtKeypointDetector("models/court_keypoint_detection.pt")
    tactical_view_converter = TacticalViewConverter(court_image_path="./images/basketball_court.png")

    assigner = TeamAssigner()
    assigner.load_model()
    player_drawer = PlayerTracksDrawer()
    ball_drawer = BallTracksDrawer()
    hoop_drawer = HoopTracksDrawer()
    ball_acquisition_detector = BallAquisitionDetector()
    court_keypoint_drawer = CourtKeypointsDrawer()
    tactical_view_drawer = TacticalViewDrawer()
    team_scoreboard = TeamSoreboardDrawer()

    

    # --- Set up video streaming ---
    streamer = VideoStreamer(
        video_path=args.input_video,
        output_path=args.output_video,
        save_output=args.save,
        fps=30
    )

    frame_count = 0

    try:
        for frame in streamer:
            # -- Ball tracking --
            ball_bbox = ball_tracker.get_object_track(frame)  # Detect the ball in the current frame (returns bbox or None)
            ball_tracker.update_history(frame_count, ball_bbox)  # Update the ball's tracked history with motion filtering
            ball_dict = ball_tracker.ball_history[-1] if ball_tracker.ball_history else {}  # Get the most recent ball state for drawing

            # -- Hoop tracking --
            hoop_data = hoop_tracker.get_hoop_bbox(frame) # can also return made hoop if it rocognizes it 

            # -- Player tracking --
            track_dict = tracker.get_object_tracks(frame) # player tracker
            team_assignment = assigner.get_frame_team_assignments(frame, track_dict)

            # Ball Acquisition
            ball_tracks_frame = {1: {"bbox": ball_bbox}} if ball_bbox else {}
            player_with_ball, last_player_with_ball = ball_acquisition_detector.detect_ball_possession(track_dict, ball_tracks_frame, team_assignment)

            # Court Keypoint Detector
            court_keypoints = court_keypoint_detector.get_court_keypoints(frame)

            # -- Drawing --
            frame = ball_drawer.draw_frame(frame, ball_dict)
            frame = player_drawer.draw_frame(frame, track_dict, team_assignment, player_with_ball)
            frame = hoop_drawer.draw_frame(frame, hoop_data)
            frame = team_scoreboard.draw_frame(frame, team_assignment, last_player_with_ball, hoop_data)
            # frame = court_keypoint_drawer.draw_frame(frame, court_keypoints)
            # frame = tactical_view_drawer.draw_frame(frame, tactical_view_converter.court_image_path, tactical_view_converter.width, tactical_view_converter.height, tactical_view_converter.key_points)

            # -- Show and optionally save frame --
            cv2.imshow('Live Video', frame)
            streamer.write(frame)

            if cv2.waitKey(int(1000 / streamer.fps)) & 0xFF == ord('q'):
                break

            frame_count += 1

    finally:
        streamer.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
