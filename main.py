import argparse
import cv2

from utils import VideoStreamer
from trackers import PlayerTracker, BallTracker, HoopTracker
from team_assigner.team_assigner import TeamAssigner
from drawers.player_tracks_drawer import PlayerTracksDrawer
from drawers.ball_tracks_drawer import BallTracksDrawer
from drawers.hoop_tracks_drawer import HoopTracksDrawer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', required=True, help="Path to input video")
    parser.add_argument('--output_video', default="outputs/output.avi", help="Path to save output")
    parser.add_argument('--save', action='store_true', help="Enable saving output video")
    args = parser.parse_args()

    # --- Initialize components ---
    tracker = PlayerTracker("models/player_detector.pt")
    ball_tracker = BallTracker("models/ball_detector.pt")
    hoop_tracker = HoopTracker("models/player_detector.pt")
    assigner = TeamAssigner()
    assigner.load_model()
    player_drawer = PlayerTracksDrawer()
    ball_drawer = BallTracksDrawer()
    hoop_drawer = HoopTracksDrawer()
    

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
            hoop_data = hoop_tracker.get_hoop_bbox(frame)

            # -- Player tracking --
            if frame_count % 50 == 0:
                assigner.player_team_dict.clear()

            track_dict = tracker.get_object_tracks(frame)
            team_assignment = assigner.get_frame_team_assignments(frame, track_dict)

            # -- Drawing --
            frame = ball_drawer.draw_frame(frame, ball_dict)
            frame = player_drawer.draw_frame(frame, track_dict, team_assignment)
            frame = hoop_drawer.draw_frame(frame, hoop_data)

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
