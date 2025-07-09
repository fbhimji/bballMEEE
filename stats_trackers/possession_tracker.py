from collections import deque


class PossessionTracker:
    """
    Tracks who has the ball each frame (just logs basic info).
    """
    def __init__(self):
        self.possession_history = deque(maxlen=600)  # Last 20 sec at 30fps
        self.current_team = None

    def log_possession(self, player_with_ball, team_assignment, frame_count):
        # Who's got the ball (None for loose ball)
        current_team = team_assignment.get(player_with_ball) if player_with_ball != -1 else None
        self.possession_history.append({
            'frame': frame_count,
            'player_id': player_with_ball,
            'team': current_team
        })
        # Possession change?
        changed = False
        if current_team != self.current_team:
            changed = self.current_team is not None
            self.current_team = current_team
        return changed

    def get_possession_sequence(self, end_frame, lookback_frames=300):
        # Simple windowed slice: last `lookback_frames` before end_frame
        start_frame = max(0, end_frame - lookback_frames)
        return [p for p in self.possession_history if start_frame <= p['frame'] <= end_frame]