from .utils import draw_ellipse, draw_triangle

class PlayerTracksDrawer:
    def __init__(self, team_1_color=[255, 245, 238], team_2_color=[128, 0, 0]):
        self.default_player_team_id = 1
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color

    def draw_frame(self, frame, track_dict, assignment_dict, player_with_ball):
        frame = frame.copy()
        for track_id, player in track_dict.items():
            team_id = assignment_dict.get(track_id, self.default_player_team_id)

            color = self.team_1_color if team_id == 1 else self.team_2_color

            if track_id == player_with_ball: # player that has ball
                frame = draw_triangle(frame, player["bbox"], (0,0,255))

            frame = draw_ellipse(frame, player['bbox'], color, track_id)
        return frame
