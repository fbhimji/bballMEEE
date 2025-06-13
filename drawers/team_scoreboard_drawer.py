import cv2

class TeamSoreboardDrawer:
    def __init__(self):
        self.team_1_score = 0
        self.team_2_score = 0
    
    def update_score(self, player_assignment, last_player_with_ball, hoop_data): # player_assignmetn returns {player_id: team_id} last player a list with lat 2 players with ball and returns player_id for [0] and -1 if no one has ball, hoop_data returns -1 if ball made was detected or the location of the hoop
        if hoop_data != -1:
            return
        else:
            if not last_player_with_ball:
                print("HOOP DETECTED, but list of last_players is empty. Cannot assign score.")
                return
            scoring_player_id = last_player_with_ball[0][0]
            # assist_player = last_player_with_ball[1][0] later for assist logic
            scoring_team = player_assignment.get(scoring_player_id)
            if scoring_team == 1:
                self.team_1_score += 1
            elif scoring_team == 2:
                self.team_2_score += 1
            else:
                # This will now correctly trigger if the player wasn't found in player_assignment.
                print(f"COULD NOT DETECT WHO SCORED. Player ID: {scoring_player_id}, Team: {scoring_team}")

    def draw_frame(self, frame, player_assignment, last_player_with_ball, hoop_data):
        """
        Processes a single frame to update the score and draw the scoreboard.

        Args:
            frame (numpy.ndarray): The single video frame to draw on.
            player_assignment (dict): Player assignment dictionary for this frame.
            last_player_with_ball (int): The last player with the ball for this frame.

        Returns:
            numpy.ndarray: The frame with the scoreboard drawn on it.
        """
        # 1. Update the score based on the current frame's events
        self.update_score(player_assignment, last_player_with_ball, hoop_data)

        # 2. Draw the scoreboard overlay on the frame
        
        # Draw a semi-transparent rectangle
        overlay = frame.copy()
        font_scale = 0.9
        font_thickness = 2
        
        # Define positions based on frame dimensions
        frame_height, frame_width = overlay.shape[:2]
        rect_x1 = int(frame_width * 0.01) 
        rect_y1 = int(frame_height * 0.02)
        rect_x2 = int(frame_width * 0.25)  
        rect_y2 = int(frame_height * 0.15)
        
        # Text positions
        text_x = int(frame_width * 0.03)  
        text_y1 = int(frame_height * 0.07)  
        text_y2 = int(frame_height * 0.13)

        # Draw the rectangle and apply transparency
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Put the current score text on the frame
        cv2.putText(frame, f"Team 1 Score: {self.team_1_score}", (text_x, text_y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
        cv2.putText(frame, f"Team 2 Score: {self.team_2_score}", (text_x, text_y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

        # 3. Return the single modified frame
        return frame

            