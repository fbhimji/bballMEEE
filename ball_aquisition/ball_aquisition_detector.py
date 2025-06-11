import sys 
sys.path.append('../')
from drawers.utils import measure_distance, get_center_of_bbox

class BallAquisitionDetector:
    def __init__(self):
        self.possession_threshold = 50 # max distance from player
        self.min_frames = 5 # min frames a ball needs to be with player so they have posession
        self.containment_threshold = 0.8 # the bounding box overalp percentage if ball overlaps 80 with player that player has ball

        self.consecutive_possession_count = {}  
    
    def get_key_basketball_player_assignment_points(self, player_bbox, ball_center):
        ball_center_x = ball_center[0]
        ball_center_y = ball_center[1]

        x1, y1, x2, y2 = player_bbox
        width = x2-x1
        height = y2-y1

        #see if ball is in y coordinates of bbox or x coordinates
        output_points = []

        if ball_center_y > y1 and ball_center_y < y2: # within limits of bbox of player
            output_points.append((x1, ball_center_y)) 
            output_points.append((x2, ball_center_y))  
        
        if ball_center_x > x1 and ball_center_x < x2: # within limits of bbox of player
            output_points.append((ball_center_x, y1))
            output_points.append((ball_center_x, y2))

        output_points += [
            (x1 + width//2, y1),          # top center
            (x2, y1),                      # top right
            (x1, y1),                      # top left
            (x2, y1 + height//2),          # center right
            (x1, y1 + height//2),          # center left
            (x1 + width//2, y1 + height//2), # center point
            (x2, y2),                      # bottom right
            (x1, y2),                      # bottom left
            (x1 + width//2, y2),          # bottom center
            (x1 + width//2, y1 + height//3), # mid-top center
        ]

        return output_points
    
    def find_minimum_distance_to_ball(self, ball_center, player_bbox):
        key_points = self.get_key_basketball_player_assignment_points(player_bbox, ball_center)

        min_dist = 99999

        for key_point in key_points:
            distance = measure_distance(ball_center, key_point)
            if distance < min_dist:
                min_dist = distance
        return min_dist
    
    def calculate_ball_containment_ratio(self, player_bbox, ball_bbox):

        px1, py1, px2, py2 = player_bbox
        bx1, by1, bx2, by2 = ball_bbox

        ball_area = (bx2-bx1) * (by2-by1)
        
        #this will help us get the intersection part of the boxes between player and ball
        intersection_x1 = max(px1, bx1)
        intersection_y1 = max(py1, by1)
        intersection_x2 = min(px2, bx2)
        intersection_y2 = min(py2, by2)

        if intersection_x2 < intersection_x1 or intersection_y2 < intersection_y1:
            return 0.0
            
        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)

        containment_ratio = intersection_area/ball_area

        return containment_ratio
    
    def find_best_candidate_for_possession(self, ball_center, player_tracks_frame, ball_bbox):

        high_containment_players = []
        regular_distance_players = []

        for player_id, player_info in player_tracks_frame.items():
            player_bbox = player_info.get("bbox", [])
            if not player_bbox:
                continue

            containment = self.calculate_ball_containment_ratio(player_bbox, ball_bbox)
            min_distance = self.find_minimum_distance_to_ball(ball_center, player_bbox)

            #see if containemnt is better to use or min distance, if player is containing ball, and if distance is too far dont add this player at all
            if containment > self.containment_threshold:
                high_containment_players.append((player_id, containment))
            else:
                regular_distance_players.append((player_id, min_distance))

        # FIX: Move logic outside the loop
        if high_containment_players:
            best_candidate = max(high_containment_players, key=lambda x: x[1])
            return best_candidate[0]
        
        # second priority 
        if regular_distance_players:
            best_candidate = min(regular_distance_players, key=lambda x: x[1])
            if best_candidate[1] < self.possession_threshold:
                return best_candidate[0]
            
        return -1
        
    def detect_ball_possession(self, player_tracks_frame, ball_tracks_frame):
        # returns -1's so by default no onw has ball and if we have best candidate we have someone w ball
        ball_info = ball_tracks_frame.get(1, {})
        if not ball_info:
            self.consecutive_possession_count = {}
            return -1
        ball_bbox = ball_info.get("bbox", [])
        if not ball_bbox:
            self.consecutive_possession_count = {} 
            return -1
        ball_center = get_center_of_bbox(ball_bbox)

        best_player_id = self.find_best_candidate_for_possession(ball_center, player_tracks_frame, ball_bbox)

        if best_player_id != -1:
            number_of_consecutive_frames = self.consecutive_possession_count.get(best_player_id, 0)+1 
            self.consecutive_possession_count = {best_player_id:number_of_consecutive_frames}  

            if self.consecutive_possession_count[best_player_id] >= self.min_frames:
                return best_player_id
        else:
            self.consecutive_possession_count = {}
        return -1