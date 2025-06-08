from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel

class TeamAssigner: # idea to improve this - keep history of what each player was identified as and based on those data points keep color or change color, or maybe 3 frames in a row that assign a person to a team then they are on that team.
    # """
    # Assigns players to teams based on jersey color using a vision-language model (CLIP).
    # """

    def __init__(self,
                 team_1_class_name="white shirt",
                 team_2_class_name="dark blue shirt"):
        self.team_colors = {}
        self.player_team_dict = {}

        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name

    def load_model(self):
        # """
        # Loads the pre-trained CLIP model.
        # """
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    def get_player_color(self, frame, bbox):
        # """
        # Classifies the jersey color of a player using CLIP.

        # Args:
        #     frame (np.ndarray): Current frame.
        #     bbox (list or tuple): [x1, y1, x2, y2] bounding box.

        # Returns:
        #     str: Predicted jersey description.
        # """
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        if image.size == 0:
            return "unknown"

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        classes = [self.team_1_class_name, self.team_2_class_name]
        inputs = self.processor(text=classes, images=pil_image, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

        return classes[probs.argmax(dim=1)[0]]

    def get_player_team(self, frame, player_bbox, player_id):
        # """
        # Returns team assignment for a single player, using cache if available.

        # Returns:
        #     int: Team ID (1 or 2)
        # """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        if player_color == self.team_1_class_name:
            team_id = 1
        else:
            team_id = 2

        self.player_team_dict[player_id] = team_id
        return team_id

    def get_frame_team_assignments(self, frame, track_dict):
        # """
        # Assigns teams to all players in the current frame.

        # Args:
        #     frame (np.ndarray): The current frame.
        #     track_dict (dict): {player_id: {'bbox': [...]}} for this frame.

        # Returns:
        #     dict: {player_id: team_id}
        # """
        assignment = {}

        for player_id, player in track_dict.items():
            team_id = self.get_player_team(frame, player["bbox"], player_id)
            assignment[player_id] = team_id

        return assignment