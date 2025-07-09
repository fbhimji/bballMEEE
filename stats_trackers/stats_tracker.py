from .possession_tracker import PossessionTracker
from .offensive_stats import OffensiveStats

class StatsTracker:
    """
    Simple basketball stats tracker focused only on made shots and assists.
    Clean integration with your existing main.py code.
    """
    
    def __init__(self):
        # Initialize components
        self.possession_tracker = PossessionTracker()        # Logs possession data
        self.shot_detector = OffensiveStats()         # Handles shots and assists
        
        # Debug mode - set to True to see detailed analysis
        self.debug_mode = False
        
    def process_frame(self, player_with_ball, team_assignment, frame_count):
        """
        Process one frame of basketball action.
        Call this every frame in your main.py loop - very lightweight.
        
        Args:
            player_with_ball: Player ID from your ball_acquisition_detector
            team_assignment: Dict from your team_assigner  
            frame_count: Your existing frame_count from main.py
            
        Returns:
            None: This method just logs data, no events generated
        """
        # Log possession data (very fast operation every frame)
        possession_changed = self.possession_tracker.log_possession(
            player_with_ball, team_assignment, frame_count
        )
        
        # Note: We don't analyze possession changes here
        # We only analyze when baskets are definitely made
        
        return None
    
    def process_made_basket(self, frame_count, team_assignment):
        """
        Process a made basket detected by your HoopTracker.
        This is where the magic happens - we work backward to find shooter and assister.
        
        Args:
            frame_count: Frame where basket was made (from your hoop_interaction)
            team_assignment: Current team assignment
            
        Returns:
            dict: Basket analysis results with shooter and assister
        """
        print(f"🏀 BASKET MADE at frame {frame_count} - Analyzing possession...")
        
        # Get the possession sequence that led to this basket
        possession_sequence = self.possession_tracker.get_possession_sequence(frame_count)
        
        # Debug: Print possession sequence if debug mode enabled
        if self.debug_mode:
            self.shot_detector.print_debug_possession_sequence(possession_sequence)
        
        # Analyze for shooter and assister
        basket_analysis = self.shot_detector.analyze_made_basket(
            possession_sequence, team_assignment, frame_count
        )
        
        # Reset possession tracking (basket ends possession)
        self.possession_tracker.current_team = None
        self.possession_tracker.possession_start_frame = None
        
        return basket_analysis
    
    # === SIMPLE INTERFACE METHODS (same pattern as your other components) ===
    
    def get_player_stats(self, player_id):
        """
        Get current stats for a player.
        
        Args:
            player_id: Player ID
            
        Returns:
            dict: Player statistics (points, made shots, assists)
        """
        return self.shot_detector.get_player_stats(player_id)
    
    def get_team_score(self, team_id):
        """
        Get current team score (for your scoreboard).
        
        Args:
            team_id: Team ID (1 or 2)
            
        Returns:
            int: Team score
        """
        return self.shot_detector.get_team_score(team_id)
    
    def print_current_stats(self, team_assignment):
        """
        Print current game statistics (for debugging).
        
        Args:
            team_assignment: Dict mapping player_id -> team_id
        """
        print(f"\n📊 CURRENT GAME STATS:")
        print(f"Team 1: {self.get_team_score(1)} points")
        print(f"Team 2: {self.get_team_score(2)} points")
        
        print(f"\nPlayer Stats:")
        for player_id in team_assignment.keys():
            stats = self.get_player_stats(player_id)
            team = team_assignment.get(player_id)
            
            if stats['points'] > 0 or stats['assists'] > 0:  # Only show players with stats
                print(f"  Player {player_id} (Team {team}): "
                      f"{stats['points']} pts, "
                      f"{stats['field_goals_made']} FGM, "
                      f"{stats['assists']} ast")
    
    def enable_debug_mode(self):
        """Enable detailed debug output for possession analysis."""
        self.debug_mode = True
        print("🐛 Debug mode enabled - will show detailed possession sequences")
    
    def disable_debug_mode(self):
        """Disable debug output for normal operation.""" 
        self.debug_mode = False
        print("📊 Debug mode disabled - normal operation")
    
    def get_game_summary(self, team_assignment):
        """
        Get simple game summary focused on scoring.
        
        Args:
            team_assignment: Dict mapping player_id -> team_id
            
        Returns:
            dict: Game summary with team scores and top scorers
        """
        summary = {
            'team_scores': {
                1: self.get_team_score(1),
                2: self.get_team_score(2)
            },
            'top_scorers': []
        }
        
        # Find top scorers
        player_scores = []
        for player_id in team_assignment.keys():
            stats = self.get_player_stats(player_id)
            if stats['points'] > 0:
                player_scores.append({
                    'player_id': player_id,
                    'points': stats['points'],
                    'assists': stats['assists'],
                    'team': team_assignment.get(player_id)
                })
        
        # Sort by points
        summary['top_scorers'] = sorted(player_scores, key=lambda x: x['points'], reverse=True)
        
        return summary