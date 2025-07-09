from collections import defaultdict


class OffensiveStats:
    """
    Simple detector focused only on made shots and assists.
    Uses backward analysis when baskets are made to find shooter and assister.
    """
    
    def __init__(self):
        # Player statistics - only shots and assists for now
        self.player_stats = defaultdict(lambda: {
            'points': 0,
            'field_goals_made': 0,
            'assists': 0
        })
        
        # Team statistics - just score for now
        self.team_stats = {1: {'score': 0}, 2: {'score': 0}}
        
        # Analysis parameters - adjusted for your BallAquisitionDetector
        # Your ball detector already requires 8 consecutive frames for possession
        # So we can be more lenient here since we know possession is real
        self.min_possession_frames = 1    # Any detected possession counts (since it's already filtered)
        self.max_assist_window = 90       # 3 seconds to look back for assists
        
    def analyze_made_basket(self, possession_sequence, team_assignment, frame_count):
        """
        Analyze a possession that ended with a made basket.
        Work backward to find who shot and who assisted.
        
        Args:
            possession_sequence: List of possession frames from PossessionTracker
            team_assignment: Dict mapping player_id -> team_id
            frame_count: Frame where basket was made
            
        Returns:
            dict: Analysis results with shooter, assister, points
        """
        print(f"\n🏀 ANALYZING MADE BASKET at frame {frame_count}")
        
        if not possession_sequence:
            print("   ❌ No possession sequence available")
            return None
            
        # Step 1: Find who shot the ball (last person with meaningful possession)
        shooter_info = self._find_shooter_from_sequence(possession_sequence)
        
        if not shooter_info:
            print("   ❌ Could not identify shooter")
            return None
            
        shooter_id = shooter_info['player_id']
        print(f"   🎯 SHOOTER: Player {shooter_id} (had ball for {shooter_info['duration']} frames)")
        
        # Step 2: Find who assisted (person before shooter with meaningful possession)
        assister_id = self._find_assister_from_sequence(possession_sequence, shooter_id)
        
        if assister_id:
            print(f"   🤝 ASSIST: Player {assister_id} → Player {shooter_id}")
        else:
            print(f"   📊 NO ASSIST: Individual effort by Player {shooter_id}")
        
        # Step 3: Update player and team stats
        self._record_made_shot(shooter_id, assister_id, team_assignment)
        
        return {
            'event_type': 'made_basket',
            'shooter_id': shooter_id,
            'assister_id': assister_id,
            'points': 2,  # Assume 2-pointer for now
            'frame': frame_count,
            'shooter_possession_duration': shooter_info['duration']
        }
    
    def _find_shooter_from_sequence(self, possession_sequence):
        """
        Find who shot the ball by looking for last meaningful possession.
        Work backward through the sequence, allowing for ball flight time.
        
        Args:
            possession_sequence: List of possession frames
            
        Returns:
            dict: Shooter info with player_id, frame, duration
        """
        if not possession_sequence:
            return None
            
        # First, find all players who had meaningful possession in this sequence
        possession_segments = self._find_possession_segments(possession_sequence)
        
        # Look for the last player with meaningful possession
        # (ball might be flying through air at the end)
        for segment in reversed(possession_segments):
            if segment['duration'] >= self.min_possession_frames:
                return {
                    'player_id': segment['player_id'],
                    'frame': segment['end_frame'],
                    'duration': segment['duration']
                }
        
        return None
    
    def _find_possession_segments(self, possession_sequence):
        """
        Break possession sequence into segments by player.
        
        Args:
            possession_sequence: List of possession frames
            
        Returns:
            list: List of possession segments with player_id, start_frame, end_frame, duration
        """
        if not possession_sequence:
            return []
            
        segments = []
        current_player = None
        segment_start = 0
        
        for i, possession in enumerate(possession_sequence):
            player_id = possession['player_id']
            
            # Skip invalid players but track position
            if player_id == -1:
                if current_player is not None:
                    # End current segment
                    segments.append({
                        'player_id': current_player,
                        'start_frame': possession_sequence[segment_start]['frame'],
                        'end_frame': possession_sequence[i-1]['frame'],
                        'start_index': segment_start,
                        'end_index': i-1,
                        'duration': i - segment_start
                    })
                    current_player = None
                continue
            
            # Start new segment or continue current one
            if player_id != current_player:
                if current_player is not None:
                    # End previous segment
                    segments.append({
                        'player_id': current_player,
                        'start_frame': possession_sequence[segment_start]['frame'],
                        'end_frame': possession_sequence[i-1]['frame'],
                        'start_index': segment_start,
                        'end_index': i-1,
                        'duration': i - segment_start
                    })
                
                # Start new segment
                current_player = player_id
                segment_start = i
        
        # Handle final segment
        if current_player is not None:
            segments.append({
                'player_id': current_player,
                'start_frame': possession_sequence[segment_start]['frame'],
                'end_frame': possession_sequence[-1]['frame'],
                'start_index': segment_start,
                'end_index': len(possession_sequence) - 1,
                'duration': len(possession_sequence) - segment_start
            })
        
        return segments
    
    def _find_assister_from_sequence(self, possession_sequence, shooter_id):
        """
        Find who assisted by looking for previous teammate with meaningful possession.
        
        Args:
            possession_sequence: List of possession frames
            shooter_id: ID of player who scored
            
        Returns:
            int: Assister player ID or None
        """
        # Get possession segments for better analysis
        possession_segments = self._find_possession_segments(possession_sequence)
        
        # Find the shooter's segment
        shooter_segment = None
        for segment in possession_segments:
            if segment['player_id'] == shooter_id and segment['duration'] >= self.min_possession_frames:
                shooter_segment = segment
                break
        
        if not shooter_segment:
            return None
        
        # Look for previous meaningful possession by a different player
        for segment in reversed(possession_segments):
            # Skip if it's the shooter or after the shooter
            if (segment['player_id'] == shooter_id or 
                segment['start_index'] >= shooter_segment['start_index']):
                continue
            
            # Check if this player had meaningful possession
            if segment['duration'] >= self.min_possession_frames:
                # Check timing - was it recent enough to be an assist?
                time_gap = shooter_segment['start_index'] - segment['end_index']
                if time_gap <= self.max_assist_window:
                    return segment['player_id']
        
        return None
    
    def _calculate_possession_duration(self, possession_sequence, start_index):
        """
        Calculate how many consecutive frames a player had possession.
        
        Args:
            possession_sequence: List of possession frames
            start_index: Starting index in sequence
            
        Returns:
            int: Number of consecutive frames
        """
        if start_index >= len(possession_sequence):
            return 0
            
        player_id = possession_sequence[start_index]['player_id']
        duration = 1
        
        # Count consecutive frames with same player
        for i in range(start_index + 1, len(possession_sequence)):
            if possession_sequence[i]['player_id'] == player_id:
                duration += 1
            else:
                break  # Different player, stop counting
        
        return duration
    
    def _record_made_shot(self, shooter_id, assister_id, team_assignment):
        """
        Update stats for a made shot.
        
        Args:
            shooter_id: Player who made shot
            assister_id: Player who assisted (or None)
            team_assignment: Dict mapping player_id -> team_id
        """
        # Update shooter stats
        self.player_stats[shooter_id]['field_goals_made'] += 1
        self.player_stats[shooter_id]['points'] += 2
        
        # Update assister stats
        if assister_id:
            self.player_stats[assister_id]['assists'] += 1
        
        # Update team score
        team_id = team_assignment.get(shooter_id)
        if team_id and team_id in self.team_stats:
            self.team_stats[team_id]['score'] += 2
            
        print(f"   📊 STATS UPDATED:")
        print(f"      Player {shooter_id}: +2 PTS, +1 FGM (Total: {self.player_stats[shooter_id]['points']} pts)")
        if assister_id:
            print(f"      Player {assister_id}: +1 AST (Total: {self.player_stats[assister_id]['assists']} ast)")
        print(f"      Team {team_id}: {self.team_stats[team_id]['score']} points")
    
    def get_player_stats(self, player_id):
        """
        Get current stats for a player.
        
        Args:
            player_id: Player ID
            
        Returns:
            dict: Player statistics (points, made shots, assists)
        """
        return self.player_stats[player_id].copy()
    
    def get_team_score(self, team_id):
        """
        Get current team score.
        
        Args:
            team_id: Team ID (1 or 2)
            
        Returns:
            int: Team score
        """
        return self.team_stats[team_id]['score']
    
    def print_debug_possession_sequence(self, possession_sequence):
        """
        Debug method: Print possession sequence to understand what happened.
        
        Args:
            possession_sequence: List of possession frames to analyze
        """
        print(f"\n📋 POSSESSION SEQUENCE DEBUG ({len(possession_sequence)} frames):")
        
        if not possession_sequence:
            print("   ❌ No possession data available")
            return
        
        # Get possession segments for better visualization
        segments = self._find_possession_segments(possession_sequence)
        
        print(f"   📊 Found {len(segments)} possession segments:")
        for i, segment in enumerate(segments):
            status = "✅" if segment['duration'] >= self.min_possession_frames else "❌"
            print(f"   {status} Player {segment['player_id']}: frames {segment['start_frame']}-{segment['end_frame']} ({segment['duration']} frames)")
        
        print(f"   📏 Minimum possession threshold: {self.min_possession_frames} frames (since BallAquisitionDetector already requires 8)")
        print(f"   ⏰ Maximum assist window: {self.max_assist_window} frames")
        
        # Show frame-by-frame if sequence is short
        if len(possession_sequence) <= 20:
            print(f"   📝 Frame-by-frame breakdown:")
            for possession in possession_sequence:
                player = possession['player_id'] if possession['player_id'] != -1 else "None"
                print(f"      Frame {possession['frame']}: Player {player}")