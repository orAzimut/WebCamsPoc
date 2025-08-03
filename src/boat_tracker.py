import time

class SimpleBoatTracker:
    """Simple tracker that only tracks last save time per boat ID"""
    
    def __init__(self):
        self.last_save_times = {}  # boat_id -> last save timestamp
        self.save_counts = {}      # boat_id -> number of saves
        self.min_save_interval = 4  # 10 seconds as requested
        self.max_saves_per_boat = 100
        
    def should_save_boat(self, boat_id, current_time):
        """Check if we should save this boat based on time interval"""
        
        # First time seeing this boat ID
        if boat_id not in self.last_save_times:
            return True, "New boat ID detected"
        
        # Check time interval
        time_since_last_save = current_time - self.last_save_times[boat_id]
        if time_since_last_save >= self.min_save_interval:
            return True, f"Time interval met: {time_since_last_save:.1f}s >= {self.min_save_interval}s"
        
        # Check max saves limit
        if self.save_counts.get(boat_id, 0) >= self.max_saves_per_boat:
            return False, f"Max saves reached for boat {boat_id}"
        
        return False, f"Time interval too short: {time_since_last_save:.1f}s < {self.min_save_interval}s"
    
    def record_save(self, boat_id, current_time):
        """Record that we saved an image for this boat"""
        self.last_save_times[boat_id] = current_time
        self.save_counts[boat_id] = self.save_counts.get(boat_id, 0) + 1
    
    def get_tracked_boat_ids(self):
        """Get all currently tracked boat IDs"""
        return set(self.last_save_times.keys())
    
    def get_save_count(self, boat_id):
        """Get save count for a specific boat"""
        return self.save_counts.get(boat_id, 0)