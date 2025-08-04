import time
import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional

class BoatTracker:
    """Smart tracker that saves boats based on view diversity for ReID training"""
    
    def __init__(self):
        self.last_save_times = {}       # boat_id -> last save timestamp
        self.save_counts = {}           # boat_id -> number of saves
        self.boat_features = {}         # boat_id -> dict of features from last save
        
        # Configuration - purely content-based filtering
        
        # More restrictive thresholds
        self.bbox_change_threshold = 0.3     #stricter - higer
        self.histogram_threshold = 0.6        # stricter - lower 
        self.movement_threshold = 75        # stricter higher 
        self.feature_similarity_threshold = 0.1  # stricter - lower
        
        # Require multiple criteria for change detection
        self.min_criteria_for_change = 2      # At least 2 out of 4 criteria must detect change
        
    def should_save_boat(self, boat_id, current_time, bbox=None, boat_image=None, frame_shape=None):
        """
        Enhanced decision making for saving boats based on view diversity
        """
        
        # First time seeing this boat ID
        if boat_id not in self.last_save_times:
            if bbox is not None and boat_image is not None:
                self._record_boat_features(boat_id, current_time, bbox, boat_image)
            return True, "New boat ID detected"
        
        # Enhanced detection requires bbox and image
        if bbox is None or boat_image is None:
            return False, "Missing bbox or boat_image for view analysis"
        
        # Get previous features
        prev_features = self.boat_features.get(boat_id, {})
        if not prev_features:
            self._record_boat_features(boat_id, current_time, bbox, boat_image)
            return True, "No previous features stored"
        
        # Check for significant view changes
        view_change_detected, reason, criteria_met_count, met_criteria = self._detect_view_change(bbox, boat_image, prev_features)
        
        if view_change_detected:
            self._record_boat_features(boat_id, current_time, bbox, boat_image)
            return True, f"View change detected ({criteria_met_count}/{4} criteria): {reason}"
        
        # Build detailed message showing which criteria were met vs needed
        all_criteria = ['bbox', 'histogram', 'movement', 'features']
        unmet_criteria = [c for c in all_criteria if c not in met_criteria]
        
        return False, f"No significant view change. Met: [{', '.join(met_criteria)}], Need {self.min_criteria_for_change}/{4}, Missing: [{', '.join(unmet_criteria)}]"
    
    def _detect_view_change(self, current_bbox, current_image, prev_features):
        """Detect if there's a significant view change - now requires multiple criteria"""
        
        criteria_results = {}
        met_criteria = []
        
        # Method 1: Bounding box analysis
        bbox_changed = self._bbox_changed_significantly(current_bbox, prev_features.get('bbox'))
        criteria_results['bbox'] = bbox_changed
        if bbox_changed:
            met_criteria.append("bbox")
        
        # Method 2: Histogram comparison  
        hist_changed = self._histogram_changed_significantly(current_image, prev_features.get('histogram'))
        criteria_results['histogram'] = hist_changed
        if hist_changed:
            met_criteria.append("histogram")
        
        # Method 3: Position change
        pos_changed = self._position_changed_significantly(current_bbox, prev_features.get('bbox'))
        criteria_results['movement'] = pos_changed
        if pos_changed:
            met_criteria.append("movement")
        
        # Method 4: Visual features comparison
        features_changed = self._visual_features_changed(current_image, prev_features.get('orb_features'))
        criteria_results['features'] = features_changed
        if features_changed:
            met_criteria.append("features")
        
        # Count how many criteria detected change
        criteria_met_count = len(met_criteria)
        
        # Require multiple criteria to detect significant change
        significant_change = criteria_met_count >= self.min_criteria_for_change
        
        # Create detailed reason text
        if met_criteria:
            reason_text = f"Met criteria: {', '.join(met_criteria)}"
        else:
            reason_text = "no criteria met"
        
        return significant_change, reason_text, criteria_met_count, met_criteria
    
    def _bbox_changed_significantly(self, current_bbox, prev_bbox):
        """Check if bounding box geometry changed significantly - more restrictive"""
        if prev_bbox is None:
            return True
        
        curr_x, curr_y, curr_w, curr_h = current_bbox
        prev_x, prev_y, prev_w, prev_h = prev_bbox
        
        # Calculate aspect ratio change
        curr_aspect = curr_w / curr_h if curr_h > 0 else 0
        prev_aspect = prev_w / prev_h if prev_h > 0 else 0
        
        if prev_aspect > 0:
            aspect_change = abs(curr_aspect - prev_aspect) / prev_aspect
            if aspect_change > self.bbox_change_threshold:
                return True
        
        # Calculate size change
        curr_area = curr_w * curr_h
        prev_area = prev_w * prev_h
        
        if prev_area > 0:
            size_change = abs(curr_area - prev_area) / prev_area
            if size_change > self.bbox_change_threshold:
                return True
        
        return False
    
    def _histogram_changed_significantly(self, current_image, prev_histogram):
        """Check if color histogram changed significantly - more restrictive"""
        if prev_histogram is None:
            return True
        
        try:
            # Ensure image is valid
            if current_image is None or current_image.size == 0:
                return False
                
            # Calculate histogram for current image
            if len(current_image.shape) == 3:
                current_hist = cv2.calcHist([current_image], [0, 1, 2], None, 
                                          [8, 8, 8], [0, 256, 0, 256, 0, 256])
            else:
                current_hist = cv2.calcHist([current_image], [0], None, [256], [0, 256])
            
            # Normalize histograms
            cv2.normalize(current_hist, current_hist)
            
            # Calculate correlation (closer to 1.0 = more similar)
            correlation = cv2.compareHist(current_hist, prev_histogram, cv2.HISTCMP_CORREL)
            
            # Only consider it changed if correlation is quite low
            return correlation < self.histogram_threshold
            
        except Exception as e:
            print(f"Histogram comparison failed: {e}")
            return False  # If histogram calculation fails, don't assume change
    
    def _position_changed_significantly(self, current_bbox, prev_bbox):
        """Check if boat moved significantly - more restrictive"""
        if prev_bbox is None:
            return True
        
        curr_center_x = current_bbox[0] + current_bbox[2] // 2
        curr_center_y = current_bbox[1] + current_bbox[3] // 2
        
        prev_center_x = prev_bbox[0] + prev_bbox[2] // 2
        prev_center_y = prev_bbox[1] + prev_bbox[3] // 2
        
        distance = np.sqrt((curr_center_x - prev_center_x)**2 + 
                          (curr_center_y - prev_center_y)**2)
        
        return distance > self.movement_threshold
    
    def _visual_features_changed(self, current_image, prev_features):
        """Check if ORB visual features changed significantly - more restrictive"""
        if prev_features is None:
            return True
        
        try:
            # Ensure image is valid
            if current_image is None or current_image.size == 0:
                return False
                
            # Initialize ORB detector
            orb = cv2.ORB_create(nfeatures=100)
            
            # Convert to grayscale if needed
            if len(current_image.shape) == 3:
                gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = current_image
            
            # Detect keypoints and descriptors
            kp, desc = orb.detectAndCompute(gray, None)
            
            if desc is None or prev_features is None:
                return True
            
            # Match features using brute force matcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(desc, prev_features)
            
            # Calculate similarity based on good matches
            if len(matches) == 0:
                return True
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            good_matches = [m for m in matches if m.distance < 30]  # More restrictive distance threshold
            
            similarity = len(good_matches) / max(len(desc), len(prev_features))
            
            return similarity < self.feature_similarity_threshold
            
        except Exception as e:
            print(f"Visual feature comparison failed: {e}")
            return False  # If feature extraction fails, don't assume change
    
    def _record_boat_features(self, boat_id, current_time, bbox, boat_image):
        """Record features of the boat for future comparison"""
        try:
            features = {
                'bbox': bbox,
                'timestamp': current_time
            }
            
            # Ensure image is valid
            if boat_image is None or boat_image.size == 0:
                self.boat_features[boat_id] = features
                return
            
            # Store histogram
            try:
                if len(boat_image.shape) == 3:
                    histogram = cv2.calcHist([boat_image], [0, 1, 2], None, 
                                           [8, 8, 8], [0, 256, 0, 256, 0, 256])
                else:
                    histogram = cv2.calcHist([boat_image], [0], None, [256], [0, 256])
                
                cv2.normalize(histogram, histogram)
                features['histogram'] = histogram
            except Exception as e:
                print(f"Histogram calculation failed: {e}")
            
            # Store ORB features
            try:
                orb = cv2.ORB_create(nfeatures=100)
                if len(boat_image.shape) == 3:
                    gray = cv2.cvtColor(boat_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = boat_image
                kp, desc = orb.detectAndCompute(gray, None)
                features['orb_features'] = desc
            except Exception as e:
                print(f"ORB feature extraction failed: {e}")
            
            self.boat_features[boat_id] = features
            
        except Exception as e:
            print(f"Feature recording failed: {e}")
            # Store minimal features as fallback
            self.boat_features[boat_id] = {
                'bbox': bbox,
                'timestamp': current_time
            }
    
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
    
    def get_boat_stats(self, boat_id):
        """Get comprehensive stats for a boat"""
        return {
            'save_count': self.get_save_count(boat_id),
            'last_save_time': self.last_save_times.get(boat_id),
            'has_features': boat_id in self.boat_features,
            'last_bbox': self.boat_features.get(boat_id, {}).get('bbox')
        }
    
    def configure_thresholds(self, bbox_threshold=None, histogram_threshold=None, 
                           movement_threshold=None, feature_threshold=None,
                           min_criteria=None):
        """Configure detection thresholds and requirements"""
        if bbox_threshold is not None:
            self.bbox_change_threshold = bbox_threshold
        if histogram_threshold is not None:
            self.histogram_threshold = histogram_threshold  
        if movement_threshold is not None:
            self.movement_threshold = movement_threshold
        if feature_threshold is not None:
            self.feature_similarity_threshold = feature_threshold
        if min_criteria is not None:
            self.min_criteria_for_change = min_criteria
    
    def enable_debug_mode(self):
        """Enable detailed logging for debugging"""
        self.debug_mode = True
        
    def _debug_print(self, message):
        """Print debug message if debug mode is enabled"""
        if hasattr(self, 'debug_mode') and self.debug_mode:
            print(f"[BoatTracker DEBUG] {message}")