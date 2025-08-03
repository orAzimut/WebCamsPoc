import time
import os
import cv2
import numpy as np
import json
import re
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
from ultralytics import YOLO
import math
from collections import defaultdict

# Import required packages for SORT and CV methods
try:
    from skimage.metrics import structural_similarity as ssim
    from scipy.spatial.distance import cosine
    from filterpy.kalman import KalmanFilter
    import lap
except ImportError as e:
    print(f"⚠ Missing package: {e}")
    print("Please install: pip install scikit-image scipy filterpy lap")
    exit(1)

# Simple SORT Tracker Implementation
class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        try:
            from filterpy.kalman import KalmanFilter
        except ImportError:
            print("Error: filterpy not installed. Run: pip install filterpy")
            raise
            
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.convert_x_to_bbox(self.kf.x)

    @staticmethod
    def convert_bbox_to_z(bbox):
        """
        Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h    #scale is just area
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        """
        Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if(score==None):
            return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
        else:
            return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
    return(o)  


def associate_detections_to_trackers(detections, trackers, iou_threshold = 0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0], m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,5))

class BoatTracker:
    """Tracks individual boats and manages diversity scoring"""
    
    def __init__(self, boat_id, initial_bbox, initial_time):
        self.boat_id = boat_id
        self.last_saved_bbox = initial_bbox
        self.last_saved_time = initial_time
        self.last_detection_time = initial_time
        self.save_count = 0
        self.max_saves = 100  # Maximum images per boat
        
        # Diversity thresholds
        self.min_time_interval = 5  # seconds (reduced for better testing)
        self.min_position_change = 50  # pixels
        self.min_size_change_ratio = 0.15  # 15% change
        
    def calculate_diversity_score(self, new_bbox, current_time):
        """Calculate how different this detection is from the last saved one"""
        
        # Time score (0-1, higher = more time passed)
        time_diff = current_time - self.last_saved_time
        time_score = min(time_diff / (self.min_time_interval * 2), 1.0)
        
        # Position score (0-1, higher = more movement)
        old_center = self.get_bbox_center(self.last_saved_bbox)
        new_center = self.get_bbox_center(new_bbox)
        distance = math.sqrt((old_center[0] - new_center[0])**2 + (old_center[1] - new_center[1])**2)
        position_score = min(distance / (self.min_position_change * 2), 1.0)
        
        # Size score (0-1, higher = more size change)
        old_area = self.get_bbox_area(self.last_saved_bbox)
        new_area = self.get_bbox_area(new_bbox)
        size_ratio = abs(new_area - old_area) / old_area if old_area > 0 else 0
        size_score = min(size_ratio / (self.min_size_change_ratio * 2), 1.0)
        
        # Combined score with weights
        combined_score = (time_score * 0.4 + position_score * 0.4 + size_score * 0.2)
        
        return combined_score, {
            'time_score': time_score,
            'position_score': position_score, 
            'size_score': size_score,
            'time_diff': time_diff,
            'distance': distance,
            'size_ratio': size_ratio
        }

    def should_save_image(self, new_bbox, current_time):
        """Determine if we should save this detection"""
        
        # Always save the first detection of this boat
        if self.save_count == 0:
            return True, "First detection of this boat"
        
        # Hard constraints
        if current_time - self.last_saved_time < self.min_time_interval:
            return False, f"Time interval too short ({current_time - self.last_saved_time:.1f}s < {self.min_time_interval}s)"
            
        if self.save_count >= self.max_saves:
            return False, "Max saves reached"
        
        # Diversity scoring
        score, details = self.calculate_diversity_score(new_bbox, current_time)
        
        # Save if diversity score is high enough
        threshold = 0.3  # Adjust this to be more/less selective
        if score >= threshold:
            return True, f"Diversity score: {score:.2f} (threshold: {threshold})"
        else:
            return False, f"Diversity score too low: {score:.2f}"
    
    def update_after_save(self, bbox, current_time):
        """Update tracker after saving an image"""
        self.last_saved_bbox = bbox
        self.last_saved_time = current_time
        self.save_count += 1
    
    def update_detection(self, bbox, current_time):
        """Update tracker with new detection (without saving)"""
        self.last_detection_time = current_time
    
    @staticmethod
    def get_bbox_center(bbox):
        """Get center point of bounding box [x1, y1, x2, y2]"""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    @staticmethod
    def get_bbox_area(bbox):
        """Get area of bounding box"""
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

class YOLOBoatDetector:
    """Handles YOLO11 boat detection"""
    
    def __init__(self):
        self.model = None
        self.boat_classes = ['boat', 'ship']  # COCO class names for boats
        
    def initialize_model(self):
        """Initialize YOLO11 model"""
        try:
            print("🔥 Loading YOLO11 model...")
            # Load YOLOv11 model (will download automatically if not present)
            self.model = YOLO('yolo11n.pt')
            # Force GPU if available:
            import torch
            if torch.cuda.is_available():
                self.model.to('cuda')
                print("✓ Using GPU acceleration")
            else:
                print("⚠ Using CPU (slower)")  # Use nano version for speed, change to 'yolo11s.pt' or 'yolo11m.pt' for better accuracy
            print("✓ YOLO11 model loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to load YOLO11 model: {e}")
            print("Make sure you have ultralytics installed: pip install ultralytics")
            return False
    
    def detect_boats(self, image):
        """Detect boats in image and return bounding boxes"""
        try:
            if self.model is None:
                return []
            
            # Run inference
            results = self.model(image, verbose=False)
            
            boats = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class name
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id].lower()
                        confidence = float(box.conf[0])
                        
                        # Check if it's a boat/ship with good confidence
                        if any(boat_class in class_name for boat_class in self.boat_classes) and confidence > 0.3:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            boats.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                                'class': class_name
                            })
            
            return boats
            
        except Exception as e:
            print(f"✗ Error in boat detection: {e}")
            return []

class BoatTrackingManager:
    """Manages multiple boat trackers and ID assignment"""
    
    def __init__(self):
        self.active_trackers = {}
        self.next_boat_id = 1
        self.max_tracking_distance = 100  # Max pixels to consider same boat
        self.tracker_timeout = 30  # Remove tracker after 30 seconds of no detection
        
    def assign_detections_to_trackers(self, detections, current_time):
        """Assign detections to existing trackers or create new ones"""
        
        # Remove old trackers
        self.cleanup_old_trackers(current_time)
        
        assigned_detections = []
        
        for detection in detections:
            bbox = detection['bbox']
            center = BoatTracker.get_bbox_center(bbox)
            
            # Find closest existing tracker
            best_tracker_id = None
            min_distance = float('inf')
            
            for tracker_id, tracker in self.active_trackers.items():
                tracker_center = BoatTracker.get_bbox_center(tracker.last_saved_bbox)
                distance = math.sqrt((center[0] - tracker_center[0])**2 + (center[1] - tracker_center[1])**2)
                
                if distance < min_distance and distance < self.max_tracking_distance:
                    min_distance = distance
                    best_tracker_id = tracker_id
            
            if best_tracker_id is not None:
                # Assign to existing tracker
                assigned_detections.append({
                    'detection': detection,
                    'tracker_id': best_tracker_id,
                    'is_new_tracker': False
                })
                self.active_trackers[best_tracker_id].update_detection(bbox, current_time)
            else:
                # Create new tracker
                new_tracker_id = self.next_boat_id
                self.next_boat_id += 1
                
                self.active_trackers[new_tracker_id] = BoatTracker(new_tracker_id, bbox, current_time)
                assigned_detections.append({
                    'detection': detection,
                    'tracker_id': new_tracker_id,
                    'is_new_tracker': True
                })
        
        return assigned_detections
    
    def cleanup_old_trackers(self, current_time):
        """Remove trackers that haven't been updated recently"""
        to_remove = []
        for tracker_id, tracker in self.active_trackers.items():
            if current_time - tracker.last_detection_time > self.tracker_timeout:
                to_remove.append(tracker_id)
        
        for tracker_id in to_remove:
            print(f"🗑️ Removing inactive tracker {tracker_id}")
            del self.active_trackers[tracker_id]

class LocationExtractor:
    """Extracts camera location from YouTube video"""
    
    def __init__(self, driver):
        self.driver = driver
        self.location = "Unknown"
        
    def extract_location_from_youtube(self):
        """Extract location from YouTube video title, description, or URL"""
        try:
            location_keywords = {
                'hamburg': ['hamburg', 'hamburg harbor', 'hamburg port', 'elbe'],
                'new_york': ['new york', 'nyc', 'brooklyn', 'manhattan', 'hudson', 'east river'],
                'miami': ['miami', 'miami beach', 'biscayne', 'port miami'],
                'london': ['london', 'thames', 'london bridge', 'tower bridge'],
                'singapore': ['singapore', 'marina bay', 'singapore strait'],
                'sydney': ['sydney', 'sydney harbor', 'sydney harbour', 'circular quay'],
                'san_francisco': ['san francisco', 'golden gate', 'sf bay', 'alcatraz'],
                'vancouver': ['vancouver', 'vancouver harbor', 'burrard inlet'],
                'amsterdam': ['amsterdam', 'amsterdam harbor', 'ij river'],
                'venice': ['venice', 'grand canal', 'san marco', 'rialto'],
                'monaco': ['monaco', 'monte carlo', 'port hercule'],
                'oslo': ['oslo', 'oslo harbor', 'oslofjord'],
                'stockholm': ['stockholm', 'stockholm harbor', 'gamla stan'],
                'copenhagen': ['copenhagen', 'nyhavn', 'copenhagen harbor'],
                'barcelona': ['barcelona', 'port vell', 'barcelona port'],
                'marseille': ['marseille', 'old port', 'vieux port'],
                'naples': ['naples', 'napoli', 'bay of naples'],
                'istanbul': ['istanbul', 'bosphorus', 'golden horn'],
                'seattle': ['seattle', 'puget sound', 'elliott bay'],
                'boston': ['boston', 'boston harbor', 'charlestown'],
                'chicago': ['chicago', 'lake michigan', 'navy pier'],
                'toronto': ['toronto', 'toronto harbor', 'lake ontario'],
                'hong_kong': ['hong kong', 'victoria harbor', 'tsim sha tsui'],
                'tokyo': ['tokyo', 'tokyo bay', 'odaiba'],
                'rio': ['rio', 'rio de janeiro', 'guanabara bay', 'copacabana'],
                'dubai': ['dubai', 'dubai marina', 'palm jumeirah'],
                'lisbon': ['lisbon', 'tagus', 'belem'],
                'gibraltar': ['gibraltar', 'strait of gibraltar'],
                'suez': ['suez', 'suez canal'],
                'panama': ['panama', 'panama canal'],
                'bosphorus': ['bosphorus', 'istanbul strait']
            }
            
            # Get video title
            try:
                title_element = self.driver.find_element(By.CSS_SELECTOR, "h1.ytd-video-primary-info-renderer")
                title = title_element.text.lower()
                print(f"📍 Video title: {title}")
            except:
                title = ""
            
            # Get video description
            try:
                # Click show more to expand description
                try:
                    show_more = self.driver.find_element(By.CSS_SELECTOR, "#expand")
                    if show_more.is_displayed():
                        show_more.click()
                        time.sleep(1)
                except:
                    pass
                
                description_element = self.driver.find_element(By.CSS_SELECTOR, "#description-text")
                description = description_element.text.lower()
                print(f"📍 Description preview: {description[:200]}...")
            except:
                description = ""
            
            # Get URL
            current_url = self.driver.current_url.lower()
            
            # Combine all text for location detection
            combined_text = f"{title} {description} {current_url}"
            
            # Check for location keywords
            detected_locations = []
            for location, keywords in location_keywords.items():
                for keyword in keywords:
                    if keyword in combined_text:
                        detected_locations.append((location, keyword))
                        break
            
            if detected_locations:
                # Use the first detected location
                self.location = detected_locations[0][0]
                detected_keyword = detected_locations[0][1]
                print(f"✓ Location detected: {self.location} (keyword: '{detected_keyword}')")
                
                if len(detected_locations) > 1:
                    other_locations = [loc[0] for loc in detected_locations[1:]]
                    print(f"  Other possible locations: {other_locations}")
            else:
                print("⚠ Could not detect location from video content")
                self.location = "Unknown"
            
            return self.location
            
        except Exception as e:
            print(f"✗ Error extracting location: {e}")
            self.location = "Unknown"
            return self.location
class BoatAngleDetector:
    """Detects if boat angle/appearance has changed significantly using CV methods"""
    
    def __init__(self):
        self.similarity_threshold = 0.9  # SSIM threshold for "same angle"
        self.histogram_threshold = 0.8   # Histogram correlation threshold
        
    def extract_boat_region(self, frame, bbox):
        """Extract boat region from frame using bounding box"""
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), int(x2), int(y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        boat_region = frame[y1:y2, x1:x2]
        
        # Resize to standard size for comparison
        if boat_region.size > 0:
            boat_region = cv2.resize(boat_region, (64, 64))
            return boat_region
        return None
    
    def calculate_ssim_similarity(self, img1, img2):
        """Calculate structural similarity between two images"""
        try:
            # Check if SSIM is available
            from skimage.metrics import structural_similarity as ssim
            
            # Convert to grayscale if needed
            if len(img1.shape) == 3:
                img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else:
                img1_gray, img2_gray = img1, img2
                
            # Calculate SSIM
            similarity = ssim(img1_gray, img2_gray)
            return similarity
        except ImportError:
            # SSIM not available, return 0 to skip this metric
            return 0.0
        except Exception as e:
            print(f"SSIM calculation error: {e}")
            return 0.0
    
    def calculate_histogram_similarity(self, img1, img2):
        """Calculate histogram correlation between two images"""
        try:
            # Calculate histograms for each channel
            hist1 = cv2.calcHist([img1], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
            hist2 = cv2.calcHist([img2], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
            
            # Normalize histograms
            cv2.normalize(hist1, hist1)
            cv2.normalize(hist2, hist2)
            
            # Calculate correlation
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return correlation
        except Exception as e:
            print(f"Histogram calculation error: {e}")
            return 0.0
    
    def is_different_angle(self, current_frame, current_bbox, reference_frame, reference_bbox):
        """
        Determine if current detection shows boat at significantly different angle
        Returns: (is_different, similarity_scores)
        """
        try:
            # Extract boat regions
            current_boat = self.extract_boat_region(current_frame, current_bbox)
            reference_boat = self.extract_boat_region(reference_frame, reference_bbox)
            
            if current_boat is None or reference_boat is None:
                return True, {"ssim": 0.0, "histogram": 0.0}  # Assume different if extraction fails
            
            # Calculate similarities
            ssim_score = self.calculate_ssim_similarity(current_boat, reference_boat)
            hist_score = self.calculate_histogram_similarity(current_boat, reference_boat)
            
            # Determine if angle is different (either metric below threshold)
            is_different = (ssim_score < self.similarity_threshold or 
                          hist_score < self.histogram_threshold)
            
            return is_different, {
                "ssim": ssim_score,
                "histogram": hist_score
            }
            
        except Exception as e:
            print(f"Angle detection error: {e}")
            return True, {"ssim": 0.0, "histogram": 0.0}
# Replace the SORTBoatTracker class with this improved version:

class SORTBoatTracker:
    """SORT-based boat tracking with intelligent ID reuse"""
    
    def __init__(self):
        self.sort_tracker = Sort(max_age=15, min_hits=1, iou_threshold=0.3)  # Increased max_age
        self.angle_detector = BoatAngleDetector()
        self.boat_data = {}  # {track_id: boat_info}
        self.recently_lost_boats = {}  # {track_id: {'last_bbox', 'lost_time', 'boat_data'}}
        self.last_save_time = 0
        self.save_interval = 5.0  # 5 seconds minimum between any saves
        self.reuse_timeout = 30.0  # 30 seconds to reuse lost IDs
        self.reuse_distance = 150  # Max pixels to consider for ID reuse
        
    def update_tracks(self, detections):
        """Update SORT tracker with new detections and handle ID reuse"""
        if not detections:
            tracks = self.sort_tracker.update(np.empty((0, 5)))
            # Clean up old lost boats
            self.cleanup_lost_boats()
            return []
        
        # Convert detections to SORT format: [x1, y1, x2, y2, confidence]
        sort_detections = []
        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']
            sort_detections.append([bbox[0], bbox[1], bbox[2], bbox[3], conf])
        
        sort_detections = np.array(sort_detections)
        
        # Update SORT tracker
        tracks = self.sort_tracker.update(sort_detections)
        
        # Check for new tracks that might be reappearing boats
        current_track_ids = set()
        result_tracks = []
        
        for i, track in enumerate(tracks):
            original_track_id = int(track[4])  # SORT's assigned ID
            bbox = [int(track[0]), int(track[1]), int(track[2]), int(track[3])]
            
            # Check if this is a "new" track that might be a reappearing boat
            reused_id = self.check_for_id_reuse(original_track_id, bbox)
            final_track_id = reused_id if reused_id else original_track_id
            
            current_track_ids.add(final_track_id)
            
            # Find matching detection by bbox similarity
            best_detection = None
            best_iou = 0
            for det in detections:
                iou = self.calculate_iou(bbox, det['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_detection = det
            
            if best_detection:
                result_tracks.append({
                    'track_id': final_track_id,
                    'original_sort_id': original_track_id,
                    'bbox': bbox,
                    'confidence': best_detection['confidence'],
                    'class': best_detection['class'],
                    'is_new': final_track_id not in self.boat_data,
                    'is_reused': reused_id is not None
                })
        
        # Update lost boats tracking
        self.update_lost_boats_tracking(current_track_ids)
        
        return result_tracks
    
    def check_for_id_reuse(self, new_track_id, new_bbox):
        """Check if this new track should reuse a recently lost ID"""
        import time
        current_time = time.time()
        
        # Only check for truly new tracks (not in our boat_data)
        if new_track_id in self.boat_data:
            return None
        
        # Check recently lost boats
        for lost_id, lost_info in list(self.recently_lost_boats.items()):
            time_since_lost = current_time - lost_info['lost_time']
            
            # Skip if too much time has passed
            if time_since_lost > self.reuse_timeout:
                continue
            
            # Calculate distance between new detection and last known position
            distance = self.calculate_bbox_distance(new_bbox, lost_info['last_bbox'])
            
            if distance < self.reuse_distance:
                print(f"🔄 Reusing ID {lost_id} for new detection (distance: {distance:.1f}px, lost {time_since_lost:.1f}s ago)")
                
                # Restore the boat data
                self.boat_data[lost_id] = lost_info['boat_data']
                
                # Remove from lost boats
                del self.recently_lost_boats[lost_id]
                
                return lost_id
        
        return None
    
    def calculate_bbox_distance(self, bbox1, bbox2):
        """Calculate distance between centers of two bounding boxes"""
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        
        return ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
    
    def update_lost_boats_tracking(self, current_track_ids):
        """Update tracking of lost boats"""
        import time
        current_time = time.time()
        
        # Move boats that are no longer tracked to recently_lost
        for track_id in list(self.boat_data.keys()):
            if track_id not in current_track_ids:
                if track_id not in self.recently_lost_boats:
                    print(f"📤 Boat {track_id} lost - keeping ID available for reuse")
                    self.recently_lost_boats[track_id] = {
                        'last_bbox': self.boat_data[track_id]['last_saved_bbox'],
                        'lost_time': current_time,
                        'boat_data': self.boat_data[track_id].copy()
                    }
                    del self.boat_data[track_id]
        
        # Clean up old lost boats
        self.cleanup_lost_boats()
    
    def cleanup_lost_boats(self):
        """Remove boats that have been lost for too long"""
        import time
        current_time = time.time()
        
        to_remove = []
        for lost_id, lost_info in self.recently_lost_boats.items():
            if current_time - lost_info['lost_time'] > self.reuse_timeout:
                to_remove.append(lost_id)
        
        for lost_id in to_remove:
            print(f"🗑️ Permanently removing boat {lost_id} (lost for {self.reuse_timeout}s)")
            del self.recently_lost_boats[lost_id]
    
    def calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    def get_boat_size_category(self, bbox):
        """Categorize boat by bounding box area"""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        
        if area <= 32**2:  # <= 1,024 pixels
            return "small", area, 5.0  # 5 second interval
        elif area <= 96**2:  # <= 9,216 pixels  
            return "medium", area, 10.0  # 10 second interval
        else:  # > 9,216 pixels
            return "large", area, 10.0  # 10 second interval
    def should_save_boat(self, track_id, current_frame, current_bbox, current_time):
        """Determine if we should save this boat detection with smart size-based intervals"""
        
        # Get size-based save interval
        size_category, area, required_interval = self.get_boat_size_category(current_bbox)
        
        # Global time constraint - use size-based interval
        if current_time - self.last_save_time < self.save_interval:
            time_left = self.save_interval - (current_time - self.last_save_time)
            return False, f"Global save interval: {time_left:.1f}s remaining", False
        
        # Check if this is a new track ID
        if track_id not in self.boat_data:
            return True, f"New {size_category} boat ID detected (area: {area}px)", True
        
        boat_info = self.boat_data[track_id]
        
        # Size-specific time constraint for this individual boat
        time_since_last_save = current_time - boat_info['last_saved_time']
        if time_since_last_save < required_interval:
            time_left = required_interval - time_since_last_save
            return False, f"{size_category.title()} boat interval: {time_left:.1f}s remaining (need {required_interval}s)", False
        
        # Check if boat has different angle compared to last saved
        is_different_angle, similarity_scores = self.angle_detector.is_different_angle(
            current_frame, current_bbox,
            boat_info['last_saved_frame'], boat_info['last_saved_bbox']
        )
        
        if is_different_angle:
            if 'ssim' in similarity_scores:
                reason = f"Different angle detected for {size_category} boat (SSIM: {similarity_scores['ssim']:.2f}, Hist: {similarity_scores['histogram']:.2f})"
            else:
                reason = f"Different angle detected for {size_category} boat (Hist: {similarity_scores['histogram']:.2f})"
            return True, reason, False
        else:
            if 'ssim' in similarity_scores:
                reason = f"Same angle for {size_category} boat (SSIM: {similarity_scores['ssim']:.2f}, Hist: {similarity_scores['histogram']:.2f})"
            else:
                reason = f"Same angle for {size_category} boat (Hist: {similarity_scores['histogram']:.2f})"
            return False, reason, False
    
    def update_boat_data(self, track_id, frame, bbox, current_time):
        """Update boat data after saving"""
        self.boat_data[track_id] = {
            'last_saved_frame': frame.copy(),
            'last_saved_bbox': bbox.copy(),
            'last_saved_time': current_time,
            'save_count': self.boat_data.get(track_id, {}).get('save_count', 0) + 1
        }
        self.last_save_time = current_time
    
    def get_boat_stats(self):
        """Get statistics about tracked boats"""
        stats = {}
        for track_id, data in self.boat_data.items():
            stats[track_id] = data.get('save_count', 0)
        return stats
    
    def get_full_stats(self):
        """Get comprehensive statistics including lost boats"""
        return {
            'active_boats': self.get_boat_stats(),
            'recently_lost_count': len(self.recently_lost_boats),
            'lost_boat_ids': list(self.recently_lost_boats.keys())
        }

    # Also update the process_frame method to handle reused IDs:

    def process_frame(self, current_time, frame_count):
        """Process single frame for boat detection using SORT tracking with ID reuse"""
        try:
            # Ensure video is playing before processing
            if not self.ensure_video_playing():
                print("⚠ Video not playing, but continuing...")
            
            # Less frequent mouse movement to avoid interfering with video
            if frame_count % 150 == 0:  # Every 150 frames (30 seconds at 0.2s intervals)
                self.move_mouse_away_from_video()
            
            # Capture frame
            frame = self.get_video_frame()
            if frame is None:
                return False
            
            # Detect boats using YOLO
            detections = self.yolo_detector.detect_boats(frame)
            
            if detections:
                print(f"🚢 Detected {len(detections)} boat(s)")
                self.stats['boats_detected'] += len(detections)
                
                # Update SORT tracker with detections (includes ID reuse logic)
                tracks = self.sort_tracker.update_tracks(detections)
                
                if tracks:
                    print(f"📊 Tracking {len(tracks)} boat(s)")
                    
                    # Check if any boat should trigger a save
                    save_triggered = False
                    trigger_boat_id = None
                    save_reason = ""
                    
                    # Process each tracked boat to find if any should trigger a save
                    for track in tracks:
                        track_id = track['track_id']
                        is_new = track['is_new']
                        is_reused = track.get('is_reused', False)
                        bbox = track['bbox']
                        
                        # Check if we should save this detection
                        should_save, reason, is_new_id = self.sort_tracker.should_save_boat(
                            track_id, frame, bbox, current_time
                        )
                        
                        # Enhanced status display
                        if is_reused:
                            status = "REUSED ID"
                        elif is_new:
                            status = "NEW ID"
                        else:
                            status = "EXISTING"
                        
                        print(f"  🚢 Boat {track_id} ({status}): {reason}")
                        
                        if should_save and not save_triggered:
                            save_triggered = True
                            trigger_boat_id = track_id
                            save_reason = reason
                            
                            # Update the boat that triggered the save
                            self.sort_tracker.update_boat_data(track_id, frame, bbox, current_time)
                            
                            # Update statistics
                            if is_new_id:
                                self.stats['new_ids_detected'] += 1
                            else:
                                self.stats['angle_changes_detected'] += 1
                    
                    # If any boat triggered a save, save the frame with ALL detections
                    if save_triggered:
                        if self.save_frame_with_all_boats(frame, tracks, trigger_boat_id, save_reason, current_time):
                            for track in tracks:
                                self.stats['active_boat_folders'].add(track['track_id'])
            
            self.stats['frames_processed'] += 1
            return True
            
        except Exception as e:
            print(f"✗ Error processing frame: {e}")
            return False
    
class IntelligentYouTubeBoatScraper:
    """Main scraper class with intelligent boat detection"""
    
    def __init__(self, youtube_url):
        self.url = youtube_url
        self.base_dir = r"C:\Users\OrGil.AzureAD\OneDrive - AMPC\Desktop\Azimut.ai\webScrape\webCams\POC"
        
        # Keep run timestamp for metadata but don't use for folder structure
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.base_dir  # Output dir is now the base POC folder
        
        self.driver = None
        self.video_element = None
        self.location_extractor = None
        self.camera_location = "Unknown"
        
        # Initialize detection components
        self.yolo_detector = YOLOBoatDetector()
        self.sort_tracker = SORTBoatTracker()  # Replace old tracking system
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'boats_detected': 0,
            'images_saved': 0,
            'active_boat_folders': set(),
            'target_frames': 0,  # Will be set based on duration
            'new_ids_detected': 0,
            'angle_changes_detected': 0
        }
        
        # Ensure base directory exists
        os.makedirs(self.base_dir, exist_ok=True)
        print(f"✓ Base output directory: {self.base_dir}")
        print(f"✓ Files will be organized by: YEAR/MONTH/DAY/HOUR/")
        print(f"✓ Using SORT tracking with 0.2s inference interval")
        print(f"✓ Saving conditions: 5s interval + (new ID OR angle change)")
        print(f"✓ YouTube URL: {self.url}")
    
    def setup_driver(self):
        """Setup Chrome driver optimized for YouTube"""
        try:
            print("Setting up Chrome driver for YouTube...")
            
            chrome_options = Options()
            
            # Essential options for YouTube
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Set good window size for video
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--start-maximized")
            
            # Performance options
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-plugins")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--no-sandbox")
            
            # YouTube specific options
            chrome_options.add_argument("--disable-web-security")
            chrome_options.add_argument("--allow-running-insecure-content")
            
            # Autoplay and media options
            chrome_options.add_argument("--autoplay-policy=no-user-gesture-required")
            
            # Set media permissions
            prefs = {
                "profile.default_content_setting_values": {
                    "media_stream": 1,
                    "media_stream_mic": 1,
                    "media_stream_camera": 1
                }
            }
            chrome_options.add_experimental_option("prefs", prefs)
            
            # Auto-download and setup ChromeDriver
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            # Hide automation indicators
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            print("✓ Chrome driver initialized successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to initialize Chrome driver: {e}")
            return False
    
    def load_youtube_video(self):
        """Load YouTube video and prepare for detection"""
        try:
            print(f"Loading YouTube video: {self.url}")
            self.driver.get(self.url)
            
            # Wait for video player to load
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "video.html5-main-video"))
            )
            
            time.sleep(5)
            
            # Initialize location extractor and extract location
            self.location_extractor = LocationExtractor(self.driver)
            self.camera_location = self.location_extractor.extract_location_from_youtube()
            
            # Dismiss popups/notifications
            self.dismiss_youtube_popups()
            
            # Find and setup video
            video_element = self.driver.find_element(By.CSS_SELECTOR, "video.html5-main-video")
            self.video_element = video_element
            
            # Start video and set to theater mode
            self.setup_video_playback()
            
            # Move mouse away from video to avoid YouTube controls
            self.move_mouse_away_from_video()
            
            print("✓ YouTube video loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load YouTube video: {e}")
            return False
    
    def move_mouse_away_from_video(self):
        """Move mouse away from video to avoid YouTube timeline/controls"""
        try:
            # Move mouse to a safe area (sidebar or top of page)
            action = ActionChains(self.driver)
            action.move_to_element_with_offset(self.driver.find_element(By.TAG_NAME, "body"), 50, 50)
            action.perform()
            
        except Exception as e:
            print(f"⚠ Could not move mouse away from video: {e}")
    
    def dismiss_youtube_popups(self):
        """Dismiss various YouTube popups"""
        try:
            consent_selectors = [
                "[aria-label*='Accept']",
                "[aria-label*='I agree']", 
                "button[aria-label*='Accept all']",
                ".VfPpkd-LgbsSe[aria-label*='Accept']"
            ]
            
            for selector in consent_selectors:
                try:
                    consent_button = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if consent_button and consent_button.is_displayed():
                        consent_button.click()
                        print("✓ Dismissed consent dialog")
                        time.sleep(2)
                        break
                except:
                    continue
        except:
            pass
    
    def setup_video_playback(self):
        """Setup video playback and theater mode"""
        try:
            # Click on video to focus
            self.video_element.click()
            time.sleep(2)
            
            # Ensure video is playing
            is_paused = self.driver.execute_script("return document.querySelector('video.html5-main-video').paused;")
            if is_paused:
                print("🎬 Starting video playback...")
                self.video_element.send_keys(' ')  # Spacebar to play
                time.sleep(2)
            
            # Try theater mode
            self.video_element.send_keys('t')
            time.sleep(2)
            
            print("✓ Video setup complete")
            
        except Exception as e:
            print(f"Video setup warning: {e}")
    
    def ensure_video_playing(self):
        """Ensure video is playing before taking screenshot"""
        try:
            # Check if video is paused using JavaScript
            is_paused = self.driver.execute_script("return document.querySelector('video.html5-main-video').paused;")
            
            if is_paused:
                print("⏸ Video paused - resuming...")
                
                # Method 1: JavaScript play
                try:
                    self.driver.execute_script("document.querySelector('video.html5-main-video').play();")
                    time.sleep(1)
                    
                    # Verify it worked
                    still_paused = self.driver.execute_script("return document.querySelector('video.html5-main-video').paused;")
                    if not still_paused:
                        print("✓ Video resumed via JavaScript")
                        return True
                except:
                    pass
                
                # Method 2: Click play button
                try:
                    play_button = self.driver.find_element(By.CSS_SELECTOR, "button.ytp-play-button")
                    if play_button.is_displayed():
                        play_button.click()
                        time.sleep(1)
                        print("✓ Video resumed via play button")
                        return True
                except:
                    pass
                
                # Method 3: Click video and spacebar
                try:
                    self.video_element.click()
                    time.sleep(0.5)
                    self.video_element.send_keys(' ')
                    time.sleep(1)
                    print("✓ Video resumed via spacebar")
                    return True
                except:
                    pass
                
                print("⚠ Could not resume video")
                return False
            else:
                return True  # Video is already playing
                
        except Exception as e:
            print(f"✗ Error checking video status: {e}")
            return False
    
    def keep_video_active(self):
        """Keep video active without interfering with playback"""
        try:
            # Very gentle interaction - just move mouse slightly near video
            video_location = self.video_element.location
            video_size = self.video_element.size
            
            # Move mouse to a corner of the video (not center to avoid controls)
            action = ActionChains(self.driver)
            action.move_to_element_with_offset(self.video_element, 10, 10)  # Top-left corner
            action.perform()
            
            # Don't click, just hover briefly then move away
            time.sleep(0.1)
            
            # Move mouse away from video completely
            action = ActionChains(self.driver)
            action.move_by_offset(200, -50)  # Move away from video
            action.perform()
            
        except Exception as e:
            print(f"⚠ Could not keep video active: {e}")
    
    def get_video_frame(self):
        """Capture current video frame as OpenCV image"""
        try:
            # Take screenshot of video element
            video_png = self.video_element.screenshot_as_png
            
            # Convert to OpenCV format
            nparr = np.frombuffer(video_png, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return frame
            
        except Exception as e:
            print(f"✗ Error capturing video frame: {e}")
            return None
    
    def create_datetime_folder(self):
        """Create hierarchical folder structure based on current date/time"""
        now = datetime.now()
        
        # Create year/month/day/hour structure
        year = now.strftime("%Y")
        month = now.strftime("%m") 
        day = now.strftime("%d")
        hour = now.strftime("%H")
        
        datetime_folder = os.path.join(self.base_dir, year, month, day, hour)
        os.makedirs(datetime_folder, exist_ok=True)
        
        return datetime_folder
    
    def save_boat_image(self, frame, boat_detection, boat_id, current_time):
        """Save boat image with JSON metadata in date/time organized folders"""
        try:
            # Create datetime-based folder structure
            datetime_folder = self.create_datetime_folder()
            
            # Create filename (same for both image and JSON)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            base_filename = f"boat_{boat_id:03d}_{timestamp}_{self.stats['images_saved']:04d}"
            
            # Save image
            image_filepath = os.path.join(datetime_folder, f"{base_filename}.jpg")
            cv2.imwrite(image_filepath, frame)
            
            # Create JSON metadata
            bbox = boat_detection['bbox']
            metadata = {
                "timestamp": timestamp,
                "youtube_url": self.url,
                "camera_location": self.camera_location,
                "boat_id": boat_id,
                "confidence": round(boat_detection['confidence'], 3),
                "class": boat_detection['class'],
                "bbox": {
                    "x1": bbox[0],
                    "y1": bbox[1], 
                    "x2": bbox[2],
                    "y2": bbox[3],
                    "width": bbox[2] - bbox[0],
                    "height": bbox[3] - bbox[1]
                },
                "frame_info": {
                    "frame_width": frame.shape[1],
                    "frame_height": frame.shape[0],
                    "total_frames_processed": self.stats['frames_processed'],
                    "images_saved_this_run": self.stats['images_saved'] + 1
                },
                "run_info": {
                    "run_timestamp": self.run_timestamp,
                    "output_directory": self.output_dir
                },
                "datetime_path": {
                    "year": datetime.now().strftime("%Y"),
                    "month": datetime.now().strftime("%m"),
                    "day": datetime.now().strftime("%d"), 
                    "hour": datetime.now().strftime("%H"),
                    "full_path": datetime_folder
                }
            }
            
            # Save JSON metadata  
            json_filepath = os.path.join(datetime_folder, f"{base_filename}.json")
            with open(json_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Track this boat ID for statistics
            self.stats['active_boat_folders'].add(boat_id)
            self.stats['images_saved'] += 1
            
            print(f"💾 Saved boat {boat_id} image: {base_filename}.jpg + .json")
            print(f"   📁 Path: {datetime_folder}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error saving boat image: {e}")
            return False
        
    def process_frame(self, current_time, frame_count):
        """Process single frame for boat detection using SORT tracking"""
        try:
            # Ensure video is playing before processing
            if not self.ensure_video_playing():
                print("⚠ Video not playing, but continuing...")
            
            # Less frequent mouse movement to avoid interfering with video
            if frame_count % 150 == 0:  # Every 150 frames (30 seconds at 0.2s intervals)
                self.move_mouse_away_from_video()
            
            # Capture frame
            frame = self.get_video_frame()
            if frame is None:
                return False
            
            # Detect boats using YOLO
            detections = self.yolo_detector.detect_boats(frame)
            
            if detections:
                print(f"🚢 Detected {len(detections)} boat(s)")
                self.stats['boats_detected'] += len(detections)
                
                # Update SORT tracker with detections
                tracks = self.sort_tracker.update_tracks(detections)
                
                if tracks:
                    print(f"📊 Tracking {len(tracks)} boat(s)")
                    
                    # Check if any boat should trigger a save
                    save_triggered = False
                    trigger_boat_id = None
                    save_reason = ""
                    
                    # Process each tracked boat to find if any should trigger a save
                    for track in tracks:
                        track_id = track['track_id']
                        is_new = track['is_new']
                        bbox = track['bbox']
                        
                        # Check if we should save this detection
                        should_save, reason, is_new_id = self.sort_tracker.should_save_boat(
                            track_id, frame, bbox, current_time
                        )
                        
                        status = "NEW ID" if is_new else "EXISTING"
                        print(f"  🚢 Boat {track_id} ({status}): {reason}")
                        
                        if should_save and not save_triggered:
                            save_triggered = True
                            trigger_boat_id = track_id
                            save_reason = reason
                            
                            # Update the boat that triggered the save
                            self.sort_tracker.update_boat_data(track_id, frame, bbox, current_time)
                            
                            # Update statistics
                            if is_new_id:
                                self.stats['new_ids_detected'] += 1
                            else:
                                self.stats['angle_changes_detected'] += 1
                    
                    # If any boat triggered a save, save the frame with ALL detections
                    if save_triggered:
                        if self.save_frame_with_all_boats(frame, tracks, trigger_boat_id, save_reason, current_time):
                            for track in tracks:
                                self.stats['active_boat_folders'].add(track['track_id'])
            
            self.stats['frames_processed'] += 1
            return True
            
        except Exception as e:
            print(f"✗ Error processing frame: {e}")
            return False
    def save_frame_with_all_boats(self, frame, all_tracks, trigger_boat_id, save_reason, current_time):
        """Save frame with JSON containing all detected boats with size categories"""
        try:
            # Create datetime-based folder structure
            datetime_folder = self.create_datetime_folder()
            
            # Create simple, readable filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            base_filename = f"frame_{timestamp}"
            
            # Save image
            image_filepath = os.path.join(datetime_folder, f"{base_filename}.jpg")
            cv2.imwrite(image_filepath, frame)
            
            # Create simplified JSON metadata with all boats including size info
            all_boats_data = []
            for track in all_tracks:
                bbox = track['bbox']
                size_category, area, interval = self.sort_tracker.get_boat_size_category(bbox)

            
                boat_data = {
                    "boat_id": track['track_id'],
                    "track_id": track['track_id'], 
                    "confidence": round(track['confidence'], 3),
                    "class": track['class'],
                    "size_category": size_category,
                    "bbox_area": area,
                    "save_interval": interval,
                    "bbox": {
                        "x1": bbox[0],
                        "y1": bbox[1], 
                        "x2": bbox[2],
                        "y2": bbox[3],
                        "width": bbox[2] - bbox[0],
                        "height": bbox[3] - bbox[1]
                    }
                }
                all_boats_data.append(boat_data)
            
            # Find trigger boat size info
            trigger_boat_data = next((boat for boat in all_boats_data if boat['boat_id'] == trigger_boat_id), None)
            trigger_size_category = trigger_boat_data['size_category'] if trigger_boat_data else "unknown"
            
            # Main metadata structure
            metadata = {
                "timestamp": timestamp,
                "youtube_url": self.url,
                "camera_location": self.camera_location,
                "trigger_boat_id": trigger_boat_id,
                "trigger_boat_size": trigger_size_category,
                "save_reason": save_reason,
                "tracking_method": "SORT",
                "total_boats_detected": len(all_tracks),
                "boats": all_boats_data,  # All boats in this frame with size info
                "frame_info": {
                    "frame_width": frame.shape[1],
                    "frame_height": frame.shape[0]
                },
                "size_distribution": {
                    "small": len([b for b in all_boats_data if b['size_category'] == 'small']),
                    "medium": len([b for b in all_boats_data if b['size_category'] == 'medium']), 
                    "large": len([b for b in all_boats_data if b['size_category'] == 'large'])
                }
            }
            
            # Save JSON metadata  
            json_filepath = os.path.join(datetime_folder, f"{base_filename}.json")
            with open(json_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.stats['images_saved'] += 1
            
            print(f"💾 Saved frame triggered by {trigger_size_category} boat {trigger_boat_id}: {base_filename}.jpg + .json")
            print(f"   📁 Path: {datetime_folder}")
            print(f"   🚢 Contains {len(all_tracks)} boat(s): {[f"{t['track_id']}({self.sort_tracker.get_boat_size_category(t['bbox'])[0]})" for t in all_tracks]}")

            
            return True
            
        except Exception as e:
            print(f"✗ Error saving frame: {e}")
            return False

    # Update the statistics display to show size information:

    def print_statistics_for_saved_images_mode(self, successful_frames, target_saved_images):
        """Print statistics for saved images mode with size information"""
        print(f"\n📊 STATISTICS:")
        print(f"  Frames processed: {successful_frames} (0.2s intervals)")
        print(f"  Saved images: {self.stats['images_saved']}/{target_saved_images}")
        if target_saved_images > 0:
            progress = (self.stats['images_saved'] / target_saved_images) * 100
            print(f"  Progress: {progress:.1f}%")
        print(f"  Total boat detections: {self.stats['boats_detected']}")
        print(f"  New IDs detected: {self.stats['new_ids_detected']}")
        print(f"  Angle changes detected: {self.stats['angle_changes_detected']}")
        print(f"  Active boat IDs: {len(self.stats['active_boat_folders'])}")
        print(f"  Camera location: {self.camera_location}")
        print(f"  Save intervals: Small boats=5s, Medium/Large boats=10s")
        
        # Show current SORT tracker stats with size info
        boat_stats = self.sort_tracker.get_boat_stats()
        if boat_stats:
            print(f"  Currently tracked boats: {sorted(list(boat_stats.keys()))}")
            print(f"  Saves per boat: {dict(sorted(boat_stats.items()))}")
            
            # Show size distribution of currently tracked boats
            size_counts = {"small": 0, "medium": 0, "large": 0}
            for track_id in boat_stats.keys():
                if track_id in self.sort_tracker.boat_data:
                    last_bbox = self.sort_tracker.boat_data[track_id]['last_saved_bbox']
                    size_cat, _, _ = self.sort_tracker.get_boat_size_category(last_bbox)
                    size_counts[size_cat] += 1
            
            if any(size_counts.values()):
                print(f"  Boat sizes: Small={size_counts['small']}, Medium={size_counts['medium']}, Large={size_counts['large']}")
    def print_statistics(self, successful_frames=None):
        """Print current statistics"""
        print(f"\n📊 STATISTICS:")
        if successful_frames is not None:
            print(f"  Successful frames: {successful_frames}/{self.stats['target_frames']} (0.2s intervals)")
            if self.stats['target_frames'] > 0:
                progress = (successful_frames / self.stats['target_frames']) * 100
                print(f"  Progress: {progress:.1f}%")
        else:
            print(f"  Frames processed: {self.stats['frames_processed']}/{self.stats['target_frames']}")
            if self.stats['target_frames'] > 0:
                progress = (self.stats['frames_processed'] / self.stats['target_frames']) * 100
                print(f"  Progress: {progress:.1f}%")
        print(f"  Total boat detections: {self.stats['boats_detected']}")
        print(f"  Images saved: {self.stats['images_saved']}")
        print(f"  New IDs detected: {self.stats['new_ids_detected']}")
        print(f"  Angle changes detected: {self.stats['angle_changes_detected']}")
        print(f"  Active boat IDs: {len(self.stats['active_boat_folders'])}")
        print(f"  Camera location: {self.camera_location}")
        
        # Show SORT tracker stats
        boat_stats = self.sort_tracker.get_boat_stats()
        if boat_stats:
            print(f"  Tracked boats: {sorted(list(boat_stats.keys()))}")
            print(f"  Saves per boat: {dict(sorted(boat_stats.items()))}")
    
    def run_intelligent_scraping(self, duration_minutes=None, max_frames=None, check_interval=0.2):
        """Main intelligent scraping loop - can limit by time OR saved boat images"""
        
        if duration_minutes is not None and max_frames is not None:
            raise ValueError("Cannot specify both duration_minutes and max_frames")
        if duration_minutes is None and max_frames is None:
            raise ValueError("Must specify either duration_minutes or max_frames")
        
        print(f"\n=== STARTING INTELLIGENT BOAT DETECTION WITH SORT ===")
        print(f"Inference interval: {check_interval} seconds (5x faster than before)")
        print(f"Camera location: {self.camera_location}")
        
        # Determine run mode and target
        if max_frames is not None:
            # Frame-based mode (now means max saved boat images)
            target_saved_images = max_frames
            estimated_frames = target_saved_images * 25  # Rough estimate: 25 frames per save
            estimated_time = (estimated_frames * check_interval) / 60
            print(f"Mode: Saved images limit")
            print(f"Target saved boat images: {target_saved_images}")
            print(f"Estimated frames to process: ~{estimated_frames}")
            print(f"Estimated time: ~{estimated_time:.1f} minutes")
            run_by_saved_images = True
            end_time = None
        else:
            # Time-based mode  
            self.stats['target_frames'] = int((duration_minutes * 60) / check_interval)
            print(f"Mode: Time-based limit")
            print(f"Duration: {duration_minutes} minutes")
            print(f"Target frames: ~{self.stats['target_frames']} frames")
            run_by_saved_images = False
            end_time = time.time() + (duration_minutes * 60)
        
        print(f"SORT tracking: Advanced multi-object tracking with Kalman filters")
        print(f"Saving conditions: 5s minimum interval + (new ID OR different angle)")
        print(f"Angle detection: SSIM + Histogram correlation analysis")
        print(f"Files organized by: {self.base_dir}/YEAR/MONTH/DAY/HOUR/")
        
        start_time = time.time()
        attempt_count = 0  # Total attempts (for timing)
        successful_frames = 0  # Only count successful frame processing
        last_stats_time = start_time
        
        try:
            while True:
                current_time = time.time()
                
                # Check stopping conditions
                if run_by_saved_images:
                    if self.stats['images_saved'] >= target_saved_images:
                        print(f"\n✅ Reached target saved images: {target_saved_images}")
                        break
                else:
                    if current_time >= end_time:
                        print(f"\n✅ Reached time limit: {duration_minutes} minutes")
                        break
                
                # Process frame
                success = self.process_frame(current_time, attempt_count)
                
                if success:
                    successful_frames += 1
                    if run_by_saved_images:
                        print(f"📊 Frame {successful_frames} processed | Saved: {self.stats['images_saved']}/{target_saved_images} | New IDs: {self.stats['new_ids_detected']} | Angles: {self.stats['angle_changes_detected']}")
                    else:
                        print(f"📊 Successfully processed frame {successful_frames} | Saved: {self.stats['images_saved']}")
                else:
                    print(f"⚠ Frame processing failed (attempt {attempt_count + 1}), continuing...")
                
                attempt_count += 1
                
                # Print statistics every 30 seconds
                if current_time - last_stats_time >= 30:
                    if run_by_saved_images:
                        self.print_statistics_for_saved_images_mode(successful_frames, target_saved_images)
                    else:
                        self.print_statistics(successful_frames)
                    last_stats_time = current_time
                
                # Wait before next attempt (0.2 seconds)
                time.sleep(check_interval)
                
                # Very gentle video keep-alive (less frequent due to faster processing)
                if attempt_count % 250 == 0:  # Every 250 attempts (50 seconds)
                    self.keep_video_active()
                
                # Safety check - if too many consecutive failures, break
                if attempt_count > 1000 and successful_frames == 0:
                    print("❌ Too many failed attempts, stopping...")
                    break
                    
                # Safety check for saved images mode - if processing too many frames without saves
                if run_by_saved_images and successful_frames > target_saved_images * 50 and self.stats['images_saved'] == 0:
                    print(f"⚠ Processed {successful_frames} frames but saved 0 images.")
                    print("This could be normal if no boats are visible or saving conditions are strict.")
                    
                    user_input = input("Continue? (y/n): ").strip().lower()
                    if user_input != 'y':
                        break
        
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        
        # Final statistics
        print(f"\n=== SCRAPING COMPLETE ===")
        if run_by_saved_images:
            self.print_statistics_for_saved_images_mode(successful_frames, target_saved_images)
        else:
            self.print_statistics(successful_frames)
        print(f"Total runtime: {(time.time() - start_time)/60:.1f} minutes")
        print(f"Successful frames: {successful_frames}")
        print(f"Total attempts: {attempt_count}")
        if attempt_count > 0:
            success_rate = (successful_frames / attempt_count) * 100
            print(f"Success rate: {success_rate:.1f}%")
        print(f"Average processing rate: {successful_frames / ((time.time() - start_time) / 60):.1f} frames/minute")
        print(f"Output directory: {self.base_dir} (organized by date/time)")
        
        # Summary of saved files
        print(f"\n📁 FILES ORGANIZED BY DATE/TIME:")
        total_files = 0
        
        # Scan the base directory for datetime folders
        if os.path.exists(self.base_dir):
            for year in os.listdir(self.base_dir):
                year_path = os.path.join(self.base_dir, year)
                if os.path.isdir(year_path) and year.isdigit():
                    for month in os.listdir(year_path):
                        month_path = os.path.join(year_path, month)
                        if os.path.isdir(month_path) and month.isdigit():
                            for day in os.listdir(month_path):
                                day_path = os.path.join(month_path, day)
                                if os.path.isdir(day_path) and day.isdigit():
                                    for hour in os.listdir(day_path):
                                        hour_path = os.path.join(day_path, hour)
                                        if os.path.isdir(hour_path) and hour.isdigit():
                                            # Count files in this hour folder
                                            jpg_files = len([f for f in os.listdir(hour_path) if f.endswith('.jpg')])
                                            json_files = len([f for f in os.listdir(hour_path) if f.endswith('.json')])
                                            
                                            if jpg_files > 0 or json_files > 0:
                                                print(f"  {year}/{month}/{day}/{hour}h: {jpg_files} images, {json_files} JSON files")
                                                total_files += jpg_files
        
        # Also show boat ID distribution from SORT tracker
        boat_stats = self.sort_tracker.get_boat_stats()
        if boat_stats:
            print(f"\n🚢 SORT TRACKING SUMMARY:")
            for boat_id, count in sorted(boat_stats.items()):
                print(f"  Boat {boat_id:03d}: {count} images saved")
        
        print(f"\nTotal files saved: {total_files * 2} ({total_files} images + {total_files} JSON files)")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.driver:
            self.driver.quit()
            print("✓ Browser closed")

def main():
    print("🚢 INTELLIGENT YOUTUBE BOAT DETECTION SCRAPER - SORT EDITION")
    print("=" * 60)
    print("📦 Required packages: ultralytics selenium webdriver-manager opencv-python")
    print("                     numpy scikit-image scipy filterpy")
    print("⚡ Features: SORT tracking, 0.2s inference, angle detection, 5s save intervals")
    print("=" * 60)
    
    # Get YouTube URL
    youtube_url = input("Enter YouTube URL: ").strip()
    if not youtube_url:
        print("❌ No URL provided")
        return
    
    # Create scraper instance
    scraper = IntelligentYouTubeBoatScraper(youtube_url)
    
    try:
        # Initialize YOLO model
        if not scraper.yolo_detector.initialize_model():
            print("❌ Failed to initialize YOLO model")
            return
        
        # Setup browser
        if not scraper.setup_driver():
            print("❌ Failed to setup browser")
            return
        
        # Load YouTube video
        if not scraper.load_youtube_video():
            print("❌ Failed to load video")
            return
        
        print("\n✅ System ready! Choose scraping mode:")
        print("=== TIME-BASED OPTIONS ===")
        print("1. Quick test (5 minutes)")
        print("2. Medium session (15 minutes)")
        print("3. Long session (30 minutes)")
        print("4. Custom time duration")
        print("")
        print("=== SAVED BOAT IMAGES OPTIONS ===")
        print("5. Quick test (10 saved boat images)")
        print("6. Medium session (50 saved boat images)")
        print("7. Long session (100 saved boat images)")
        print("8. Custom saved boat images count")
        
        choice = input("Enter choice (1-8): ").strip()
        
        # Time-based options
        time_duration_map = {
            '1': 5,
            '2': 15,
            '3': 30
        }
        
        # Saved images options
        saved_images_map = {
            '5': 10,
            '6': 50,
            '7': 100
        }
        
        if choice in time_duration_map:
            duration = time_duration_map[choice]
            print(f"\n🚀 Starting {duration}-minute SORT tracking session...")
            scraper.run_intelligent_scraping(duration_minutes=duration)
            
        elif choice == '4':
            try:
                duration = int(input("Enter duration in minutes: "))
                print(f"\n🚀 Starting {duration}-minute SORT tracking session...")
                scraper.run_intelligent_scraping(duration_minutes=duration)
            except ValueError:
                print("Invalid input, using 15 minutes")
                scraper.run_intelligent_scraping(duration_minutes=15)
                
        elif choice in saved_images_map:
            images = saved_images_map[choice]
            print(f"\n🚀 Starting SORT session to collect {images} saved boat images...")
            scraper.run_intelligent_scraping(max_frames=images)
            
        elif choice == '8':
            try:
                images = int(input("Enter number of boat images to save: "))
                print(f"\n🚀 Starting SORT session to collect {images} saved boat images...")
                scraper.run_intelligent_scraping(max_frames=images)
            except ValueError:
                print("Invalid input, using 50 boat images")
                scraper.run_intelligent_scraping(max_frames=50)
                
        else:
            print("Invalid choice, using 15 minutes")
            scraper.run_intelligent_scraping(duration_minutes=15)
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        
    finally:
        scraper.cleanup()

if __name__ == "__main__":
    main()