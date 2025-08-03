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
            self.model = YOLO('yolo11n.pt')  # Use nano version for speed, change to 'yolo11s.pt' or 'yolo11m.pt' for better accuracy
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
        self.tracking_manager = BoatTrackingManager()
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'boats_detected': 0,
            'images_saved': 0,
            'active_boat_folders': set(),
            'target_frames': 0  # Will be set based on duration
        }
        
        # Ensure base directory exists
        os.makedirs(self.base_dir, exist_ok=True)
        print(f"✓ Base output directory: {self.base_dir}")
        print(f"✓ Files will be organized by: YEAR/MONTH/DAY/HOUR/")
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
        """Process single frame for boat detection"""
        try:
            # Ensure video is playing before processing
            if not self.ensure_video_playing():
                print("⚠ Video not playing, but continuing...")
            
            # Less frequent mouse movement to avoid interfering with video
            if frame_count % 30 == 0:  # Every 30 frames instead of 15
                self.move_mouse_away_from_video()
            
            # Capture frame
            frame = self.get_video_frame()
            if frame is None:
                return False
            
            # Detect boats
            detections = self.yolo_detector.detect_boats(frame)
            
            if detections:
                print(f"🚢 Detected {len(detections)} boat(s)")
                self.stats['boats_detected'] += len(detections)
                
                # Assign detections to trackers
                assigned_detections = self.tracking_manager.assign_detections_to_trackers(detections, current_time)
                
                # Process each assigned detection
                for assignment in assigned_detections:
                    detection = assignment['detection']
                    tracker_id = assignment['tracker_id']
                    is_new = assignment['is_new_tracker']
                    
                    tracker = self.tracking_manager.active_trackers[tracker_id]
                    
                    # Check if we should save this image
                    should_save, reason = tracker.should_save_image(detection['bbox'], current_time)
                    
                    status = "NEW" if is_new else "EXISTING"
                    print(f"  Boat {tracker_id} ({status}): {reason}")
                    
                    if should_save:
                        if self.save_boat_image(frame, detection, tracker_id, current_time):
                            tracker.update_after_save(detection['bbox'], current_time)
            
            self.stats['frames_processed'] += 1
            return True
            
        except Exception as e:
            print(f"✗ Error processing frame: {e}")
            return False
    
    def print_statistics_for_saved_images_mode(self, successful_frames, target_saved_images):
        """Print statistics for saved images mode"""
        print(f"\n📊 STATISTICS:")
        print(f"  Frames processed: {successful_frames}")
        print(f"  Saved images: {self.stats['images_saved']}/{target_saved_images}")
        if target_saved_images > 0:
            progress = (self.stats['images_saved'] / target_saved_images) * 100
            print(f"  Progress: {progress:.1f}%")
        print(f"  Total boat detections: {self.stats['boats_detected']}")
        print(f"  Active boat IDs: {len(self.stats['active_boat_folders'])}")
        print(f"  Current trackers: {len(self.tracking_manager.active_trackers)}")
        print(f"  Camera location: {self.camera_location}")
        
        if self.stats['active_boat_folders']:
            print(f"  Boat folders: {sorted(list(self.stats['active_boat_folders']))}")
            
            # Show saves per boat
            boat_saves = {}
            for boat_id in self.stats['active_boat_folders']:
                if boat_id in self.tracking_manager.active_trackers:
                    boat_saves[boat_id] = self.tracking_manager.active_trackers[boat_id].save_count
            if boat_saves:
                print(f"  Saves per boat: {dict(sorted(boat_saves.items()))}")

    def print_statistics(self, successful_frames=None):
        """Print current statistics"""
        print(f"\n📊 STATISTICS:")
        if successful_frames is not None:
            print(f"  Successful frames: {successful_frames}/{self.stats['target_frames']}")
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
        print(f"  Active boat IDs: {len(self.stats['active_boat_folders'])}")
        print(f"  Current trackers: {len(self.tracking_manager.active_trackers)}")
        print(f"  Camera location: {self.camera_location}")
        
        if self.stats['active_boat_folders']:
            print(f"  Boat folders: {sorted(list(self.stats['active_boat_folders']))}")
            
            # Show saves per boat
            boat_saves = {}
            for boat_id in self.stats['active_boat_folders']:
                if boat_id in self.tracking_manager.active_trackers:
                    boat_saves[boat_id] = self.tracking_manager.active_trackers[boat_id].save_count
            if boat_saves:
                print(f"  Saves per boat: {dict(sorted(boat_saves.items()))}")
    
    def run_intelligent_scraping(self, duration_minutes=None, max_frames=None, check_interval=2):
        """Main intelligent scraping loop - can limit by time OR saved boat images"""
        
        if duration_minutes is not None and max_frames is not None:
            raise ValueError("Cannot specify both duration_minutes and max_frames")
        if duration_minutes is None and max_frames is None:
            raise ValueError("Must specify either duration_minutes or max_frames")
        
        print(f"\n=== STARTING INTELLIGENT BOAT DETECTION ===")
        print(f"Check interval: {check_interval} seconds")
        print(f"Camera location: {self.camera_location}")
        
        # Determine run mode and target
        if max_frames is not None:
            # Frame-based mode (now means max saved boat images)
            target_saved_images = max_frames
            estimated_time = "Variable (depends on boat detection and diversity)"
            print(f"Mode: Saved images limit")
            print(f"Target saved boat images: {target_saved_images}")
            print(f"Estimated time: {estimated_time}")
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
        
        print(f"Max saves per boat: 100 frames")
        print(f"Min time between saves per boat: 5 seconds (for diversity)")
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
                        print(f"📊 Frame {successful_frames} processed | Saved images: {self.stats['images_saved']}/{target_saved_images}")
                    else:
                        print(f"📊 Successfully processed frame {successful_frames}")
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
                
                # Wait before next attempt
                time.sleep(check_interval)
                
                # Very gentle video keep-alive (less frequent)
                if attempt_count % 50 == 0:  # Every 50 attempts
                    self.keep_video_active()
                
                # Safety check - if too many consecutive failures, break
                if attempt_count > 200 and successful_frames == 0:
                    print("❌ Too many failed attempts, stopping...")
                    break
                    
                # Safety check for saved images mode - if processing too many frames without saves
                if run_by_saved_images and successful_frames > target_saved_images * 10 and self.stats['images_saved'] == 0:
                    print(f"⚠ Processed {successful_frames} frames but saved 0 images. Check if boats are being detected.")
                    print("This could be normal if no boats are visible or diversity constraints are too strict.")
                    
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
        
        # Also show boat ID distribution
        if self.stats['active_boat_folders']:
            boat_counts = {}
            # Count saves per boat from trackers
            for boat_id in self.stats['active_boat_folders']:
                if boat_id in self.tracking_manager.active_trackers:
                    boat_counts[boat_id] = self.tracking_manager.active_trackers[boat_id].save_count
            
            if boat_counts:
                print(f"\n🚢 BOAT ID DISTRIBUTION:")
                for boat_id, count in sorted(boat_counts.items()):
                    print(f"  Boat {boat_id:03d}: {count} images")
        
        print(f"\nTotal files saved: {total_files * 2} ({total_files} images + {total_files} JSON files)")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.driver:
            self.driver.quit()
            print("✓ Browser closed")

def main():
    print("🚢 INTELLIGENT YOUTUBE BOAT DETECTION SCRAPER")
    print("=" * 50)
    
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
            print(f"\n🚀 Starting {duration}-minute intelligent boat detection session...")
            scraper.run_intelligent_scraping(duration_minutes=duration)
            
        elif choice == '4':
            try:
                duration = int(input("Enter duration in minutes: "))
                print(f"\n🚀 Starting {duration}-minute intelligent boat detection session...")
                scraper.run_intelligent_scraping(duration_minutes=duration)
            except ValueError:
                print("Invalid input, using 15 minutes")
                scraper.run_intelligent_scraping(duration_minutes=15)
                
        elif choice in saved_images_map:
            images = saved_images_map[choice]
            print(f"\n🚀 Starting session to collect {images} saved boat images...")
            scraper.run_intelligent_scraping(max_frames=images)
            
        elif choice == '8':
            try:
                images = int(input("Enter number of boat images to save: "))
                print(f"\n🚀 Starting session to collect {images} saved boat images...")
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