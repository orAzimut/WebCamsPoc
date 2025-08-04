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
import math
from collections import defaultdict

from boat_tracker import BoatTracker
from yolo_detector import YOLOBoatDetector

class IntelligentYouTubeBoatScraper:
    """Main scraper class with YOLO tracking-based boat detection"""
    
    def __init__(self, youtube_url, headless=False):
        self.url = youtube_url
        self.headless = headless
        self.base_dir = r"C:\Users\OrGil.AzureAD\OneDrive - AMPC\Desktop\Azimut.ai\webScrape\webCams\POC"
        
        # Keep run timestamp for metadata but don't use for folder structure
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.base_dir  # Output dir is now the base POC folder
        
        self.driver = None
        self.video_element = None
        
        # Initialize detection components
        self.yolo_detector = YOLOBoatDetector()
        self.simple_tracker = BoatTracker()
        
        # Camera location mapping
        self.camera_locations = {
            "_KVWehizoNU": {
                "name": "Rotterdam Port 2",
                "country": "Netherlands",
                "city": "Rotterdam",
                "coordinates": {"lat": 51.9244, "lon": 4.4777},
                "timezone": "Europe/Amsterdam"
            },
            "port-of-newcastle-cam": {
                "name": "Newcastle Port",
                "country": "Australia", 
                "city": "Newcastle",
                "state": "New South Wales",
                "coordinates": {"lat": -32.9273, "lon": 151.7817},
                "timezone": "Australia/Sydney"
            },
            "oxx7MqjhOpw": {
                "name": "Dublin Port Bay",
                "country": "Ireland",
                "city": "Dublin",
                "coordinates": {"lat": 53.3498, "lon": -6.2603},
                "timezone": "Europe/Dublin"
            },
            "nmic4tt88-Y": {
                "name": "Hamburg Port 1",
                "country": "Germany",
                "city": "Hamburg", 
                "coordinates": {"lat": 53.5511, "lon": 9.9937},
                "timezone": "Europe/Berlin"
            },
            "bosporus": {
                "name": "Bosporus",
                "country": "Turkey",
                "city": "Istanbul",
                "coordinates": {"lat": 41.0082, "lon": 28.9784},
                "timezone": "Europe/Istanbul"
            },
            "nhEL83_UPpo": {
                "name": "Detroit River",
                "country": "USA",
                "city": "Detroit",
                "state": "Michigan",
                "coordinates": {"lat": 42.3314, "lon": -83.0458},
                "timezone": "America/Detroit"
            }
        }
        
        # Determine camera location for this URL
        self.camera_info = self.get_camera_location(self.url)
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'boats_detected': 0,
            'images_saved': 0,
            'active_boat_ids': set(),
            'target_frames': 0  # Will be set based on duration
        }
        
        # Ensure base directory exists
        os.makedirs(self.base_dir, exist_ok=True)
        print(f"✓ Base output directory: {self.base_dir}")
        print(f"✓ Files will be organized by: YEAR/MONTH/DAY/HOUR/")
        print(f"✓ YouTube URL: {self.url}")
        if self.camera_info:
            print(f"✓ Camera Location: {self.camera_info['name']}, {self.camera_info['country']}")
        else:
            print(f"⚠ Camera location unknown for this URL")
    
    def get_camera_location(self, url):
        """Determine camera location based on URL"""
        try:
            # Check each known camera identifier
            for identifier, location_info in self.camera_locations.items():
                if identifier in url:
                    return location_info
            
            # If no match found, return unknown location
            return {
                "name": "Unknown Camera",
                "country": "Unknown",
                "city": "Unknown",
                "coordinates": {"lat": None, "lon": None},
                "timezone": "UTC",
                "url": url
            }
            
        except Exception as e:
            print(f"⚠ Error determining camera location: {e}")
            return None
    
    def setup_driver(self):
        """Setup Chrome driver optimized for speed"""
        try:
            mode_text = "headless" if self.headless else "visible"
            print(f"⚡ Setting up Chrome driver ({mode_text} mode)...")
            start_time = time.time()
            
            chrome_options = Options()
            
            # Headless mode option
            if self.headless:
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--disable-gpu")
                print("👻 Running in headless mode (invisible browser)")
            
            # Essential options only
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--window-size=1280,720")
            chrome_options.add_argument("--autoplay-policy=no-user-gesture-required")
            
            prefs = {"profile.default_content_setting_values": {"media_stream": 1}}
            chrome_options.add_experimental_option("prefs", prefs)
            
            # IMPROVED CACHING LOGIC
            service = None
            
            # Method 2: Find cached ChromeDriver (FIXED PATHS)
            if service is None:
                try:
                    import glob
                    # CORRECTED cache locations for your system
                    cache_locations = [
                        r"C:\Users\OrGil.AzureAD\.wdm\drivers\chromedriver\win64\*\chromedriver-win32\chromedriver.exe",
                        os.path.expanduser(r"~\.wdm\drivers\chromedriver\win64\*\chromedriver-win32\chromedriver.exe"),
                        os.path.expanduser(r"~\.wdm\drivers\chromedriver\win64\*\chromedriver.exe")
                    ]
                    
                    print("🔍 Checking cache locations:")
                    for location in cache_locations:
                        cached_files = glob.glob(location)
                        if cached_files:
                            # Use the most recent one (highest version number)
                            cached_path = max(cached_files, key=os.path.getmtime)
                            service = Service(cached_path)
                            version = cached_path.split("\\")[-3]  # Extract version from path
                            print(f"✓ Using cached ChromeDriver v{version}")
                            break
                except Exception as e:
                    print(f"⚠ Cache search error: {e}")
            
            # Method 3: Download if not found
            if service is None:
                print("📥 No cached driver found, downloading...")
                from webdriver_manager.chrome import ChromeDriverManager
                driver_path = ChromeDriverManager().install()
                service = Service(driver_path)
                print(f"💾 Driver downloaded to: {driver_path}")
            
            # Create driver
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            setup_time = time.time() - start_time
            print(f"✓ Chrome driver ready in {setup_time:.1f}s")
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
    
    def save_boat_image(self, boat_image, boat_detection, current_time):
        """Save individual cropped boat image with clean JSON metadata"""
        try:
            boat_id = boat_detection['track_id']
            
            # Validate boat_image
            if boat_image is None or boat_image.size == 0:
                print(f"  ⚠ Invalid boat image for boat {boat_id}")
                return False
            
            # Create datetime-based folder structure
            datetime_folder = self.create_datetime_folder()
            
            # Create filename (same for both image and JSON)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            base_filename = f"boat_{boat_id:03d}_{timestamp}_{self.stats['images_saved']:04d}"
            
            # Save the CROPPED boat image
            image_filepath = os.path.join(datetime_folder, f"{base_filename}.jpg")
            cv2.imwrite(image_filepath, boat_image)
            
            # Create clean JSON metadata
            metadata = {
                "timestamp": timestamp,
                "youtube_url": self.url,
                "boat_id": boat_id,
                "confidence": round(boat_detection['confidence'], 3),
                "class": boat_detection['class'],
                "cropped_image_info": {
                    "width": boat_image.shape[1],
                    "height": boat_image.shape[0],
                    "channels": boat_image.shape[2] if len(boat_image.shape) == 3 else 1,
                    "note": "This JSON corresponds to a cropped boat image"
                },
                "camera_location": self.camera_info if self.camera_info else {
                    "name": "Unknown Camera",
                    "country": "Unknown",
                    "city": "Unknown",
                    "coordinates": {"lat": None, "lon": None},
                    "timezone": "UTC",
                    "url": self.url
                },
                "datetime_path": {
                    "year": datetime.now().strftime("%Y"),
                    "month": datetime.now().strftime("%m"),
                    "day": datetime.now().strftime("%d"), 
                    "hour": datetime.now().strftime("%H"),
                    "full_path": datetime_folder
                },
                "tracking_info": {
                    "save_count_for_this_boat": self.simple_tracker.get_save_count(boat_id) + 1,
                    "time_since_last_save": current_time - self.simple_tracker.last_save_times.get(boat_id, 0)
                }
            }
            
            # Save JSON metadata  
            json_filepath = os.path.join(datetime_folder, f"{base_filename}.json")
            with open(json_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update tracker and statistics
            self.simple_tracker.record_save(boat_id, current_time)
            self.stats['active_boat_ids'].add(boat_id)
            self.stats['images_saved'] += 1
            
            print(f"💾 Saved boat {boat_id} crop: {base_filename}.jpg + .json")
            print(f"   📐 Size: {boat_image.shape[1]}x{boat_image.shape[0]} pixels")
            print(f"   📁 Path: {datetime_folder}")
            print(f"   📊 Save #{self.simple_tracker.get_save_count(boat_id)} for boat {boat_id}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error saving boat image: {e}")
            return False
    
    def process_frame(self, current_time, frame_count):
            """Process single frame for boat detection using YOLO tracking"""
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
                
                # Track boats using YOLO's built-in tracking
                tracked_boats = self.yolo_detector.track_boats(frame)
                
                if tracked_boats:
                    print(f"🚢 Tracked {len(tracked_boats)} boat(s)")
                    self.stats['boats_detected'] += len(tracked_boats)
                    
                    # Process each tracked boat
                    for boat in tracked_boats:
                        track_id = boat['track_id']
                        
                        # Skip boats without valid tracking ID
                        if track_id is None:
                            print(f"  ⚠ Boat detected but no tracking ID assigned")
                            continue
                        
                        # ENHANCED DETECTION: Extract bbox and crop boat image
                        try:
                            bbox = boat['bbox']  # Should be [x1, y1, x2, y2] or [x, y, w, h]
                            
                            # Handle different bbox formats
                            if len(bbox) == 4:
                                # Check if it's [x1, y1, x2, y2] or [x, y, w, h]
                                if bbox[2] > frame.shape[1] or bbox[3] > frame.shape[0]:
                                    # Likely [x, y, w, h] format
                                    x, y, w, h = bbox
                                    x1, y1, x2, y2 = x, y, x + w, y + h
                                    bbox_for_tracker = (x, y, w, h)  # BoatTracker expects (x, y, w, h)
                                else:
                                    # Likely [x1, y1, x2, y2] format
                                    x1, y1, x2, y2 = bbox
                                    w, h = x2 - x1, y2 - y1
                                    bbox_for_tracker = (x1, y1, w, h)  # Convert to (x, y, w, h)
                            else:
                                raise ValueError(f"Unexpected bbox format: {bbox}")
                            
                            # Ensure bbox is within frame bounds
                            x1 = max(0, int(x1))
                            y1 = max(0, int(y1))
                            x2 = min(frame.shape[1], int(x2))
                            y2 = min(frame.shape[0], int(y2))
                            
                            # Create wider square crop around the boat
                            bbox_width = x2 - x1
                            bbox_height = y2 - y1
                            
                            # Calculate center of bbox
                            center_x = x1 + bbox_width // 2
                            center_y = y1 + bbox_height // 2
                            
                            # Determine square size (larger dimension + padding)
                            padding_factor = 1.5  # 50% larger than bbox
                            square_size = int(max(bbox_width, bbox_height) * padding_factor)
                            
                            # Ensure square size is even for clean cropping
                            if square_size % 2 != 0:
                                square_size += 1
                            
                            half_size = square_size // 2
                            
                            # Calculate square crop coordinates
                            crop_x1 = center_x - half_size
                            crop_y1 = center_y - half_size
                            crop_x2 = center_x + half_size
                            crop_y2 = center_y + half_size
                            
                            # Ensure crop stays within frame bounds
                            crop_x1 = max(0, crop_x1)
                            crop_y1 = max(0, crop_y1)
                            crop_x2 = min(frame.shape[1], crop_x2)
                            crop_y2 = min(frame.shape[0], crop_y2)
                            
                            # Crop wider square area around boat
                            boat_image = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                            
                            # Validate cropped image
                            if boat_image is None or boat_image.size == 0:
                                print(f"  ⚠ Invalid crop for boat {track_id}")
                                continue
                            
                            # Check if we should save this boat (ENHANCED VERSION)
                            should_save, reason = self.simple_tracker.should_save_boat(
                                track_id, 
                                current_time, 
                                bbox_for_tracker,  # (x, y, width, height)
                                boat_image         # cropped boat image
                            )
                            
                        except Exception as e:
                            print(f"  ⚠ Enhanced detection failed for boat {track_id}: {e}")
                            # Fallback to basic detection
                            should_save, reason = self.simple_tracker.should_save_boat(track_id, current_time)
                            boat_image = None
                        
                        print(f"  🚢 Boat ID {track_id}: {reason}")
                        
                        if should_save:
                            if boat_image is not None:
                                # Save the CROPPED boat image (FIXED!)
                                if self.save_boat_image(boat_image, boat, current_time):
                                    print(f"    ✅ Saved cropped boat image successfully")
                                else:
                                    print(f"    ❌ Save failed")
                            else:
                                print(f"    ❌ No valid boat image to save")
                
                self.stats['frames_processed'] += 1
                return True
                
            except Exception as e:
                print(f"✗ Error processing frame: {e}")
                return False
    
   
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
        print(f"  Active boat IDs: {len(self.stats['active_boat_ids'])}")
        print(f"  Currently tracked IDs: {len(self.simple_tracker.get_tracked_boat_ids())}")
        
        if self.stats['active_boat_ids']:
            print(f"  Boat IDs seen: {sorted(list(self.stats['active_boat_ids']))}")
            
            # Show saves per boat
            boat_saves = {}
            for boat_id in self.stats['active_boat_ids']:
                boat_saves[boat_id] = self.simple_tracker.get_save_count(boat_id)
            if boat_saves:
                print(f"  Saves per boat: {dict(sorted(boat_saves.items()))}")
    
    def run_continuous_scraping(self, check_interval=0.05):
        """Run continuous boat detection until stopped with Ctrl+C"""
        
        print(f"\n=== CONTINUOUS BOAT DETECTION ===")
        print(f"Press Ctrl+C to stop at any time...")
        print(f"Files organized by: {self.base_dir}/YEAR/MONTH/DAY/HOUR/")
        print(f"💡 Now saving individual cropped boat images (not full frames)")
        
        start_time = time.time()
        attempt_count = 0
        successful_frames = 0
        last_stats_time = start_time
        
        try:
            while True:  # Simple infinite loop
                current_time = time.time()
                
                success = self.process_frame(current_time, attempt_count)
                
                if success:
                    successful_frames += 1
                    print(f"📊 Frame {successful_frames} | Saved: {self.stats['images_saved']} | Boats: {len(self.stats['active_boat_ids'])}")
                
                attempt_count += 1
                
                # Stats every 30 seconds
                if current_time - last_stats_time >= 30:
                    self.print_statistics(successful_frames)
                    last_stats_time = current_time
                
                time.sleep(check_interval)
                
                if attempt_count % 50 == 0:
                    self.keep_video_active()
        
        except KeyboardInterrupt:
            print(f"\n🛑 Stopped after {(time.time() - start_time)/60:.1f} minutes")
            self.print_statistics(successful_frames)
            
    def cleanup(self):
        """Cleanup resources"""
        if self.driver:
            self.driver.quit()
            print("✓ Browser closed")