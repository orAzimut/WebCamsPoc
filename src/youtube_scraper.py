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

from boat_tracker import SimpleBoatTracker
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
        self.simple_tracker = SimpleBoatTracker()
        
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
            
            # Method 1: Try manual chromedriver in project folder
            local_paths = [
                os.path.join(os.path.dirname(__file__), "chromedriver.exe"),
                os.path.join(os.getcwd(), "chromedriver.exe"),
                "./chromedriver.exe"
            ]
            
            for path in local_paths:
                if os.path.exists(path):
                    service = Service(path)
                    print(f"✓ Using local ChromeDriver: {path}")
                    break
            
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
    
    def save_boat_image(self, frame, boat_detection, current_time):
        """Save boat image with JSON metadata in date/time organized folders"""
        try:
            boat_id = boat_detection['track_id']
            
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
            
            print(f"💾 Saved boat {boat_id} image: {base_filename}.jpg + .json")
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
                    
                    # Check if we should save this boat
                    should_save, reason = self.simple_tracker.should_save_boat(track_id, current_time)
                    
                    print(f"  🚢 Boat ID {track_id}: {reason}")
                    
                    if should_save:
                        if self.save_boat_image(frame, boat, current_time):
                            print(f"    ✅ Saved successfully")
                        else:
                            print(f"    ❌ Save failed")
            
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
    
    def run_intelligent_scraping(self, duration_minutes=None, max_frames=None, check_interval=0.1):
        """Main intelligent scraping loop - can limit by time OR saved boat images"""
        
        if duration_minutes is not None and max_frames is not None:
            raise ValueError("Cannot specify both duration_minutes and max_frames")
        if duration_minutes is None and max_frames is None:
            raise ValueError("Must specify either duration_minutes or max_frames")
        
        print(f"\n=== STARTING INTELLIGENT BOAT DETECTION WITH YOLO TRACKING ===")
        print(f"Check interval: {check_interval} seconds")
        print(f"Save interval: {self.simple_tracker.min_save_interval} seconds per boat")
        
        # Determine run mode and target
        if max_frames is not None:
            # Frame-based mode (now means max saved boat images)
            target_saved_images = max_frames
            estimated_time = "Variable (depends on boat detection)"
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
        
        print(f"Max saves per boat: {self.simple_tracker.max_saves_per_boat} images")
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
                    print("This could be normal if no boats are visible.")
                    
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
        
        # Show boat ID distribution
        if self.stats['active_boat_ids']:
            print(f"\n🚢 BOAT ID DISTRIBUTION:")
            for boat_id in sorted(self.stats['active_boat_ids']):
                count = self.simple_tracker.get_save_count(boat_id)
                print(f"  Boat {boat_id:03d}: {count} images")
        
        print(f"\nTotal files saved: {total_files * 2} ({total_files} images + {total_files} JSON files)")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.driver:
            self.driver.quit()
            print("✓ Browser closed")