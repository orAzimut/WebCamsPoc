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

# Google Cloud Storage imports
from google.cloud import storage
import tempfile
import io

from boat_tracker import BoatTracker
from yolo_detector import YOLOBoatDetector

class IntelligentYouTubeBoatScraper:
    """Main scraper class with YOLO tracking-based boat detection and GCS storage"""
    
    def __init__(self, youtube_url, headless=False, use_gcs=True, gcs_bucket_name=None):
        self.url = youtube_url
        self.headless = headless
        self.use_gcs = use_gcs
        
        # Local base directory (for fallback or temp files)
        self.base_dir = r"C:\Users\OrGil.AzureAD\OneDrive - AMPC\Desktop\Azimut.ai\webScrape\webCams\POC"
        
        # GCS configuration - use provided bucket name or default
        self.gcs_bucket_name = gcs_bucket_name if gcs_bucket_name else "outsource_data"
        self.gcs_client = None
        self.gcs_bucket = None
        
        # Initialize GCS if enabled
        if self.use_gcs:
            self.setup_gcs()
        
        # Keep run timestamp for metadata
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.base_dir  # Keep for compatibility
        
        self.driver = None
        self.video_element = None
        
        # Initialize detection components
        self.yolo_detector = YOLOBoatDetector()
        self.simple_tracker = BoatTracker()
        
        # Camera location mapping
        self.camera_locations = {
            "_KVWehizoNU": {
                "name": "Rotterdam_Port_2",
                "country": "Netherlands",
                "city": "Rotterdam",
                "coordinates": {"lat": 51.9244, "lon": 4.4777},
                "timezone": "Europe/Amsterdam"
            },
            "port-of-newcastle-cam": {
                "name": "Newcastle_Port",
                "country": "Australia", 
                "city": "Newcastle",
                "state": "New South Wales",
                "coordinates": {"lat": -32.9273, "lon": 151.7817},
                "timezone": "Australia/Sydney"
            },
            "oxx7MqjhOpw": {
                "name": "Dublin_Port_Bay",
                "country": "Ireland",
                "city": "Dublin",
                "coordinates": {"lat": 53.3498, "lon": -6.2603},
                "timezone": "Europe/Dublin"
            },
            "nmic4tt88-Y": {
                "name": "Hamburg_Port_1",
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
                "name": "Detroit_River",
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
            'target_frames': 0,
            'gcs_uploads': 0,
            'gcs_failures': 0
        }
        
        # Log initialization
        if self.use_gcs:
            print(f"☁️ Google Cloud Storage enabled")
            print(f"☁️ Bucket: {self.gcs_bucket_name}")
            print(f"☁️ Images path: reidentification/bronze/raw_crops/webCams/{self.camera_info['name'] if self.camera_info else 'Unknown'}")
            print(f"☁️ JSONs path: reidentification/bronze/json_lables/webCams/{self.camera_info['name'] if self.camera_info else 'Unknown'}")
        else:
            print(f"💾 Local storage mode")
            print(f"📁 Base directory: {self.base_dir}")
        
        print(f"📹 YouTube URL: {self.url}")
        if self.camera_info:
            print(f"📍 Camera Location: {self.camera_info['name']}, {self.camera_info['country']}")
        else:
            print(f"⚠️ Camera location unknown for this URL")
    
    def setup_gcs(self):
        """Setup Google Cloud Storage client"""
        try:
            # Path to credentials file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            credentials_path = os.path.join(script_dir, "..", "resources", "credentials.json")
            
            # Check if credentials file exists
            if not os.path.exists(credentials_path):
                print(f"⚠️ GCS credentials not found at: {credentials_path}")
                print(f"⚠️ Falling back to local storage")
                self.use_gcs = False
                return False
            
            # Set environment variable for GCS authentication
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            
            # Initialize GCS client
            self.gcs_client = storage.Client()
            self.gcs_bucket = self.gcs_client.bucket(self.gcs_bucket_name)
            
            print(f"✅ GCS client initialized with credentials from: {credentials_path}")
            
            # Test with a simple write operation instead of bucket.reload()
            try:
                # Try to write a test file to check permissions
                test_blob_name = f"test_access_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                test_blob = self.gcs_bucket.blob(test_blob_name)
                test_blob.upload_from_string("Test access", content_type='text/plain')
                
                # If successful, delete the test file
                test_blob.delete()
                print(f"✅ Successfully verified write access to GCS bucket: {self.gcs_bucket_name}")
                return True
                
            except Exception as e:
                if "403" in str(e):
                    print(f"⚠️ Permission denied for bucket {self.gcs_bucket_name}")
                    print(f"⚠️ Error: {e}")
                    print(f"📝 To fix this:")
                    print(f"   1. Go to GCS Console: https://console.cloud.google.com/storage/browser/{self.gcs_bucket_name}")
                    print(f"   2. Click 'Permissions' tab")
                    print(f"   3. Add your service account with 'Storage Object Admin' role")
                    print(f"   4. Service account from credentials.json")
                elif "404" in str(e):
                    print(f"⚠️ Bucket '{self.gcs_bucket_name}' not found")
                    print(f"📝 Please create the bucket or check the bucket name")
                else:
                    print(f"⚠️ Could not access bucket {self.gcs_bucket_name}: {e}")
                
                print(f"⚠️ Falling back to local storage")
                self.use_gcs = False
                return False
                
        except Exception as e:
            print(f"⚠️ Failed to setup GCS: {e}")
            print(f"⚠️ Falling back to local storage")
            self.use_gcs = False
            return False
    
    def get_camera_location(self, url):
        """Determine camera location based on URL"""
        try:
            # Check each known camera identifier
            for identifier, location_info in self.camera_locations.items():
                if identifier in url:
                    return location_info
            
            # If no match found, return unknown location
            return {
                "name": "Unknown_Camera",
                "country": "Unknown",
                "city": "Unknown",
                "coordinates": {"lat": None, "lon": None},
                "timezone": "UTC",
                "url": url
            }
            
        except Exception as e:
            print(f"⚠️ Error determining camera location: {e}")
            return None
    
    def setup_driver(self):
        """Setup Chrome driver optimized for speed"""
        try:
            mode_text = "headless" if self.headless else "visible"
            print(f"⚡ Setting up Chrome driver ({mode_text} mode)...")
            start_time = time.time()
            
            chrome_options = Options()
            
            # Headless mode option - ENHANCED FOR VIDEO STREAMING
            if self.headless:
                # Use new headless mode for better compatibility
                chrome_options.add_argument("--headless=new")  # New headless mode
                chrome_options.add_argument("--disable-gpu")
                chrome_options.add_argument("--disable-software-rasterizer")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--no-sandbox")
                
                # Force larger window size for better video rendering
                chrome_options.add_argument("--window-size=1920,1080")
                chrome_options.add_argument("--start-maximized")
                
                # Additional flags for video playback in headless
                chrome_options.add_argument("--use-fake-ui-for-media-stream")
                chrome_options.add_argument("--use-fake-device-for-media-stream")
                chrome_options.add_argument("--autoplay-policy=no-user-gesture-required")
                
                # Disable throttling that affects headless mode
                chrome_options.add_argument("--disable-background-timer-throttling")
                chrome_options.add_argument("--disable-backgrounding-occluded-windows")
                chrome_options.add_argument("--disable-renderer-backgrounding")
                chrome_options.add_argument("--disable-features=TranslateUI")
                chrome_options.add_argument("--disable-ipc-flooding-protection")
                
                print("👻 Running in headless mode with video optimization")
            else:
                chrome_options.add_argument("--window-size=1280,720")
            
            # Common options for both modes
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--autoplay-policy=no-user-gesture-required")
            
            # User agent to appear more like a real browser
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            
            prefs = {
                "profile.default_content_setting_values": {
                    "media_stream": 1,
                    "media_stream_mic": 1,
                    "media_stream_camera": 1,
                    "automatic_downloads": 1
                }
            }
            chrome_options.add_experimental_option("prefs", prefs)
            
            # IMPROVED CACHING LOGIC
            service = None
            
            # Method 2: Find cached ChromeDriver
            if service is None:
                try:
                    import glob
                    cache_locations = [
                        r"C:\Users\OrGil.AzureAD\.wdm\drivers\chromedriver\win64\*\chromedriver-win32\chromedriver.exe",
                        os.path.expanduser(r"~\.wdm\drivers\chromedriver\win64\*\chromedriver-win32\chromedriver.exe"),
                        os.path.expanduser(r"~\.wdm\drivers\chromedriver\win64\*\chromedriver.exe")
                    ]
                    
                    print("🔍 Checking cache locations:")
                    for location in cache_locations:
                        cached_files = glob.glob(location)
                        if cached_files:
                            cached_path = max(cached_files, key=os.path.getmtime)
                            service = Service(cached_path)
                            version = cached_path.split("\\")[-3]
                            print(f"✅ Using cached ChromeDriver v{version}")
                            break
                except Exception as e:
                    print(f"⚠️ Cache search error: {e}")
            
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
            print(f"✅ Chrome driver ready in {setup_time:.1f}s")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize Chrome driver: {e}")
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
            
            print("✅ YouTube video loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load YouTube video: {e}")
            return False
    
    def move_mouse_away_from_video(self):
        """Move mouse away from video to avoid YouTube timeline/controls"""
        try:
            action = ActionChains(self.driver)
            action.move_to_element_with_offset(self.driver.find_element(By.TAG_NAME, "body"), 50, 50)
            action.perform()
        except Exception as e:
            print(f"⚠️ Could not move mouse away from video: {e}")
    
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
                        print("✅ Dismissed consent dialog")
                        time.sleep(2)
                        break
                except:
                    continue
        except:
            pass
    
    def setup_video_playback(self):
        """Setup video playback and theater mode - ENHANCED FOR HEADLESS"""
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
            
            # In headless mode, set video quality and ensure autoplay
            if self.headless:
                try:
                    # Mute video (required for autoplay in headless)
                    self.driver.execute_script("""
                        var video = document.querySelector('video.html5-main-video');
                        if (video) {
                            video.muted = true;
                            video.volume = 0;
                        }
                    """)
                    
                    # Set video quality to auto or highest available
                    self.driver.execute_script("""
                        var player = document.querySelector('#movie_player');
                        if (player && player.setPlaybackQualityRange) {
                            player.setPlaybackQualityRange('hd720', 'highres');
                        }
                    """)
                    
                    # Disable annotations and cards
                    self.driver.execute_script("""
                        var settings = document.querySelector('.ytp-settings-button');
                        if (settings) {
                            // Disable annotations
                            var player = document.querySelector('#movie_player');
                            if (player && player.unloadModule) {
                                player.unloadModule('annotations_module');
                            }
                        }
                    """)
                    
                    print("✅ Headless video optimization applied")
                except Exception as e:
                    print(f"⚠️ Could not optimize headless video: {e}")
            else:
                # Try theater mode for non-headless
                self.video_element.send_keys('t')
                time.sleep(2)
            
            print("✅ Video setup complete")
            
        except Exception as e:
            print(f"Video setup warning: {e}")
    
    def ensure_video_playing(self):
        """Ensure video is playing before taking screenshot - ENHANCED FOR HEADLESS"""
        try:
            # In headless mode, be more aggressive about keeping video playing
            if self.headless:
                # Force play via multiple methods
                try:
                    # Method 1: Direct JavaScript play() with promise handling
                    self.driver.execute_script("""
                        var video = document.querySelector('video.html5-main-video');
                        if (video) {
                            video.muted = true;  // Ensure muted for autoplay
                            video.play().then(() => {
                                console.log('Video playing');
                            }).catch((e) => {
                                console.log('Play failed:', e);
                                video.click();  // Try clicking as fallback
                            });
                        }
                    """)
                    time.sleep(0.5)
                except:
                    pass
                
                # Method 2: Remove any pause overlays
                try:
                    self.driver.execute_script("""
                        var pauseOverlay = document.querySelector('.ytp-pause-overlay');
                        if (pauseOverlay) pauseOverlay.style.display = 'none';
                        
                        var spinner = document.querySelector('.ytp-spinner');
                        if (spinner) spinner.style.display = 'none';
                    """)
                except:
                    pass
            
            # Check if video is paused
            is_paused = self.driver.execute_script("return document.querySelector('video.html5-main-video').paused;")
            
            # Also check if video is buffering or has ended
            video_state = self.driver.execute_script("""
                var video = document.querySelector('video.html5-main-video');
                return {
                    paused: video.paused,
                    ended: video.ended,
                    currentTime: video.currentTime,
                    duration: video.duration,
                    readyState: video.readyState,
                    networkState: video.networkState
                };
            """)
            
            if video_state['ended']:
                print("⚠️ Video ended - refreshing page...")
                self.driver.refresh()
                time.sleep(5)
                self.load_youtube_video()
                return False
            
            if video_state['readyState'] < 2:  # Not enough data
                print("⚠️ Video buffering...")
                time.sleep(2)
                return False
            
            if is_paused:
                print("⏸️ Video paused - resuming...")
                
                # Try multiple methods to resume
                methods = [
                    # JavaScript play
                    lambda: self.driver.execute_script("document.querySelector('video.html5-main-video').play();"),
                    # Click play button
                    lambda: self.driver.find_element(By.CSS_SELECTOR, "button.ytp-play-button").click(),
                    # Spacebar
                    lambda: self.video_element.send_keys(' '),
                    # Click video element
                    lambda: self.video_element.click()
                ]
                
                for i, method in enumerate(methods):
                    try:
                        method()
                        time.sleep(1)
                        
                        still_paused = self.driver.execute_script("return document.querySelector('video.html5-main-video').paused;")
                        if not still_paused:
                            print(f"✅ Video resumed via method {i+1}")
                            return True
                    except:
                        continue
                
                print("⚠️ Could not resume video")
                return False
            
            return True
                
        except Exception as e:
            print(f"❌ Error checking video status: {e}")
            return False
    
    def keep_video_active(self):
        """Keep video active without interfering with playback"""
        try:
            action = ActionChains(self.driver)
            action.move_to_element_with_offset(self.video_element, 10, 10)
            action.perform()
            time.sleep(0.1)
            
            action = ActionChains(self.driver)
            action.move_by_offset(200, -50)
            action.perform()
            
        except Exception as e:
            print(f"⚠️ Could not keep video active: {e}")
    
    def get_video_frame(self):
        """Capture current video frame as OpenCV image"""
        try:
            video_png = self.video_element.screenshot_as_png
            nparr = np.frombuffer(video_png, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
            
        except Exception as e:
            print(f"❌ Error capturing video frame: {e}")
            return None
    
    def create_gcs_paths(self):
        """Create GCS paths for image and JSON storage"""
        now = datetime.now()
        
        # Get camera location name (replace spaces with underscores)
        location_name = self.camera_info['name'] if self.camera_info else 'Unknown_Camera'
        
        # Create year/month/day/hour structure
        year = now.strftime("%Y")
        month = now.strftime("%m") 
        day = now.strftime("%d")
        hour = now.strftime("%H")
        
        # Build GCS paths
        image_path = f"reidentification/bronze/raw_crops/webCams/{location_name}/{year}/{month}/{day}/{hour}"
        json_path = f"reidentification/bronze/json_lables/webCams/{location_name}/{year}/{month}/{day}/{hour}"
        
        return image_path, json_path
    
    def upload_to_gcs(self, blob_name, data, content_type='application/octet-stream'):
        """Upload data to Google Cloud Storage"""
        try:
            blob = self.gcs_bucket.blob(blob_name)
            
            if isinstance(data, bytes):
                blob.upload_from_string(data, content_type=content_type)
            elif isinstance(data, str):
                blob.upload_from_string(data.encode('utf-8'), content_type=content_type)
            else:
                # For numpy arrays (images)
                is_success, buffer = cv2.imencode('.jpg', data)
                if is_success:
                    blob.upload_from_string(buffer.tobytes(), content_type='image/jpeg')
                else:
                    print(f"❌ Failed to encode image")
                    return False
            
            self.stats['gcs_uploads'] += 1
            return True
            
        except Exception as e:
            print(f"❌ GCS upload failed: {e}")
            self.stats['gcs_failures'] += 1
            return False
    
    def save_boat_image(self, boat_image, boat_detection, current_time):
        """Save individual cropped boat image with clean JSON metadata"""
        try:
            boat_id = boat_detection['track_id']
            
            # Validate boat_image
            if boat_image is None or boat_image.size == 0:
                print(f"  ⚠️ Invalid boat image for boat {boat_id}")
                return False
            
            # Create filename (same for both image and JSON)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            base_filename = f"boat_{boat_id:03d}_{timestamp}_{self.stats['images_saved']:04d}"
            
            # Create metadata
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
                    "name": "Unknown_Camera",
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
                    "hour": datetime.now().strftime("%H")
                },
                "tracking_info": {
                    "save_count_for_this_boat": self.simple_tracker.get_save_count(boat_id) + 1,
                    "time_since_last_save": current_time - self.simple_tracker.last_save_times.get(boat_id, 0)
                }
            }
            
            if self.use_gcs:
                # Save to Google Cloud Storage
                image_path, json_path = self.create_gcs_paths()
                
                # Upload image
                image_blob_name = f"{image_path}/{base_filename}.jpg"
                is_success, buffer = cv2.imencode('.jpg', boat_image)
                if is_success:
                    if self.upload_to_gcs(image_blob_name, buffer.tobytes(), 'image/jpeg'):
                        print(f"☁️ Uploaded image: {image_blob_name}")
                    else:
                        print(f"❌ Failed to upload image to GCS")
                        # Fallback to local
                        self.save_local_fallback(boat_image, metadata, base_filename)
                        return False
                
                # Upload JSON
                json_blob_name = f"{json_path}/{base_filename}.json"
                json_data = json.dumps(metadata, indent=2)
                if self.upload_to_gcs(json_blob_name, json_data, 'application/json'):
                    print(f"☁️ Uploaded JSON: {json_blob_name}")
                else:
                    print(f"❌ Failed to upload JSON to GCS")
                    return False
                
                print(f"💾 Saved boat {boat_id} to GCS")
                print(f"   📐 Size: {boat_image.shape[1]}x{boat_image.shape[0]} pixels")
                print(f"   ☁️ Location: {self.camera_info['name'] if self.camera_info else 'Unknown'}")
                print(f"   📊 Save #{self.simple_tracker.get_save_count(boat_id) + 1} for boat {boat_id}")
                
            else:
                # Save locally (original behavior)
                self.save_local_fallback(boat_image, metadata, base_filename)
            
            # Update tracker and statistics
            self.simple_tracker.record_save(boat_id, current_time)
            self.stats['active_boat_ids'].add(boat_id)
            self.stats['images_saved'] += 1
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving boat image: {e}")
            return False
    
    def save_local_fallback(self, boat_image, metadata, base_filename):
        """Save to local storage as fallback"""
        try:
            # Create datetime-based folder structure locally
            now = datetime.now()
            year = now.strftime("%Y")
            month = now.strftime("%m") 
            day = now.strftime("%d")
            hour = now.strftime("%H")
            
            datetime_folder = os.path.join(self.base_dir, year, month, day, hour)
            os.makedirs(datetime_folder, exist_ok=True)
            
            # Save image locally
            image_filepath = os.path.join(datetime_folder, f"{base_filename}.jpg")
            cv2.imwrite(image_filepath, boat_image)
            
            # Save JSON locally
            json_filepath = os.path.join(datetime_folder, f"{base_filename}.json")
            with open(json_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"💾 Saved locally (fallback): {base_filename}")
            
        except Exception as e:
            print(f"❌ Local save also failed: {e}")
    
    def process_frame(self, current_time, frame_count):
        """Process single frame for boat detection using YOLO tracking"""
        try:
            # In headless mode, be more aggressive about ensuring video plays
            if self.headless:
                # Every 10 frames, force video to play
                if frame_count % 10 == 0:
                    self.driver.execute_script("""
                        var video = document.querySelector('video.html5-main-video');
                        if (video && video.paused) {
                            video.play();
                        }
                    """)
                
                # Every 100 frames in headless, refresh video element
                if frame_count % 100 == 0:
                    print("🔄 Refreshing video element (headless mode)...")
                    self.refresh_video_element()
            
            # Ensure video is playing before processing
            if not self.ensure_video_playing():
                if self.headless:
                    print("⚠️ Video not playing in headless, attempting recovery...")
                    self.refresh_video_element()
                else:
                    print("⚠️ Video not playing, but continuing...")
            
            # Less frequent mouse movement to avoid interfering with video
            if frame_count % 30 == 0 and not self.headless:
                self.move_mouse_away_from_video()
            
            # Capture frame
            frame = self.get_video_frame()
            if frame is None:
                print("⚠️ Failed to capture frame")
                if self.headless:
                    # In headless, try alternative capture method
                    frame = self.get_video_frame_alternative()
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
                        print(f"  ⚠️ Boat detected but no tracking ID assigned")
                        continue
                    
                    # Extract bbox and crop boat image
                    try:
                        bbox = boat['bbox']
                        
                        # Handle different bbox formats
                        if len(bbox) == 4:
                            if bbox[2] > frame.shape[1] or bbox[3] > frame.shape[0]:
                                x, y, w, h = bbox
                                x1, y1, x2, y2 = x, y, x + w, y + h
                                bbox_for_tracker = (x, y, w, h)
                            else:
                                x1, y1, x2, y2 = bbox
                                w, h = x2 - x1, y2 - y1
                                bbox_for_tracker = (x1, y1, w, h)
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
                            print(f"  ⚠️ Invalid crop for boat {track_id}")
                            continue
                        
                        # Check if we should save this boat
                        should_save, reason = self.simple_tracker.should_save_boat(
                            track_id, 
                            current_time, 
                            bbox_for_tracker,
                            boat_image
                        )
                        
                    except Exception as e:
                        print(f"  ⚠️ Enhanced detection failed for boat {track_id}: {e}")
                        should_save, reason = self.simple_tracker.should_save_boat(track_id, current_time)
                        boat_image = None
                    
                    print(f"  🚢 Boat ID {track_id}: {reason}")
                    
                    if should_save:
                        if boat_image is not None:
                            if self.save_boat_image(boat_image, boat, current_time):
                                print(f"    ✅ Saved successfully")
                            else:
                                print(f"    ❌ Save failed")
                        else:
                            print(f"    ❌ No valid boat image to save")
            else:
                # In headless mode, log when no boats detected
                if self.headless and frame_count % 20 == 0:
                    print(f"📊 Frame {frame_count}: No boats detected (headless mode)")
            
            self.stats['frames_processed'] += 1
            return True
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            if self.headless:
                print("🔄 Attempting recovery for headless mode...")
                self.refresh_video_element()
            return False
    
    def get_video_frame_alternative(self):
        """Alternative method to capture video frame for headless mode"""
        try:
            # Take full page screenshot and crop to video area
            full_screenshot = self.driver.get_screenshot_as_png()
            nparr = np.frombuffer(full_screenshot, np.uint8)
            full_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Get video element position and size
            location = self.video_element.location
            size = self.video_element.size
            
            # Crop to video area
            x = location['x']
            y = location['y']
            w = size['width']
            h = size['height']
            
            if w > 0 and h > 0:
                frame = full_frame[y:y+h, x:x+w]
                return frame
            
            return None
            
        except Exception as e:
            print(f"❌ Alternative frame capture failed: {e}")
            return None
    
    def refresh_video_element(self):
        """Refresh video element reference and ensure playing (for headless mode)"""
        try:
            # Re-find video element
            self.video_element = self.driver.find_element(By.CSS_SELECTOR, "video.html5-main-video")
            
            # Force play
            self.driver.execute_script("""
                var video = document.querySelector('video.html5-main-video');
                if (video) {
                    video.muted = true;
                    video.play();
                    // Seek forward slightly to ensure fresh frames
                    if (video.currentTime > 10) {
                        video.currentTime = video.currentTime + 0.1;
                    }
                }
            """)
            
            print("✅ Video element refreshed")
            
        except Exception as e:
            print(f"❌ Failed to refresh video element: {e}")
    
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
        
        if self.use_gcs:
            print(f"  ☁️ GCS uploads: {self.stats['gcs_uploads']}")
            if self.stats['gcs_failures'] > 0:
                print(f"  ⚠️ GCS failures: {self.stats['gcs_failures']}")
        
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
        
        if self.use_gcs:
            print(f"☁️ Saving to Google Cloud Storage")
            print(f"☁️ Bucket: {self.gcs_bucket_name}")
            print(f"☁️ Camera: {self.camera_info['name'] if self.camera_info else 'Unknown'}")
        else:
            print(f"💾 Saving locally to: {self.base_dir}/YEAR/MONTH/DAY/HOUR/")
        
        if self.headless:
            print(f"👻 Running in HEADLESS mode - enhanced video monitoring active")
            check_interval = 0.1  # Slightly slower in headless to ensure stability
        
        print(f"💡 Now saving individual cropped boat images (not full frames)")
        
        start_time = time.time()
        attempt_count = 0
        successful_frames = 0
        last_stats_time = start_time
        last_refresh_time = start_time  # For periodic page refresh in headless
        
        try:
            while True:  # Simple infinite loop
                current_time = time.time()
                
                # In headless mode, refresh page every 10 minutes to prevent throttling
                if self.headless and (current_time - last_refresh_time) >= 600:
                    print("🔄 Performing periodic page refresh (headless mode)...")
                    self.driver.refresh()
                    time.sleep(5)
                    if self.load_youtube_video():
                        last_refresh_time = current_time
                        print("✅ Page refreshed successfully")
                    else:
                        print("⚠️ Page refresh failed, continuing anyway")
                
                success = self.process_frame(current_time, attempt_count)
                
                if success:
                    successful_frames += 1
                    if not self.headless or successful_frames % 10 == 0:  # Less verbose in headless
                        print(f"📊 Frame {successful_frames} | Saved: {self.stats['images_saved']} | Boats: {len(self.stats['active_boat_ids'])}")
                
                attempt_count += 1
                
                # Stats every 30 seconds
                if current_time - last_stats_time >= 30:
                    self.print_statistics(successful_frames)
                    last_stats_time = current_time
                    
                    # In headless, also check video health
                    if self.headless:
                        video_health = self.check_video_health()
                        if not video_health:
                            print("⚠️ Video health check failed, attempting recovery...")
                            self.refresh_video_element()
                
                time.sleep(check_interval)
                
                # Keep video active more frequently in headless mode
                if self.headless and attempt_count % 20 == 0:
                    self.keep_video_active_headless()
                elif not self.headless and attempt_count % 50 == 0:
                    self.keep_video_active()
        
        except KeyboardInterrupt:
            print(f"\n🛑 Stopped after {(time.time() - start_time)/60:.1f} minutes")
            self.print_statistics(successful_frames)
    
    def check_video_health(self):
        """Check if video is healthy (for headless mode monitoring)"""
        try:
            health = self.driver.execute_script("""
                var video = document.querySelector('video.html5-main-video');
                if (!video) return null;
                return {
                    hasVideo: true,
                    currentTime: video.currentTime,
                    duration: video.duration,
                    paused: video.paused,
                    ended: video.ended,
                    readyState: video.readyState,
                    error: video.error
                };
            """)
            
            if not health or not health['hasVideo']:
                print("❌ Video element not found")
                return False
            
            if health['error']:
                print(f"❌ Video error detected")
                return False
            
            if health['ended']:
                print("⚠️ Video ended")
                return False
            
            if health['readyState'] < 2:
                print("⚠️ Video not ready")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Video health check failed: {e}")
            return False
    
    def keep_video_active_headless(self):
        """Keep video active in headless mode without mouse movements"""
        try:
            # Use JavaScript to interact with video
            self.driver.execute_script("""
                var video = document.querySelector('video.html5-main-video');
                if (video) {
                    // Slightly adjust current time to trigger activity
                    if (video.currentTime > 1) {
                        video.currentTime = video.currentTime + 0.01;
                    }
                    // Ensure still playing
                    if (video.paused) {
                        video.play();
                    }
                }
            """)
        except Exception as e:
            print(f"⚠️ Could not keep video active (headless): {e}")
            
    def cleanup(self):
        """Cleanup resources"""
        if self.driver:
            self.driver.quit()
            print("✅ Browser closed")
        
        if self.use_gcs and self.gcs_client:
            self.gcs_client.close()
            print("☁️ GCS client closed")