#!/usr/bin/env python3
"""
Diagnostic script to test and debug headless mode issues
Run this to verify headless mode is working correctly
"""

import time
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def test_headless_video_capture(youtube_url, test_duration=60):
    """Test video capture in headless mode"""
    
    print("🧪 HEADLESS MODE DIAGNOSTIC TEST")
    print("=" * 60)
    
    # Setup headless Chrome
    chrome_options = Options()
    
    # Enhanced headless configuration
    chrome_options.add_argument("--headless=new")  # New headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-software-rasterizer")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--use-fake-ui-for-media-stream")
    chrome_options.add_argument("--use-fake-device-for-media-stream")
    chrome_options.add_argument("--autoplay-policy=no-user-gesture-required")
    chrome_options.add_argument("--disable-background-timer-throttling")
    chrome_options.add_argument("--disable-backgrounding-occluded-windows")
    chrome_options.add_argument("--disable-renderer-backgrounding")
    chrome_options.add_argument("--disable-features=TranslateUI")
    chrome_options.add_argument("--disable-ipc-flooding-protection")
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
    
    driver = None
    
    try:
        print("🌐 Starting headless Chrome...")
        driver = webdriver.Chrome(options=chrome_options)
        
        print(f"📺 Loading YouTube: {youtube_url}")
        driver.get(youtube_url)
        
        # Wait for video element
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "video.html5-main-video"))
        )
        time.sleep(5)
        
        video_element = driver.find_element(By.CSS_SELECTOR, "video.html5-main-video")
        
        # Force video to play and mute
        driver.execute_script("""
            var video = document.querySelector('video.html5-main-video');
            if (video) {
                video.muted = true;
                video.volume = 0;
                video.play();
                console.log('Video forced to play');
            }
        """)
        
        print("✅ Video loaded")
        print(f"⏱️ Running {test_duration} second test...")
        print("-" * 40)
        
        start_time = time.time()
        frame_count = 0
        successful_captures = 0
        failed_captures = 0
        black_frames = 0
        
        while time.time() - start_time < test_duration:
            frame_count += 1
            
            # Check video status
            video_status = driver.execute_script("""
                var video = document.querySelector('video.html5-main-video');
                return {
                    exists: !!video,
                    paused: video ? video.paused : null,
                    currentTime: video ? video.currentTime : null,
                    duration: video ? video.duration : null,
                    readyState: video ? video.readyState : null
                };
            """)
            
            # Try to capture frame
            try:
                # Method 1: Direct element screenshot
                video_png = video_element.screenshot_as_png
                nparr = np.frombuffer(video_png, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Check if frame is not black
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    mean_brightness = np.mean(gray)
                    
                    if mean_brightness < 10:
                        black_frames += 1
                        print(f"⚫ Frame {frame_count}: Black frame detected")
                    else:
                        successful_captures += 1
                        if successful_captures % 10 == 0:
                            print(f"✅ Frame {frame_count}: Captured successfully (brightness: {mean_brightness:.1f})")
                else:
                    failed_captures += 1
                    print(f"❌ Frame {frame_count}: Failed to decode")
                    
            except Exception as e:
                failed_captures += 1
                print(f"❌ Frame {frame_count}: Capture failed - {e}")
            
            # If video paused, try to resume
            if video_status['paused']:
                print(f"⏸️ Video paused at frame {frame_count}, resuming...")
                driver.execute_script("document.querySelector('video.html5-main-video').play();")
            
            # Every 10 frames, force play
            if frame_count % 10 == 0:
                driver.execute_script("""
                    var video = document.querySelector('video.html5-main-video');
                    if (video && video.paused) {
                        video.play();
                    }
                """)
            
            time.sleep(0.5)  # Wait between captures
        
        # Print results
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS:")
        print(f"  Total frames attempted: {frame_count}")
        print(f"  ✅ Successful captures: {successful_captures}")
        print(f"  ❌ Failed captures: {failed_captures}")
        print(f"  ⚫ Black frames: {black_frames}")
        
        success_rate = (successful_captures / frame_count) * 100 if frame_count > 0 else 0
        print(f"  📈 Success rate: {success_rate:.1f}%")
        
        if success_rate > 80:
            print("\n✅ HEADLESS MODE WORKING WELL")
        elif success_rate > 50:
            print("\n⚠️ HEADLESS MODE PARTIALLY WORKING")
        else:
            print("\n❌ HEADLESS MODE HAS ISSUES")
            
            print("\n🔧 Troubleshooting suggestions:")
            print("  1. Update Chrome and ChromeDriver")
            print("  2. Try '--headless=new' instead of '--headless'")
            print("  3. Increase window size to 1920x1080")
            print("  4. Add more video-specific flags")
            print("  5. Consider using Xvfb virtual display")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if driver:
            driver.quit()
            print("\n🧹 Browser closed")

def main():
    # Test URL - Rotterdam Port
    test_url = "https://www.youtube.com/watch?v=_KVWehizoNU"
    
    print("Select test mode:")
    print("1. Quick test (30 seconds)")
    print("2. Standard test (60 seconds)")
    print("3. Extended test (120 seconds)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    durations = {"1": 30, "2": 60, "3": 120}
    duration = durations.get(choice, 60)
    
    print(f"\n🚀 Starting {duration} second headless test...")
    test_headless_video_capture(test_url, duration)

if __name__ == "__main__":
    main()