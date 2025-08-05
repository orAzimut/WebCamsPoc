import yaml
import os
from youtube_scraper import IntelligentYouTubeBoatScraper

def main():
    print("🚢 INTELLIGENT YOUTUBE BOAT DETECTION SCRAPER (YOLO TRACKING)")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "..", "resources", "config.yaml")
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error reading config.yaml: {e}")
        return
    
    youtube_url = config['youtube_url'].strip()
    headless = config['headless']
    
    print(f"📺 YouTube URL: {youtube_url}")
    if headless:
        print("🔧 Running in background mode (invisible browser)")
    else:
        print("🔧 Running with visible browser")
    
    # Create scraper instance
    scraper = IntelligentYouTubeBoatScraper(youtube_url, headless=headless)
    
    try:
        # Initialize YOLO model
        print("\n🤖 Initializing YOLO model...")
        if not scraper.yolo_detector.initialize_model():
            print("❌ Failed to initialize YOLO model")
            return
        
        # Setup browser
        print("🌐 Setting up browser...")
        if not scraper.setup_driver():
            print("❌ Failed to setup browser")
            return
        
        # Load YouTube video
        print("📺 Loading YouTube video...")
        if not scraper.load_youtube_video():
            print("❌ Failed to load video")
            return
        
        print("\n✅ System ready!")
        print("🚀 Starting continuous boat detection...")
        print("💡 Press Ctrl+C to stop at any time")
        print("📁 Images will be saved to: {}".format(scraper.base_dir))
        print("📊 Statistics will be shown every 30 seconds")
        print("-" * 60)
        
        # Run with very large duration (effectively forever until Ctrl+C)
        scraper.run_continuous_scraping()

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user (Ctrl+C)")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        
    finally:
        print("\n🧹 Cleaning up...")
        scraper.cleanup()
        print("✅ Done!")

if __name__ == "__main__":
    main()