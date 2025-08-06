import yaml
import os
from youtube_scraper import IntelligentYouTubeBoatScraper

def main():
    print("🚢 INTELLIGENT YOUTUBE BOAT DETECTION SCRAPER (YOLO TRACKING + GCS)")
    print("=" * 60)
    
    # Load configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "..", "resources", "config.yaml")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error reading config.yaml: {e}")
        return
    
    # Extract configuration
    youtube_url = config['youtube_url'].strip()
    headless = config.get('headless', False)
    use_gcs = config.get('use_gcs', True)  # Default to GCS if not specified
    
    print(f"📺 YouTube URL: {youtube_url}")
    
    if headless:
        print("🔧 Running in background mode (invisible browser)")
    else:
        print("🔧 Running with visible browser")
    
    if use_gcs:
        print("☁️ Storage mode: Google Cloud Storage")
    else:
        print("💾 Storage mode: Local filesystem")
    
    # Create scraper instance with GCS support
    scraper = IntelligentYouTubeBoatScraper(
        youtube_url=youtube_url,
        headless=headless,
        use_gcs=use_gcs
    )
    
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
        
        if use_gcs:
            print(f"☁️ Images will be saved to GCS: reidentification/bronze/raw_crops/webCams/")
            print(f"☁️ JSONs will be saved to GCS: reidentification/bronze/json_lables/webCams/")
        else:
            print(f"📁 Images will be saved locally to: {scraper.base_dir}")
        
        print("📊 Statistics will be shown every 30 seconds")
        print("-" * 60)
        
        # Run continuous scraping
        scraper.run_continuous_scraping()

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user (Ctrl+C)")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n🧹 Cleaning up...")
        scraper.cleanup()
        print("✅ Done!")

if __name__ == "__main__":
    main()