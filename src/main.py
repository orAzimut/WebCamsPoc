from youtube_scraper import IntelligentYouTubeBoatScraper

def main():
    print("🚢 INTELLIGENT YOUTUBE BOAT DETECTION SCRAPER (YOLO TRACKING)")
    print("=" * 60)
    
    # Get YouTube URL
    youtube_url = input("Enter YouTube URL: ").strip()
    if not youtube_url:
        print("❌ No URL provided")
        return
    
    # Ask about headless mode
    headless_choice = input("\nRun in background (invisible browser)? (y/n): ").strip().lower()
    headless = headless_choice == 'y'
    
    # Create scraper instance
    scraper = IntelligentYouTubeBoatScraper(youtube_url, headless=headless)
    
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