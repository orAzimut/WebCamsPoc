# YouTube Boat Detection POC

This is a proof-of-concept project for intelligent boat detection from YouTube live streams using YOLO11 and SORT tracking.

## Project Structure

```
webCams/POC/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── kalman_tracker.py        # KalmanBoxTracker class - SORT individual tracking
│   ├── sort_tracker.py          # Sort class - SORT multi-object tracking
│   ├── diversity_scorer.py      # DiversityScorer class - diversity scoring
│   ├── yolo_detector.py         # YOLOBoatDetector class - YOLO11 detection
│   └── youtube_scraper.py      # IntelligentYouTubeBoatScraper class - main scraper
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
├── README.md                   # This file
└── ItaysPoc.py                 # Original monolithic file (for reference)
```

## Classes Overview

### 1. KalmanBoxTracker (`src/kalman_tracker.py`)
- SORT individual object tracker using Kalman filtering
- Tracks 7 states: [u, v, s, r, u', v', s'] (center, scale, aspect ratio + velocities)
- Provides predict() and update() methods for SORT integration

### 2. Sort (`src/sort_tracker.py`)
- SORT (Simple Online and Realtime Tracking) algorithm implementation
- Manages multiple KalmanBoxTracker instances
- Uses Hungarian algorithm for optimal assignment
- Handles track creation, update, and deletion

### 3. DiversityScorer (`src/diversity_scorer.py`)
- Handles diversity scoring for tracked objects
- Prevents saving too many similar images of the same boat
- Uses time, position, and size changes to calculate diversity scores

### 4. YOLOBoatDetector (`src/yolo_detector.py`)
- Handles YOLO11 model initialization and boat detection
- Detects boats/ships in video frames
- Returns bounding boxes with confidence scores

### 5. IntelligentYouTubeBoatScraper (`src/youtube_scraper.py`)
- Main scraper class that orchestrates everything
- Handles YouTube video loading and browser automation
- Processes frames, detects boats, and saves images with metadata

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have Chrome browser installed

## Usage

Run the main script:
```bash
python main.py
```

The script will:
1. Ask for a YouTube URL
2. Initialize YOLO11 model
3. Setup Chrome browser
4. Load the YouTube video
5. Present scraping options (time-based or saved images-based)

## Features

- **Intelligent Boat Detection**: Uses YOLO11 to detect boats/ships
- **SORT Multi-Object Tracking**: Advanced tracking with Kalman filtering
- **Diversity Scoring**: Prevents saving too many similar images
- **Organized Output**: Files saved in YEAR/MONTH/DAY/HOUR structure
- **Metadata**: Each image comes with detailed JSON metadata
- **Statistics**: Real-time progress and statistics
- **High Performance**: Optimized for ~5 FPS with YOLO11

## Output Structure

Images and metadata are saved in:
```
POC/
├── 2025/
│   ├── 08/
│   │   ├── 03/
│   │   │   ├── 17/
│   │   │   │   ├── boat_001_20250803_172600_707_0000.jpg
│   │   │   │   ├── boat_001_20250803_172600_707_0000.json
│   │   │   │   └── ...
```

## Configuration

Key parameters can be adjusted in the respective classes:
- **DiversityScorer**: Diversity thresholds, time intervals
- **YOLOBoatDetector**: Confidence thresholds, model size
- **Sort**: IOU threshold, max age, min hits
- **IntelligentYouTubeBoatScraper**: Check intervals, save limits
