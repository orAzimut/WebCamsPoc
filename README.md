# 🚢 Intelligent YouTube Boat Detection Scraper

An advanced AI-powered system that automatically detects and captures diverse boat images from YouTube webcam streams for ReID (Re-Identification) model training. Uses YOLO11 for real-time boat detection with intelligent tracking and diversity algorithms to ensure high-quality training data.

## 🎯 **Purpose**

This tool is specifically designed for collecting **diverse boat images** for ReID model training by:
- Automatically detecting boats in YouTube webcam streams
- Tracking individual boats with persistent IDs
- Ensuring diversity in captured images (different angles, positions, times)
- Organizing data in a structured, time-based hierarchy
- Providing rich metadata for each captured image

## ✨ **Key Features**

### 🤖 **Intelligent Detection**
- **YOLO11 Integration**: Real-time boat/ship detection with high accuracy
- **Multi-Boat Tracking**: Assigns persistent IDs to individual boats
- **Smart Diversity Algorithm**: Captures varied angles and positions of the same boat
- **Location Recognition**: Automatically detects camera location from video content

### 🎬 **YouTube Integration**
- **Seamless Video Handling**: Automatically manages video playback and controls
- **Anti-Pause System**: Prevents YouTube from auto-pausing during collection
- **Theater Mode**: Optimizes video quality for better detection
- **Popup Dismissal**: Handles YouTube consent dialogs and notifications

### 📊 **Data Management**
- **Time-Based Organization**: Files organized by Year/Month/Day/Hour
- **Rich Metadata**: JSON files with bounding boxes, confidence, location data
- **Batch Processing**: Supports both time-based and target-based collection
- **Real-Time Statistics**: Live progress tracking and performance metrics

## 🛠️ **Requirements**

### **Python Dependencies**
```bash
pip install ultralytics selenium webdriver-manager opencv-python numpy
```

### **System Requirements**
- **Python 3.8+**
- **Chrome Browser** (latest version)
- **Windows/Linux/macOS**
- **GPU recommended** (for faster YOLO inference)

### **Hardware Recommendations**
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 10GB+ free space for image collection
- **Internet**: Stable connection for YouTube streaming

## 📦 **Installation**

1. **Clone/Download** the script file
2. **Install dependencies**:
   ```bash
   pip install ultralytics selenium webdriver-manager opencv-python numpy
   ```
3. **Verify Chrome** is installed and updated
4. **Run the script**:
   ```bash
   python boat_detector.py
   ```

## 🚀 **Usage**

### **Basic Usage**
```bash
python boat_detector.py
```

### **Interactive Setup**
1. **Enter YouTube URL** of webcam stream
2. **Choose collection mode**:
   - Time-based (collect for X minutes)
   - Image-based (collect X boat images)
3. **System automatically**:
   - Initializes YOLO11 model
   - Opens browser and loads video
   - Detects location from video content
   - Starts intelligent collection

### **Collection Modes**

#### **Time-Based Collection**
```
=== TIME-BASED OPTIONS ===
1. Quick test (5 minutes)
2. Medium session (15 minutes)  
3. Long session (30 minutes)
4. Custom time duration
```

#### **Image-Based Collection** 
```
=== SAVED BOAT IMAGES OPTIONS ===
5. Quick test (10 saved boat images)
6. Medium session (50 saved boat images)
7. Long session (100 saved boat images)  
8. Custom saved boat images count
```

## 📁 **Output Structure**

### **File Organization**
```
POC/
├── 2025/
│   └── 07/
│       └── 31/
│           ├── 10/  # 10 AM hour
│           │   ├── boat_001_20250731_103045_123_0001.jpg
│           │   ├── boat_001_20250731_103045_123_0001.json
│           │   ├── boat_002_20250731_103127_456_0002.jpg
│           │   └── boat_002_20250731_103127_456_0002.json
│           └── 11/  # 11 AM hour
│               ├── boat_001_20250731_110234_789_0003.jpg
│               └── boat_001_20250731_110234_789_0003.json
```

### **Filename Format**
```
boat_{ID:03d}_{timestamp}_{sequence:04d}.{jpg|json}
```
- **ID**: Persistent boat tracker ID (001, 002, etc.)
- **Timestamp**: YearMonthDay_HourMinuteSecond_Milliseconds
- **Sequence**: Global image counter for this run

### **JSON Metadata Example**
```json
{
  "timestamp": "20250731_103045_123",
  "youtube_url": "https://youtu.be/wBVq_Qoegmo",
  "camera_location": "hamburg",
  "boat_id": 1,
  "confidence": 0.847,
  "class": "boat",
  "bbox": {
    "x1": 245, "y1": 123, "x2": 398, "y2": 267,
    "width": 153, "height": 144
  },
  "frame_info": {
    "frame_width": 1280, "frame_height": 720,
    "total_frames_processed": 45,
    "images_saved_this_run": 12
  },
  "datetime_path": {
    "year": "2025", "month": "07", "day": "31", "hour": "10",
    "full_path": "C:/path/to/POC/2025/07/31/10"
  }
}
```

## 🧠 **Algorithm Details**

### **Diversity Scoring System**
The system ensures diverse training data through multi-criteria scoring:

#### **Time Component (40% weight)**
- Minimum 5 seconds between saves per boat
- Longer gaps = higher diversity score

#### **Position Component (40% weight)**  
- Tracks bounding box centroid movement
- Boats must move >50 pixels for high score
- Ensures different viewing angles

#### **Size Component (20% weight)**
- Monitors bounding box area changes  
- Size changes >15% indicate distance/angle variation
- Captures boats at different scales

#### **Tracking Algorithm**
1. **Detection**: YOLO11 identifies boats in each frame
2. **Assignment**: Matches detections to existing trackers via centroid distance
3. **Scoring**: Calculates diversity score vs. last saved image
4. **Decision**: Saves image if score exceeds threshold (0.3)
5. **Update**: Updates tracker with new detection data

### **Location Detection**  
Automatically identifies camera location from:
- **Video titles**: "Hamburg Harbor Live Stream"
- **Descriptions**: Text mentioning city/port names
- **URLs**: Contains location keywords
- **Supports 25+ major ports**: Hamburg, NYC, Miami, London, Singapore, etc.

## 📊 **Live Statistics**

### **Real-Time Display**
```
📊 STATISTICS:
  Frames processed: 45
  Saved images: 12/50
  Progress: 24.0%
  Total boat detections: 23
  Active boat IDs: 3
  Current trackers: 2
  Camera location: hamburg

🚢 BOAT ID DISTRIBUTION:
  Boat 001: 5 images
  Boat 002: 4 images  
  Boat 003: 3 images
```

### **Final Summary**
```
📁 FILES ORGANIZED BY DATE/TIME:
  2025/07/31/10h: 8 images, 8 JSON files
  2025/07/31/11h: 4 images, 4 JSON files

Total files saved: 24 (12 images + 12 JSON files)
Success rate: 87.5%
```

## ⚙️ **Configuration**

### **Adjustable Parameters**
```python
# In BoatTracker class
self.min_time_interval = 5      # Seconds between saves per boat
self.min_position_change = 50   # Pixels movement threshold  
self.min_size_change_ratio = 0.15  # 15% size change threshold
self.max_saves = 100           # Max images per boat ID

# In YOLOBoatDetector class  
confidence_threshold = 0.3     # Minimum detection confidence
```

### **Output Path**
```python
self.base_dir = r"C:\Users\OrGil.AzureAD\OneDrive - AMPC\Desktop\Azimut.ai\webScrape\webCams\POC"
```

## 🔧 **Troubleshooting**

### **Common Issues**

#### **"No boats detected"**
- Check if video contains visible boats/ships
- Try different YouTube webcam URLs
- Verify YOLO11 model downloaded correctly

#### **"Video keeps pausing"**
- System includes anti-pause mechanisms
- Check internet connection stability
- Try different browser/video quality

#### **"YOLO model not found"**
```bash
# Manually download model
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

#### **"ChromeDriver issues"**
- System auto-downloads ChromeDriver
- Ensure Chrome browser is updated
- Check firewall/antivirus blocking downloads

### **Performance Optimization**

#### **For Speed**
- Use `yolo11n.pt` (nano model)
- Increase check_interval (3-5 seconds)
- Reduce image quality/resolution

#### **For Accuracy**  
- Use `yolo11m.pt` or `yolo11l.pt`
- Decrease check_interval (1-2 seconds)
- Use GPU acceleration

## 📈 **Best Practices**

### **For ReID Training Data**
1. **Use diverse webcam locations** (different ports/harbors)
2. **Collect across different times** (morning/afternoon/evening)
3. **Target 50-100 images per boat** for good diversity
4. **Monitor boat ID distribution** to ensure balanced dataset
5. **Verify bounding box quality** in saved JSON metadata

### **For Long Collection Sessions**
1. **Use stable internet connection**
2. **Monitor disk space** (images can accumulate quickly)
3. **Run during off-peak hours** for better YouTube stability
4. **Use time-based mode** for predictable duration

### **For Multiple Locations**
1. **Run separate sessions** for different webcam locations
2. **Location data automatically saved** in JSON metadata
3. **Combine datasets** from different ports for robust training

## 🌍 **Supported Locations**

The system automatically detects these major ports/harbors:
- **Hamburg** (Germany)
- **New York City** (USA) 
- **Miami** (USA)
- **London/Thames** (UK)
- **Singapore** (Singapore)
- **Sydney** (Australia)
- **San Francisco** (USA)
- **Vancouver** (Canada)
- **Amsterdam** (Netherlands)
- **Venice** (Italy)
- **And 15+ more major ports**

## 🤝 **Contributing**

To improve the system:
1. **Add new location keywords** in `LocationExtractor` class
2. **Adjust diversity thresholds** for different use cases  
3. **Enhance YOLO detection** for specific boat types
4. **Optimize performance** for different hardware configurations

## 📄 **License**

This tool is designed for research and educational purposes. Ensure compliance with YouTube's Terms of Service and respect copyright/privacy when collecting data from public webcam streams.

## 🆘 **Support**

For issues or questions:
1. Check **Troubleshooting** section above
2. Verify all **Requirements** are installed
3. Test with **Quick test** mode first
4. Review **console output** for specific error messages

---

**Happy boat collecting! 🚢📸**
