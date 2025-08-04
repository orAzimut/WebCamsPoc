import os
from ultralytics import YOLO

class YOLOBoatDetector:
    """Handles YOLO11 boat detection with built-in tracking"""
    
    def __init__(self):
        self.model = None
        self.boat_classes = ['boat', 'ship']  # COCO class names for boats
        
    def initialize_model(self):
        """Initialize YOLO11 model using existing file"""
        try:
            print("🔥 Loading YOLO11 model...")
            
            # Use your existing model file
            model_path = r"C:\Users\OrGil.AzureAD\OneDrive - AMPC\Desktop\Azimut.ai\webScrape\webCams\POC\yolo11n.pt"
            
            # Check if file exists
            if not os.path.exists(model_path):
                print(f"✗ Model file not found at: {model_path}")
                # Fallback to auto-download
                print("Trying auto-download instead...")
                self.model = YOLO('yolo11n.pt')  # This will download automatically
            else:
                print(f"✓ Found existing model at: {model_path}")
                self.model = YOLO(model_path)
            
            # Move model to GPU if available
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                self.model.to(device)
                print(f"✓ YOLO11 model loaded on GPU: {torch.cuda.get_device_name(0)}")
                print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                device = 'cpu'
                print("⚠ CUDA not available, using CPU")
                
            self.device = device
            print("✓ YOLO11 model loaded successfully")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load YOLO11 model: {e}")
            print("Make sure you have ultralytics installed: pip install ultralytics")
            return False
    
    def track_boats(self, image):
        """Track boats in image using YOLO's built-in tracking and return tracked boats"""
        try:
            if self.model is None:
                return []
            
            print(f"🔍 Tracking with device: {self.device}")
            
            # Use YOLO's built-in tracking with optimized parameters
            results = self.model.track(
                image, 
                persist=True, 
                verbose=False, 
                device=self.device,
                imgsz=640,          # Smaller size for speed
                conf=0.15,           # Higher confidence threshold
                iou=0.5,
                max_det=8,          # Fewer detections for speed
                half=True,          # FP16 precision
                agnostic_nms=True,  # Faster NMS
                augment=False,      # No augmentation for speed
                tracker="bytetrack.yaml"
            )
            
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
                            
                            # Get tracking ID (this is the key difference - YOLO provides the ID)
                            track_id = None
                            if hasattr(box, 'id') and box.id is not None:
                                track_id = int(box.id[0])
                            
                            boats.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                                'class': class_name,
                                'track_id': track_id  # This is provided by YOLO tracking
                            })
            
            print(f"🚢 Found {len(boats)} boats with tracking IDs")
            for boat in boats:
                print(f"  Boat ID: {boat['track_id']}, Confidence: {boat['confidence']:.3f}")
            
            return boats
            
        except Exception as e:
            print(f"✗ Error in boat tracking: {e}")
            return []