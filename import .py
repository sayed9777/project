import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import time
from ultralytics import YOLO
import os
import warnings
warnings.filterwarnings('ignore')

class ViolenceDetectionSystem:
    def __init__(self):
        # Initialize parameters
        self.confidence_threshold = 0.65
        self.frame_buffer = deque(maxlen=16)
        self.violence_detected = False
        self.alert_cooldown = 0
        self.alert_cooldown_time = 30  # frames
        
        # Load models
        self.yolo_model = self.load_yolo_model()
        self.violence_model = self.load_violence_model()
        
    def load_yolo_model(self):
        """Load YOLO model for human detection"""
        try:
            model = YOLO("yolov8n.pt")  # Using newer YOLOv8
            print("✅ YOLO model loaded successfully")
            return model
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            return None
    
    def load_violence_model(self):
        """Load or create violence detection model"""
        try:
            # Try to load pre-trained model
            model_path = "violence_model.h5"
            if os.path.exists(model_path):
                model = keras.models.load_model(model_path)
                print("✅ Pre-trained violence model loaded")
            else:
                model = self.create_violence_model()
                print("⚠️ Using untrained model - consider training on violence dataset")
            return model
        except Exception as e:
            print(f"❌ Error loading violence model: {e}")
            return self.create_violence_model()
    
    def create_violence_model(self):
        """Create a more efficient violence detection model"""
        model = keras.Sequential([
            # Feature extraction with Conv3D for spatiotemporal features
            keras.layers.Conv3D(32, (3, 3, 3), activation='relu', 
                               input_shape=(16, 112, 112, 3)),
            keras.layers.MaxPooling3D((1, 2, 2)),
            keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
            keras.layers.MaxPooling3D((1, 2, 2)),
            keras.layers.Conv3D(128, (3, 3, 3), activation='relu'),
            keras.layers.MaxPooling3D((1, 2, 2)),
            
            # Temporal processing
            keras.layers.Reshape((16, -1)),
            keras.layers.LSTM(64, return_sequences=False),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def preprocess_frame(self, frame):
        """Enhanced frame preprocessing"""
        # Resize to smaller dimensions for faster processing
        frame = cv2.resize(frame, (112, 112))
        
        # Normalize pixel values
        frame = frame.astype('float32') / 255.0
        
        # Apply slight contrast enhancement
        frame = np.clip(frame * 1.1, 0, 1)
        
        return frame

    def detect_humans(self, frame):
        """Enhanced human detection with YOLO"""
        if self.yolo_model is None:
            return []
            
        try:
            results = self.yolo_model(frame, verbose=False)
            human_boxes = []
            
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    # Class 0 is person in YOLO
                    if cls == 0 and conf > 0.5:  # Increased confidence threshold
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        human_boxes.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': conf
                        })
            
            return human_boxes
        except Exception as e:
            print(f"Error in human detection: {e}")
            return []

    def calculate_optical_flow(self, prev_frame, curr_frame):
        """Calculate optical flow between frames"""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        return flow

    def analyze_movement_patterns(self, frame_sequence):
        """Enhanced movement analysis with multiple indicators"""
        violence_indicators = {
            'rapid_movement': False,
            'aggressive_gestures': False,
            'close_proximity': False,
            'falling_motions': False,
            'movement_score': 0.0
        }
        
        if len(frame_sequence) < 2:
            return violence_indicators
        
        try:
            # Calculate optical flow for the latest frames
            flow = self.calculate_optical_flow(frame_sequence[-2], frame_sequence[-1])
            
            # Calculate movement magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            avg_magnitude = np.mean(magnitude)
            
            # Calculate movement variance (indicates erratic movement)
            movement_variance = np.var(magnitude)
            
            # Update indicators based on thresholds
            violence_indicators['movement_score'] = min(avg_magnitude / 10.0, 1.0)
            
            if avg_magnitude > 4.0:
                violence_indicators['rapid_movement'] = True
            
            if movement_variance > 15.0:  # High variance indicates erratic movement
                violence_indicators['aggressive_gestures'] = True
            
            return violence_indicators
            
        except Exception as e:
            print(f"Error in movement analysis: {e}")
            return violence_indicators

    def detect_camera_shake(self, frame_sequence):
        """Improved camera shake detection"""
        if len(frame_sequence) < 2:
            return False
        
        try:
            flow = self.calculate_optical_flow(frame_sequence[-2], frame_sequence[-1])
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            
            # Calculate global movement (camera shake)
            global_movement = np.mean(magnitude)
            
            # Camera shake typically affects the entire frame uniformly
            movement_std = np.std(magnitude)
            
            # High global movement with low variance indicates camera shake
            return global_movement > 15.0 and movement_std < 5.0
            
        except Exception as e:
            print(f"Error in camera shake detection: {e}")
            return False

    def predict_violence_heuristic(self, movement_analysis, human_count):
        """Heuristic violence prediction based on movement patterns"""
        score = 0.0
        
        # Base score from movement
        score += movement_analysis['movement_score'] * 0.6
        
        # Bonus for multiple indicators
        if movement_analysis['rapid_movement']:
            score += 0.2
        if movement_analysis['aggressive_gestures']:
            score += 0.3
        if human_count >= 2:  # Multiple people increases violence likelihood
            score += 0.1
            
        return min(score, 1.0)

    def predict_violence_model(self, frame_sequence):
        """Model-based violence prediction"""
        if len(frame_sequence) < 16 or self.violence_model is None:
            return 0.0
        
        try:
            # Prepare sequence for model
            processed_sequence = np.array([self.preprocess_frame(f) for f in frame_sequence])
            processed_sequence = np.expand_dims(processed_sequence, axis=0)
            
            # Predict
            prediction = self.violence_model.predict(processed_sequence, verbose=0)
            return float(prediction[0][0])
            
        except Exception as e:
            print(f"Error in model prediction: {e}")
            return 0.0

    def combined_violence_prediction(self, frame_sequence, human_boxes):
        """Combine multiple methods for robust violence detection"""
        movement_analysis = self.analyze_movement_patterns(frame_sequence)
        human_count = len(human_boxes)
        
        # Get predictions from different methods
        heuristic_score = self.predict_violence_heuristic(movement_analysis, human_count)
        model_score = self.predict_violence_model(frame_sequence)
        
        # Combine scores (weight heuristic more if model is untrained)
        final_score = (heuristic_score * 0.7 + model_score * 0.3)
        
        return final_score, movement_analysis

    def draw_detection_info(self, frame, violence_score, human_boxes, movement_analysis):
        """Enhanced visualization of detection information"""
        # Status overlay
        status_text = f"Violence Score: {violence_score:.2f}"
        status_color = (0, 255, 0)  # Green
        alert_level = "LOW"
        
        if violence_score > 0.7:
            status_color = (0, 0, 255)  # Red
            alert_level = "HIGH"
        elif violence_score > 0.4:
            status_color = (0, 165, 255)  # Orange
            alert_level = "MEDIUM"
        
        # Draw status bar
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (50, 50, 50), -1)
        cv2.putText(frame, f"Status: {alert_level} - {status_text}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Draw human bounding boxes
        for human in human_boxes:
            x1, y1, x2, y2 = human['bbox']
            conf = human['confidence']
            
            box_color = (0, 255, 0)  # Green for normal
            if violence_score > 0.4:
                box_color = (0, 165, 255)  # Orange for medium alert
            if violence_score > 0.7:
                box_color = (0, 0, 255)  # Red for high alert
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"Human: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
        
        # Draw movement indicators
        y_offset = 60
        for indicator, value in movement_analysis.items():
            if isinstance(value, bool):
                color = (0, 255, 0) if not value else (0, 0, 255)
                text = f"{indicator}: {'Yes' if value else 'No'}"
                cv2.putText(frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_offset += 20
        
        return frame

    def trigger_alert(self, violence_score, alert_type="violence"):
        """Enhanced alert system with cooldown"""
        if self.alert_cooldown > 0:
            self.alert_cooldown -= 1
            return
            
        if alert_type == "violence":
            print(f"🚨 VIOLENCE ALERT! Confidence: {violence_score:.2%}")
        elif alert_type == "camera_shake":
            print(f"⚠️ CAMERA DISTURBANCE DETECTED!")
            
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 50)
        
        self.alert_cooldown = self.alert_cooldown_time

    def process_video_stream(self, camera_source=0):
        """Main video processing loop with enhanced detection"""
        cap = cv2.VideoCapture(camera_source)
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            print("❌ Error: Unable to open video source.")
            return
        
        print("🎥 Starting Enhanced Violence Detection System...")
        print("Press 'q' to quit, 'r' to reset alerts")
        
        frame_count = 0
        fps_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to capture frame.")
                break
            
            frame_count += 1
            
            # Reduce processing load by skipping some frames
            if frame_count % 2 != 0:
                continue
                
            # Maintain frame buffer for temporal analysis
            self.frame_buffer.append(frame.copy())
            
            # Detect humans in the frame
            human_boxes = self.detect_humans(frame)
            
            # Check for camera shake
            camera_shake = self.detect_camera_shake(list(self.frame_buffer))
            if camera_shake:
                self.trigger_alert(0.0, "camera_shake")
                cv2.putText(frame, "CAMERA DISTURBANCE!", (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # Perform violence detection if we have enough frames
            if len(self.frame_buffer) >= 8:  # Reduced from 16 for faster response
                violence_score, movement_analysis = self.combined_violence_prediction(
                    list(self.frame_buffer), human_boxes
                )
                
                # Update display with detection information
                frame = self.draw_detection_info(frame, violence_score, human_boxes, movement_analysis)
                
                # Trigger alerts based on violence score
                if violence_score > self.confidence_threshold:
                    self.violence_detected = True
                    self.trigger_alert(violence_score, "violence")
                    
                    # Draw prominent alert
                    cv2.putText(frame, "VIOLENCE DETECTED!", 
                               (frame.shape[1]//2 - 100, frame.shape[0]//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                else:
                    self.violence_detected = False
            
            # Calculate and display FPS
            current_time = time.time()
            fps = 1.0 / (current_time - fps_time) if current_time - fps_time > 0 else 0
            fps_time = current_time
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Enhanced Violence Detection System', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.violence_detected = False
                self.alert_cooldown = 0
                print("🔃 Alerts reset")
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ System shutdown complete.")

# === Main Execution ===
if __name__ == "__main__":
    print("Initializing Enhanced Violence Detection System...")
    
    # Create system instance
    violence_detector = ViolenceDetectionSystem()
    
    # Start processing
    # Use 0 for webcam, or replace with video file path
    violence_detector.process_video_stream(0)
