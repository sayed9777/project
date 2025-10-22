import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import threading
import time

class ViolenceDetectionSystem:
    def __init__(self):
        self.model = self.load_violence_detection_model()
        self.violence_detected = False
        self.confidence_threshold = 0.75
        self.frame_buffer = deque(maxlen=16)  # Store last 16 frames for temporal analysis
        
    def load_violence_detection_model(self):
        """
        Load pre-trained violence detection model
        You can use models like:
        - SlowFast Networks
        - I3D (Inflated 3D ConvNet)
        - Custom CNN-LSTM models
        """
        try:
            # Placeholder for actual model loading
            # In practice, you'd load a pre-trained model
            model = self.create_model()
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def create_model(self):
        """Create a violence detection model architecture"""
        # This is a simplified example - use pre-trained models in production
        model = keras.Sequential([
            keras.layers.TimeDistributed(
                keras.layers.Conv2D(32, (3, 3), activation='relu'),
                input_shape=(16, 224, 224, 3)
            ),
            keras.layers.TimeDistributed(keras.layers.MaxPooling2D(2, 2)),
            keras.layers.TimeDistributed(keras.layers.Conv2D(64, (3, 3), activation='relu')),
            keras.layers.TimeDistributed(keras.layers.MaxPooling2D(2, 2)),
            keras.layers.TimeDistributed(keras.layers.Flatten()),
            keras.layers.LSTM(64),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(2, activation='softmax')  # Violence vs Non-violence
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        # Resize frame
        frame = cv2.resize(frame, (224, 224))
        # Normalize pixel values
        frame = frame.astype('float32') / 255.0
        return frame
    
    def detect_human_poses(self, frame):
        """Detect human poses and bounding boxes"""
        # Using OpenCV or MediaPipe for pose detection
        # This helps focus on human interactions
        pose_detector = self.setup_pose_detector()
        poses = pose_detector.process(frame)
        
        human_boxes = []
        if poses.pose_landmarks:
            # Extract bounding box around detected person
            landmarks = poses.pose_landmarks.landmark
            x_coords = [landmark.x for landmark in landmarks]
            y_coords = [landmark.y for landmark in landmarks]
            
            x_min = int(min(x_coords) * frame.shape[1])
            y_min = int(min(y_coords) * frame.shape[0])
            x_max = int(max(x_coords) * frame.shape[1])
            y_max = int(max(y_coords) * frame.shape[0])
            
            human_boxes.append((x_min, y_min, x_max, y_max))
        
        return human_boxes
    
    def analyze_movement_patterns(self, frame_sequence):
        """Analyze movement patterns for violence indicators"""
        violence_indicators = {
            'rapid_movement': False,
            'aggressive_gestures': False,
            'close_proximity_fighting': False,
            'falling_motions': False
        }
        
        # Analyze optical flow between frames
        if len(frame_sequence) >= 2:
            prev_frame = cv2.cvtColor(frame_sequence[-2], cv2.COLOR_BGR2GRAY)
            curr_frame = cv2.cvtColor(frame_sequence[-1], cv2.COLOR_BGR2GRAY)
            
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Calculate movement magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            avg_magnitude = np.mean(magnitude)
            
            if avg_magnitude > 5.0:  # Threshold for rapid movement
                violence_indicators['rapid_movement'] = True
        
        return violence_indicators
    
    def predict_violence(self, frame_sequence):
        """Predict violence probability from frame sequence"""
        if len(frame_sequence) < 16:
            return 0.0
        
        # Preprocess sequence
        processed_sequence = np.array([self.preprocess_frame(frame) for frame in frame_sequence])
        processed_sequence = np.expand_dims(processed_sequence, axis=0)
        
        # Get prediction
        prediction = self.model.predict(processed_sequence)
        violence_probability = prediction[0][1]  # Assuming index 1 is violence class
        
        return violence_probability
    
    def draw_violence_alert(self, frame, probability, bbox=None):
        """Draw violence detection alert on frame"""
        alert_text = f"VIOLENCE DETECTED: {probability:.2%}"
        
        # Red background for alert
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Alert text
        cv2.putText(frame, alert_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw bounding box around detected violence area
        if bbox:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
            cv2.putText(frame, "VIOLENCE", (x_min, y_min-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def process_video_stream(self, camera_source=0):
        """Main processing function for video stream"""
        cap = cv2.VideoCapture(camera_source)
        
        print("Starting violence detection system...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Store frame in buffer
            self.frame_buffer.append(frame.copy())
            
            # Process if we have enough frames
            if len(self.frame_buffer) >= 16:
                # Detect human poses
                human_boxes = self.detect_human_poses(frame)
                
                # Analyze movement patterns
                movement_analysis = self.analyze_movement_patterns(list(self.frame_buffer))
                
                # Predict violence probability
                violence_prob = self.predict_violence(list(self.frame_buffer))
                
                # Check if violence is detected
                if violence_prob > self.confidence_threshold:
                    self.violence_detected = True
                    # Draw alert on frame
                    bbox = human_boxes[0] if human_boxes else None
                    frame = self.draw_violence_alert(frame, violence_prob, bbox)
                    
                    # Trigger alert (you can add sound, notification, etc.)
                    self.trigger_alert(violence_prob)
                else:
                    self.violence_detected = False
                    cv2.putText(frame, "Monitoring...", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Violence Detection System', frame)
            
            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def trigger_alert(self, confidence):
        """Trigger alert when violence is detected"""
        print(f"🚨 VIOLENCE ALERT! Confidence: {confidence:.2%}")
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # You can add:
        # - Sound alarm
        # - Send email/SMS
        # - Save screenshot
        # - Notify security personnel
    
    def setup_pose_detector(self):
        """Setup pose detection model"""
        # This would initialize MediaPipe or similar pose detection
        # Placeholder implementation
        class PoseDetector:
            def process(self, frame):
                return type('obj', (object,), {'pose_landmarks': None})()
        return PoseDetector()

# Usage example
if __name__ == "__main__":
    # Initialize violence detection system
    violence_detector = ViolenceDetectionSystem()
    
    # Start processing (0 for default camera, or RTSP URL for IP camera)
    violence_detector.process_video_stream(0)