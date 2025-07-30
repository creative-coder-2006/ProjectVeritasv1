import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
import os
from datetime import datetime

class VideoAnalyzer:
    def __init__(self):
        """Initialize video analysis models."""
        self.face_cascade = None
        self.deepfake_model = None
        self._load_models()
    
    def _load_models(self):
        """Load the required models."""
        try:
            # Load OpenCV face detection
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Load a pre-trained model for deepfake detection
            # For now, we'll use a simple approach with face detection and analysis
            print("Video analysis models loaded successfully")
            
        except Exception as e:
            print(f"Error loading video analysis models: {e}")
            self.face_cascade = None
    
    def extract_frames(self, video_path, max_frames=100):
        """Extract frames from video for analysis."""
        frames = []
        face_frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return frames, face_frames
            
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = max(1, total_frames // max_frames)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    
                    # Detect faces in frame
                    if self.face_cascade:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                        
                        if len(faces) > 0:
                            face_frames.append({
                                'frame': frame_rgb,
                                'faces': faces,
                                'frame_number': frame_count
                            })
                
                frame_count += 1
                
                if len(frames) >= max_frames:
                    break
            
            cap.release()
            
        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")
        
        return frames, face_frames
    
    def analyze_frame_quality(self, frame):
        """Analyze frame quality metrics."""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Calculate quality metrics
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate noise level
            noise_level = np.std(gray - cv2.GaussianBlur(gray, (5, 5), 0))
            
            return {
                'brightness': brightness,
                'contrast': contrast,
                'sharpness': laplacian_var,
                'noise_level': noise_level
            }
            
        except Exception as e:
            print(f"Error analyzing frame quality: {e}")
            return {
                'brightness': 0.0,
                'contrast': 0.0,
                'sharpness': 0.0,
                'noise_level': 0.0
            }
    
    def detect_face_anomalies(self, face_frames):
        """Detect anomalies in face regions that might indicate deepfakes."""
        anomalies = []
        
        if not face_frames:
            return anomalies
        
        try:
            # Analyze face consistency across frames
            face_sizes = []
            face_positions = []
            
            for frame_data in face_frames:
                faces = frame_data['faces']
                for (x, y, w, h) in faces:
                    face_sizes.append(w * h)
                    face_positions.append((x, y))
            
            if len(face_sizes) > 1:
                # Check for sudden changes in face size (potential deepfake artifact)
                size_changes = np.diff(face_sizes)
                size_variance = np.var(size_changes)
                
                if size_variance > np.mean(face_sizes) * 0.5:  # High variance
                    anomalies.append('inconsistent_face_sizes')
                
                # Check for unrealistic face movements
                if len(face_positions) > 1:
                    position_changes = []
                    for i in range(1, len(face_positions)):
                        dx = face_positions[i][0] - face_positions[i-1][0]
                        dy = face_positions[i][1] - face_positions[i-1][1]
                        distance = np.sqrt(dx**2 + dy**2)
                        position_changes.append(distance)
                    
                    if position_changes:
                        max_movement = max(position_changes)
                        if max_movement > 100:  # Unrealistic movement
                            anomalies.append('unrealistic_face_movement')
            
            # Check for face detection consistency
            face_count_per_frame = [len(frame_data['faces']) for frame_data in face_frames]
            if len(set(face_count_per_frame)) > 1:
                anomalies.append('inconsistent_face_count')
            
        except Exception as e:
            print(f"Error detecting face anomalies: {e}")
            anomalies.append('analysis_error')
        
        return anomalies
    
    def analyze_video_compression_artifacts(self, frames):
        """Analyze video for compression artifacts that might indicate manipulation."""
        artifacts = []
        
        try:
            for i, frame in enumerate(frames):
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                
                # Detect block artifacts (common in compressed/manipulated videos)
                # Apply DCT-like analysis
                h, w = gray.shape
                block_size = 8
                
                for y in range(0, h - block_size, block_size):
                    for x in range(0, w - block_size, block_size):
                        block = gray[y:y+block_size, x:x+block_size]
                        
                        # Check for uniform blocks (compression artifact)
                        block_std = np.std(block)
                        if block_std < 5:  # Very uniform block
                            artifacts.append('compression_artifacts')
                            break
                    if 'compression_artifacts' in artifacts:
                        break
                
                if 'compression_artifacts' in artifacts:
                    break
                
                # Limit analysis to first few frames
                if i > 10:
                    break
            
        except Exception as e:
            print(f"Error analyzing compression artifacts: {e}")
            artifacts.append('analysis_error')
        
        return artifacts
    
    def calculate_deepfake_probability(self, face_frames, frames, anomalies):
        """Calculate probability of deepfake based on various indicators."""
        probability = 0.5  # Base probability
        
        try:
            # Adjust based on face anomalies
            if 'inconsistent_face_sizes' in anomalies:
                probability += 0.2
            if 'unrealistic_face_movement' in anomalies:
                probability += 0.15
            if 'inconsistent_face_count' in anomalies:
                probability += 0.1
            
            # Adjust based on compression artifacts
            if 'compression_artifacts' in anomalies:
                probability += 0.1
            
            # Adjust based on face detection quality
            if face_frames:
                face_detection_rate = len(face_frames) / len(frames)
                if face_detection_rate < 0.3:  # Few faces detected
                    probability += 0.1
                elif face_detection_rate > 0.8:  # Too many faces
                    probability += 0.05
            
            # Cap probability
            probability = min(1.0, max(0.0, probability))
            
        except Exception as e:
            print(f"Error calculating deepfake probability: {e}")
            probability = 0.5
        
        return probability
    
    def analyze_video(self, video_path):
        """Comprehensive video analysis for deepfake detection."""
        if not video_path or not os.path.exists(video_path):
            return {
                'deepfake_probability': 0.5,
                'confidence_score': 0.0,
                'anomalies': [],
                'face_count': 0,
                'frame_count': 0,
                'error': 'Video file not found'
            }
        
        try:
            # Extract frames
            frames, face_frames = self.extract_frames(video_path)
            
            if not frames:
                return {
                    'deepfake_probability': 0.5,
                    'confidence_score': 0.0,
                    'anomalies': [],
                    'face_count': 0,
                    'frame_count': 0,
                    'error': 'Could not extract frames'
                }
            
            # Analyze frame quality
            quality_metrics = []
            for frame in frames[:10]:  # Analyze first 10 frames
                quality = self.analyze_frame_quality(frame)
                quality_metrics.append(quality)
            
            # Detect face anomalies
            face_anomalies = self.detect_face_anomalies(face_frames)
            
            # Detect compression artifacts
            compression_artifacts = self.analyze_video_compression_artifacts(frames)
            
            # Combine all anomalies
            all_anomalies = face_anomalies + compression_artifacts
            
            # Calculate deepfake probability
            deepfake_probability = self.calculate_deepfake_probability(
                face_frames, frames, all_anomalies
            )
            
            # Calculate confidence based on analysis quality
            confidence_score = min(0.9, len(frames) / 100 + len(face_frames) / 50)
            
            return {
                'deepfake_probability': deepfake_probability,
                'confidence_score': confidence_score,
                'anomalies': all_anomalies,
                'face_count': len(face_frames),
                'frame_count': len(frames),
                'quality_metrics': {
                    'avg_brightness': np.mean([q['brightness'] for q in quality_metrics]),
                    'avg_contrast': np.mean([q['contrast'] for q in quality_metrics]),
                    'avg_sharpness': np.mean([q['sharpness'] for q in quality_metrics]),
                    'avg_noise': np.mean([q['noise_level'] for q in quality_metrics])
                },
                'face_anomalies': face_anomalies,
                'compression_artifacts': compression_artifacts
            }
            
        except Exception as e:
            print(f"Error in video analysis: {e}")
            return {
                'deepfake_probability': 0.5,
                'confidence_score': 0.0,
                'anomalies': ['analysis_error'],
                'face_count': 0,
                'frame_count': 0,
                'error': str(e)
            }
    
    def analyze_multiple_videos(self, video_paths):
        """Analyze multiple videos."""
        results = []
        
        for i, video_path in enumerate(video_paths):
            try:
                result = self.analyze_video(video_path)
                result['video_index'] = i
                result['video_path'] = video_path
                results.append(result)
            except Exception as e:
                print(f"Error analyzing video {i}: {e}")
                results.append({
                    'deepfake_probability': 0.5,
                    'confidence_score': 0.0,
                    'anomalies': [],
                    'face_count': 0,
                    'frame_count': 0,
                    'error': str(e),
                    'video_index': i,
                    'video_path': video_path
                })
        
        return results

# Global analyzer instance
video_analyzer = VideoAnalyzer()

def analyze_video(video_path):
    """Convenience function to analyze video."""
    return video_analyzer.analyze_video(video_path)

def analyze_multiple_videos(video_paths):
    """Convenience function to analyze multiple videos."""
    return video_analyzer.analyze_multiple_videos(video_paths) 