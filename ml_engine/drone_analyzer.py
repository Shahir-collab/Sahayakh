import cv2
import numpy as np
from tensorflow import keras
import librosa
import joblib
from datetime import datetime
import logging

class DroneVideoAnalyzer:
    """Analyze drone video feeds for survivor detection"""
    
    def __init__(self):
        self.heat_threshold = 30  # Temperature threshold for human detection
        self.motion_threshold = 25
        self.person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
    
    def detect_heat_signatures(self, thermal_frame):
        """Detect potential human heat signatures in thermal imaging"""
        
        # Convert thermal data to temperature values
        temp_frame = thermal_frame.astype(np.float32)
        
        # Create mask for human body temperature range (36-42°C)
        human_temp_mask = cv2.inRange(temp_frame, 36, 42)
        
        # Find contours of potential heat signatures
        contours, _ = cv2.findContours(human_temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size (human-like heat signature)
            if 500 < area < 5000:  # Adjust based on drone altitude
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate center point
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Calculate confidence based on temperature and shape
                roi = temp_frame[y:y+h, x:x+w]
                avg_temp = np.mean(roi)
                confidence = min((avg_temp - 30) / 12, 1.0)  # Normalize to 0-1
                
                detections.append({
                    'type': 'heat_signature',
                    'bbox': (x, y, w, h),
                    'center': (center_x, center_y),
                    'confidence': confidence,
                    'temperature': avg_temp
                })
        
        return detections
    
    def detect_motion(self, frame1, frame2):
        """Detect motion between two consecutive frames"""
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate frame difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Apply threshold
        _, thresh = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours of moving objects
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size
            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (human-like)
                aspect_ratio = h / w
                if 1.5 < aspect_ratio < 4.0:
                    detections.append({
                        'type': 'motion',
                        'bbox': (x, y, w, h),
                        'area': area,
                        'confidence': min(area / 5000, 1.0)
                    })
        
        return detections
    
    def detect_persons_visual(self, frame):
        """Detect persons using computer vision"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect using Haar cascade
        bodies = self.person_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        detections = []
        for (x, y, w, h) in bodies:
            detections.append({
                'type': 'visual_person',
                'bbox': (x, y, w, h),
                'confidence': 0.7  # Fixed confidence for Haar cascade
            })
        
        return detections

class DroneAudioAnalyzer:
    """Analyze drone audio feeds for distress calls"""
    
    def __init__(self):
        self.sample_rate = 22050
        self.hop_length = 512
        self.distress_keywords = ['help', '救命', 'सहायता', 'مدد']  # Multi-language
    
    def detect_human_voice(self, audio_data):
        """Detect human voice patterns in audio"""
        
        # Extract audio features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)
        
        # Human voice typically has specific frequency characteristics
        # Fundamental frequency range: 80-250 Hz for adults
        f0 = librosa.yin(audio_data, fmin=80, fmax=250)
        
        # Calculate voice probability based on features
        voice_probability = 0.0
        
        # Check for periodic patterns (indicative of voiced speech)
        if np.mean(f0[f0 > 0]) > 0:
            voice_probability += 0.4
        
        # Check spectral centroid (voice typically 1000-4000 Hz)
        avg_centroid = np.mean(spectral_centroids)
        if 1000 < avg_centroid < 4000:
            voice_probability += 0.3
        
        # Check MFCC patterns
        mfcc_variance = np.var(mfccs)
        if mfcc_variance > 10:  # Threshold for speech-like variation
            voice_probability += 0.3
        
        return min(voice_probability, 1.0)
    
    def detect_distress_calls(self, audio_data):
        """Detect distress calls and shouting patterns"""
        
        # Analyze volume and intensity
        rms_energy = librosa.feature.rms(y=audio_data)[0]
        avg_energy = np.mean(rms_energy)
        
        # Detect sudden volume spikes (shouting)
        energy_spikes = np.where(rms_energy > avg_energy * 2)[0]
        
        # Analyze frequency patterns for distress
        stft = librosa.stft(audio_data)
        magnitude = np.abs(stft)
        
        # Look for high-frequency components (screaming/shouting)
        high_freq_energy = np.mean(magnitude[1000:, :])  # Above 1kHz
        
        distress_score = 0.0
        
        # Volume spike factor
        if len(energy_spikes) > 0:
            distress_score += 0.4
        
        # High frequency energy factor
        if high_freq_energy > np.mean(magnitude) * 1.5:
            distress_score += 0.3
        
        # Voice detection factor
        voice_prob = self.detect_human_voice(audio_data)
        distress_score += voice_prob * 0.3
        
        return min(distress_score, 1.0)

class DroneCoordinator:
    """Coordinate drone operations and integrate analysis results"""
    
    def __init__(self):
        self.video_analyzer = DroneVideoAnalyzer()
        self.audio_analyzer = DroneAudioAnalyzer()
        self.active_drones = {}
        
    def process_drone_feed(self, drone_id, video_frame, thermal_frame=None, audio_data=None, gps_coords=None):
        """Process complete drone feed and return analysis results"""
        
        results = {
            'drone_id': drone_id,
            'timestamp': datetime.now(),
            'gps_coords': gps_coords,
            'detections': []
        }
        
        # Visual analysis
        if video_frame is not None:
            visual_detections = self.video_analyzer.detect_persons_visual(video_frame)
            results['detections'].extend(visual_detections)
        
        # Thermal analysis
        if thermal_frame is not None:
            heat_detections = self.video_analyzer.detect_heat_signatures(thermal_frame)
            results['detections'].extend(heat_detections)
        
        # Audio analysis
        if audio_data is not None:
            voice_prob = self.audio_analyzer.detect_human_voice(audio_data)
            distress_prob = self.audio_analyzer.detect_distress_calls(audio_data)
            
            results['audio_analysis'] = {
                'voice_probability': voice_prob,
                'distress_probability': distress_prob
            }
        
        # Calculate overall survivor probability
        results['survivor_probability'] = self.calculate_survivor_probability(results)
        
        return results
    
    def calculate_survivor_probability(self, analysis_results):
        """Calculate overall probability of survivors in the area"""
        
        probability = 0.0
        
        # Visual detections
        for detection in analysis_results['detections']:
            if detection['type'] == 'visual_person':
                probability = max(probability, detection['confidence'] * 0.8)
            elif detection['type'] == 'heat_signature':
                probability = max(probability, detection['confidence'] * 0.7)
            elif detection['type'] == 'motion':
                probability = max(probability, detection['confidence'] * 0.5)
        
        # Audio analysis
        if 'audio_analysis' in analysis_results:
            audio = analysis_results['audio_analysis']
            audio_prob = (audio['voice_probability'] * 0.6 + 
                         audio['distress_probability'] * 0.8)
            probability = max(probability, audio_prob)
        
        return min(probability, 1.0)
    
    def prioritize_search_areas(self, drone_results):
        """Prioritize search areas based on multiple drone feeds"""
        
        areas = {}
        
        for result in drone_results:
            if result['gps_coords'] and result['survivor_probability'] > 0.3:
                lat, lng = result['gps_coords']
                
                # Create search grid (100m squares)
                grid_lat = round(lat * 1000) / 1000
                grid_lng = round(lng * 1000) / 1000
                grid_key = f"{grid_lat},{grid_lng}"
                
                if grid_key not in areas:
                    areas[grid_key] = {
                        'coordinates': (grid_lat, grid_lng),
                        'probability_scores': [],
                        'detection_count': 0,
                        'last_updated': result['timestamp']
                    }
                
                areas[grid_key]['probability_scores'].append(result['survivor_probability'])
                areas[grid_key]['detection_count'] += len(result['detections'])
                areas[grid_key]['last_updated'] = max(areas[grid_key]['last_updated'], result['timestamp'])
        
        # Calculate final priority scores
        prioritized_areas = []
        for grid_key, area_data in areas.items():
            avg_probability = np.mean(area_data['probability_scores'])
            max_probability = np.max(area_data['probability_scores'])
            
            # Combine factors for final priority
            priority_score = (avg_probability * 0.6 + 
                            max_probability * 0.3 + 
                            min(area_data['detection_count'] / 10, 0.1))
            
            prioritized_areas.append({
                'coordinates': area_data['coordinates'],
                'priority_score': priority_score,
                'detection_count': area_data['detection_count'],
                'last_updated': area_data['last_updated']
            })
        
        # Sort by priority score
        prioritized_areas.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return prioritized_areas