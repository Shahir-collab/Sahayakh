import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import re
from datetime import datetime, timedelta
import logging

class NLPProcessor:
    """Process SOS messages using NLP techniques"""
    
    def __init__(self):
        self.emergency_keywords = {
            'critical': ['dying', 'unconscious', 'bleeding', 'heart attack', 'stroke', 'breathing', 'chest pain'],
            'urgent': ['trapped', 'stuck', 'injured', 'broken', 'fire', 'smoke', 'water rising'],
            'moderate': ['help', 'lost', 'scared', 'alone', 'cold', 'hungry'],
            'pregnancy': ['pregnant', 'labor', 'contractions', 'baby'],
            'child': ['child', 'baby', 'kid', 'infant', 'toddler'],
            'elderly': ['old', 'elderly', 'senior', 'grandfather', 'grandmother']
        }
        
    def extract_urgency_score(self, message):
        """Extract urgency score from message text"""
        if not message:
            return 0.5
        
        message = message.lower()
        score = 0.0
        
        # Check for critical keywords
        for keyword in self.emergency_keywords['critical']:
            if keyword in message:
                score += 0.3
        
        # Check for urgent keywords
        for keyword in self.emergency_keywords['urgent']:
            if keyword in message:
                score += 0.2
        
        # Check for vulnerable groups
        for keyword in self.emergency_keywords['pregnancy']:
            if keyword in message:
                score += 0.15
                
        for keyword in self.emergency_keywords['child']:
            if keyword in message:
                score += 0.1
                
        for keyword in self.emergency_keywords['elderly']:
            if keyword in message:
                score += 0.1
        
        # Check for moderate keywords
        for keyword in self.emergency_keywords['moderate']:
            if keyword in message:
                score += 0.05
        
        # Message length factor (longer messages might indicate more serious situations)
        length_factor = min(len(message.split()) / 50, 0.1)
        score += length_factor
        
        # Normalize score to 0-1 range
        return min(score, 1.0)
    
    def extract_victim_count(self, message):
        """Extract estimated number of victims from message"""
        if not message:
            return 1
        
        # Look for numbers in text
        numbers = re.findall(r'\b\d+\b', message)
        if numbers:
            return min(int(numbers[0]), 10)  # Cap at 10 for safety
        
        # Look for quantity words
        quantity_words = {
            'few': 3, 'several': 4, 'many': 6, 'family': 4,
            'group': 5, 'crowd': 8, 'us': 2, 'we': 2
        }
        
        for word, count in quantity_words.items():
            if word in message.lower():
                return count
                
        return 1

class DisasterZoneAnalyzer:
    """Analyze disaster zones and calculate risk scores"""
    
    def __init__(self):
        self.zone_types = {
            'flood': {'base_risk': 0.7, 'time_decay': 0.95},
            'fire': {'base_risk': 0.9, 'time_decay': 0.8},
            'earthquake': {'base_risk': 0.8, 'time_decay': 0.9},
            'cyclone': {'base_risk': 0.85, 'time_decay': 0.85},
            'landslide': {'base_risk': 0.75, 'time_decay': 0.9}
        }
    
    def calculate_location_risk(self, latitude, longitude, disaster_zones):
        """Calculate risk score based on location and active disaster zones"""
        max_risk = 0.1  # Base risk
        
        for zone in disaster_zones:
            # Calculate distance to zone center (simplified)
            distance = self.haversine_distance(
                latitude, longitude, 
                zone['center_lat'], zone['center_lng']
            )
            
            # Risk decreases with distance
            if distance <= zone['radius']:
                zone_risk = self.zone_types.get(zone['type'], {'base_risk': 0.5})['base_risk']
                
                # Time factor - risk decreases over time
                hours_since = (datetime.now() - zone['created_at']).total_seconds() / 3600
                time_factor = self.zone_types.get(zone['type'], {'time_decay': 0.9})['time_decay'] ** (hours_since / 24)
                
                # Distance factor - risk decreases with distance from center
                distance_factor = max(0, 1 - (distance / zone['radius']))
                
                calculated_risk = zone_risk * time_factor * distance_factor
                max_risk = max(max_risk, calculated_risk)
        
        return min(max_risk, 1.0)
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points on Earth"""
        R = 6371  # Earth's radius in kilometers
        
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c

class MLPriorityEngine:
    """Main ML-powered priority scoring engine"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.nlp_processor = NLPProcessor()
        self.zone_analyzer = DisasterZoneAnalyzer()
        self.is_trained = False
        self.feature_names = [
            'time_elapsed_hours',
            'request_type_encoded',
            'message_urgency_score',
            'victim_count',
            'location_risk_score',
            'age_factor',
            'has_medical_condition',
            'weather_severity',
            'nearby_rescuer_count',
            'historical_response_time'
        ]
        
    def prepare_features(self, sos_request, survivor_info, context_data):
        """Prepare feature vector for ML model"""
        
        # Time-based features
        time_elapsed = (datetime.now() - sos_request['created_at']).total_seconds() / 3600
        
        # Request type encoding
        request_type_map = {
            'medical': 5, 'fire': 4, 'trapped': 4, 'collapsed': 4,
            'flood': 3, 'other': 2
        }
        request_type_encoded = request_type_map.get(sos_request['request_type'], 2)
        
        # NLP features
        message_urgency = self.nlp_processor.extract_urgency_score(sos_request.get('description', ''))
        victim_count = self.nlp_processor.extract_victim_count(sos_request.get('description', ''))
        
        # Location risk
        location_risk = self.zone_analyzer.calculate_location_risk(
            sos_request['latitude'], 
            sos_request['longitude'],
            context_data.get('disaster_zones', [])
        )
        
        # Survivor characteristics
        age = survivor_info.get('age', 30)
        age_factor = 1.0
        if age < 18 or age > 65:
            age_factor = 1.3
        elif age < 12 or age > 75:
            age_factor = 1.5
            
        has_medical_condition = 1 if survivor_info.get('medical_conditions') else 0
        
        # Environmental factors
        weather_severity = context_data.get('weather_severity', 0.3)
        
        # Resource availability
        nearby_rescuer_count = len(context_data.get('nearby_rescuers', []))
        
        # Historical data
        historical_response_time = context_data.get('avg_response_time', 15) / 60  # Convert to hours
        
        features = [
            time_elapsed,
            request_type_encoded,
            message_urgency,
            victim_count,
            location_risk,
            age_factor,
            has_medical_condition,
            weather_severity,
            nearby_rescuer_count,
            historical_response_time
        ]
        
        return np.array(features).reshape(1, -1)
    
    def train_model(self, training_data):
        """Train the ML model using historical data"""
        
        # Prepare training features and targets
        X = []
        y = []
        
        for record in training_data:
            features = self.prepare_features(
                record['sos_request'],
                record['survivor_info'],
                record['context_data']
            )
            X.append(features[0])
            y.append(record['actual_priority_score'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train ensemble model
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logging.info(f"Model trained - MSE: {mse:.4f}, R2: {r2:.4f}")
        
        self.is_trained = True
        return {'mse': mse, 'r2': r2}
    
    def calculate_priority_score(self, sos_request, survivor_info, context_data):
        """Calculate priority score using ML model + rule-based fallback"""
        
        # Rule-based scoring (fallback)
        rule_based_score = self._rule_based_scoring(sos_request, survivor_info, context_data)
        
        if not self.is_trained:
            return rule_based_score
        
        try:
            # ML-based scoring
            features = self.prepare_features(sos_request, survivor_info, context_data)
            features_scaled = self.scaler.transform(features)
            ml_score = self.model.predict(features_scaled)[0]
            
            # Combine ML and rule-based scores (weighted average)
            final_score = 0.7 * ml_score + 0.3 * rule_based_score
            
            # Ensure score is in valid range (0-25)
            return max(0, min(25, final_score))
            
        except Exception as e:
            logging.error(f"ML scoring failed: {e}")
            return rule_based_score
    
    def _rule_based_scoring(self, sos_request, survivor_info, context_data):
        """Fallback rule-based priority scoring"""
        
        base_score = 10.0
        
        # Time factor
        time_elapsed = (datetime.now() - sos_request['created_at']).total_seconds() / 3600
        time_score = min(time_elapsed * 1.5, 8)
        
        # Request type factor
        type_scores = {
            'medical': 7, 'fire': 6, 'trapped': 6, 'collapsed': 6,
            'flood': 4, 'other': 3
        }
        type_score = type_scores.get(sos_request['request_type'], 3)
        
        # Age factor
        age = survivor_info.get('age', 30)
        age_score = 0
        if age < 18 or age > 65:
            age_score = 2
        if age < 12 or age > 75:
            age_score = 4
        
        # Medical condition factor
        medical_score = 3 if survivor_info.get('medical_conditions') else 0
        
        # Message urgency (using NLP)
        message_urgency = self.nlp_processor.extract_urgency_score(
            sos_request.get('description', '')
        )
        urgency_score = message_urgency * 5
        
        # Location risk
        location_risk = self.zone_analyzer.calculate_location_risk(
            sos_request['latitude'], 
            sos_request['longitude'],
            context_data.get('disaster_zones', [])
        )
        location_score = location_risk * 6
        
        total_score = (base_score + time_score + type_score + 
                      age_score + medical_score + urgency_score + location_score)
        
        return min(total_score, 25)
    
    def get_priority_level(self, score):
        """Convert priority score to level"""
        if score >= 20:
            return 3  # Critical
        elif score >= 15:
            return 2  # High
        elif score >= 10:
            return 1  # Medium
        else:
            return 0  # Low
    
    def save_model(self, filepath):
        """Save trained model to file"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load trained model from file"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            self.feature_names = model_data['feature_names']
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")