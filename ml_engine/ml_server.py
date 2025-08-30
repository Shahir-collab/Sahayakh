
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import redis
import json
import logging
from datetime import datetime

from priority_engine import MLPriorityEngine
from drone_analyzer import DroneCoordinator

# Initialize FastAPI app
app = FastAPI(title="SAHAYAK+ ML Service", version="1.0.0")

# Initialize Redis connection
redis_client = redis.Redis.from_url("redis://redis:6379/1")

# Initialize ML components
priority_engine = MLPriorityEngine()
drone_coordinator = DroneCoordinator()

# Pydantic models for API
class SOSRequest(BaseModel):
    id: str
    survivor_id: str
    location: Dict[str, float]  # {"latitude": float, "longitude": float}
    request_type: str
    description: Optional[str] = ""
    created_at: datetime

class SurvivorInfo(BaseModel):
    id: str
    age: Optional[int] = 30
    medical_conditions: Optional[List[str]] = []
    special_needs: Optional[str] = ""

class ContextData(BaseModel):
    disaster_zones: Optional[List[Dict]] = []
    nearby_rescuers: Optional[List[Dict]] = []
    weather_severity: Optional[float] = 0.3
    avg_response_time: Optional[float] = 15.0

class PriorityResponse(BaseModel):
    priority_score: float
    priority_level: int
    confidence: float
    factors: Dict[str, float]

@app.post("/calculate_priority", response_model=PriorityResponse)
async def calculate_priority(
    sos_request: SOSRequest,
    survivor_info: SurvivorInfo,
    context_data: ContextData
):
    """Calculate priority score for SOS request"""
    
    try:
        # Convert Pydantic models to dictionaries
        sos_data = {
            'id': sos_request.id,
            'survivor_id': sos_request.survivor_id,
            'latitude': sos_request.location['latitude'],
            'longitude': sos_request.location['longitude'],
            'request_type': sos_request.request_type,
            'description': sos_request.description,
            'created_at': sos_request.created_at
        }
        
        survivor_data = {
            'id': survivor_info.id,
            'age': survivor_info.age,
            'medical_conditions': survivor_info.medical_conditions,
            'special_needs': survivor_info.special_needs
        }
        
        context = {
            'disaster_zones': context_data.disaster_zones,
            'nearby_rescuers': context_data.nearby_rescuers,
            'weather_severity': context_data.weather_severity,
            'avg_response_time': context_data.avg_response_time
        }
        
        # Calculate priority score
        priority_score = priority_engine.calculate_priority_score(
            sos_data, survivor_data, context
        )
        
        priority_level = priority_engine.get_priority_level(priority_score)
        
        # Cache result in Redis
        cache_key = f"priority:{sos_request.id}"
        cache_data = {
            'score': priority_score,
            'level': priority_level,
            'calculated_at': datetime.now().isoformat()
        }
        redis_client.setex(cache_key, 3600, json.dumps(cache_data))
        
        return PriorityResponse(
            priority_score=priority_score,
            priority_level=priority_level,
            confidence=0.85,  # Model confidence
            factors={
                'time_factor': min((datetime.now() - sos_request.created_at).total_seconds() / 3600 * 2, 8),
                'type_factor': {'medical': 7, 'fire': 6, 'trapped': 6}.get(sos_request.request_type, 3),
                'age_factor': 2 if (survivor_info.age < 18 or survivor_info.age > 65) else 0,
                'location_factor': 5  # Simplified
            }
        )
        
    except Exception as e:
        logging.error(f"Priority calculation error: {e}")
        raise HTTPException(status_code=500, detail="Priority calculation failed")

@app.post("/train_model")
async def train_model(training_data: List[Dict]):
    """Train the ML model with new data"""
    
    try:
        result = priority_engine.train_model(training_data)
        
        # Save trained model
        priority_engine.save_model("/app/models/priority_model.joblib")
        
        return {"status": "success", "metrics": result}
        
    except Exception as e:
        logging.error(f"Model training error: {e}")
        raise HTTPException(status_code=500, detail="Model training failed")

@app.post("/process_drone_feed")
async def process_drone_feed(
    drone_id: str,
    gps_coords: List[float],
    has_video: bool = False,
    has_thermal: bool = False,
    has_audio: bool = False
):
    """Process drone feed data"""
    
    try:
        # This would normally receive actual video/audio data
        # For demo, we'll simulate the analysis
        
        mock_results = drone_coordinator.process_drone_feed(
            drone_id=drone_id,
            video_frame=None,  # Would be actual frame data
            thermal_frame=None,  # Would be actual thermal data
            audio_data=None,  # Would be actual audio data
            gps_coords=tuple(gps_coords)
        )
        
        # Store results in Redis for real-time access
        results_key = f"drone_results:{drone_id}:{datetime.now().isoformat()}"
        redis_client.setex(results_key, 1800, json.dumps(mock_results, default=str))
        
        return {
            "drone_id": drone_id,
            "analysis_complete": True,
            "survivor_probability": mock_results.get('survivor_probability', 0.0),
            "detections": len(mock_results.get('detections', []))
        }
        
    except Exception as e:
        logging.error(f"Drone feed processing error: {e}")
        raise HTTPException(status_code=500, detail="Drone feed processing failed")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_trained": priority_engine.is_trained,
        "redis_connected": redis_client.ping(),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
