from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import joblib
import pandas as pd
import numpy as np
import os

# --- ML Model Loading ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'isolation_forest.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.joblib')

try:
    iso_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[OK] ML Model loaded successfully from", MODEL_PATH)
except Exception as e:
    iso_model = None
    scaler = None
    print(f"[WARNING] Failed to load ML model: {e}")

# --- State Management ---
# Structure: { room_id: [{'temp': x, 'gas': y, 'timestamp': z}, ...] }
ROOM_HISTORY = {}
LATEST_ROOM_STATUS = {}

def update_room_history(room_id, temp, gas):
    room_id = str(room_id)
    if room_id not in ROOM_HISTORY:
        ROOM_HISTORY[room_id] = []
    
    ROOM_HISTORY[room_id].append({'temp': temp, 'gas': gas})
    
    # Keep only the last 3 for rolling features
    if len(ROOM_HISTORY[room_id]) > 3:
        ROOM_HISTORY[room_id] = ROOM_HISTORY[room_id][-3:]
        
    return ROOM_HISTORY[room_id]

def extract_features(history):
    """Calculates features based on the history buffer."""
    if len(history) == 0:
        return None
        
    df = pd.DataFrame(history)
    
    current_temp = df['temp'].iloc[-1]
    current_gas = df['gas'].iloc[-1]
    
    gas_roll_mean = df['gas'].mean()
    gas_roll_std = df['gas'].std(ddof=0) if len(df) > 1 else 0.0
    temp_roll_mean = df['temp'].mean()
    
    gas_delta = current_gas - df['gas'].iloc[-2] if len(df) > 1 else 0.0
    temp_delta = current_temp - df['temp'].iloc[-2] if len(df) > 1 else 0.0
    
    risk_score = current_gas * (1 + max(0, temp_delta))
    
    feature_dict = {
        'Temperature_C': current_temp,
        'MQ2_Gas': current_gas,
        'gas_roll_mean': gas_roll_mean,
        'gas_roll_std': gas_roll_std,
        'temp_roll_mean': temp_roll_mean,
        'gas_delta': gas_delta,
        'temp_delta': temp_delta,
        'risk_score': risk_score
    }
    
    return pd.DataFrame([feature_dict])

def index(request):
    return render(request, 'dashboard/index.html')

def rooms(request):
    return render(request, 'dashboard/rooms.html')

def analytics(request):
    return render(request, 'dashboard/analytics.html')

def energy(request):
    return render(request, 'dashboard/energy.html')

def alerts(request):
    return render(request, 'dashboard/alerts.html')

def settings_view(request):
    return render(request, 'dashboard/settings.html')

def simulator(request):
    return render(request, 'dashboard/simulator.html')

@csrf_exempt
def update_sensors(request):
    """RPC/API endpoint for ESP32 or Simulator to send real-time data."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            room_id = str(data.get('room_id', '101'))
            temp = float(data.get('temperature', 22.0))
            gas = float(data.get('gas_level', 300))
            
            # 1. Update Memory
            history = update_room_history(room_id, temp, gas)
            
            ai_alert = False
            anomaly_score = 0.0
            
            # 2. ML Inference
            if iso_model and scaler and len(history) >= 1:
                features_df = extract_features(history)
                features_list = ['Temperature_C', 'MQ2_Gas', 'gas_roll_mean', 'gas_roll_std', 'temp_roll_mean', 'gas_delta', 'temp_delta', 'risk_score']
                X_scaled = scaler.transform(features_df[features_list])
                
                # Predict (-1 is anomaly, 1 is normal)
                pred = iso_model.predict(X_scaled)[0]
                anomaly_score = float(-iso_model.score_samples(X_scaled)[0])
                
                if pred == -1:
                    ai_alert = True
                
            # 3. Update Global Status for Web Polling
            LATEST_ROOM_STATUS[room_id] = {
                'room': room_id,
                'temp': round(temp, 1),
                'mqtt_gas': round(gas, 0),
                'pir': data.get('motion', 0),
                'acOn': False,
                'ai_alert': ai_alert,
                'anomaly_score': round(anomaly_score, 3)
            }

            return JsonResponse({"status": "success", "ai_alert": ai_alert, "score": anomaly_score})
        except Exception as e:
            return JsonResponse({"status": "error", "msg": str(e)}, status=400)
    return JsonResponse({"status": "error", "msg": "POST required"}, status=400)

def get_room_status(request, room_id):
    room_id_str = str(room_id)
    if room_id_str in LATEST_ROOM_STATUS:
        return JsonResponse(LATEST_ROOM_STATUS[room_id_str])
        
    data = {
        'room': room_id,
        'temp': 22.5,
        'mqtt_gas': 412,
        'pir': False,
        'acOn': False,
        'ai_alert': False,
        'anomaly_score': 0.0
    }
    return JsonResponse(data)

@csrf_exempt
def toggle_ac(request, room_id):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            return JsonResponse({"status": "success", "ac_state": data.get('state')})
        except Exception as e:
            return JsonResponse({"status": "error", "msg": str(e)}, status=400)
    return JsonResponse({"status": "error", "msg": "POST required"}, status=400)
