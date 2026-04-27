from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
import json
import joblib
import pandas as pd
import os
import time

# --- ML Model Loading ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'random_forest.joblib')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.joblib')
FEATURES_PATH = os.path.join(BASE_DIR, 'model_features.joblib')

try:
    rf_model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_features = joblib.load(FEATURES_PATH)
    print("[OK] Multi-Class ML Model loaded successfully.")
except Exception as e:
    rf_model = None
    scaler = None
    model_features = None
    print(f"[WARNING] Failed to load ML model: {e}")

# --- State Management ---
ROOM_HISTORY = {}
LATEST_ROOM_STATUS = {}
ACTIVE_ROOMS = ['101', '102', '103', '104', '105', '106']

# Initialize default statuses
for r in ACTIVE_ROOMS:
    LATEST_ROOM_STATUS[r] = {
        'room': r,
        'temp': '--',
        'hum': '--',
        'co2': '--',
        'light': '--',
        'sound': '--',
        'pir': False,
        'acOn': False,
        'door_open': False,
        'red_led': False,
        'green_led': True, # Default normal state
        'ai_alert': False,
        'predicted_class': 'Offline',
        'sensor_health': 'OFFLINE',
        'last_updated': 0
    }

def update_room_history(room_id, data):
    room_id = str(room_id)
    if room_id not in ROOM_HISTORY:
        ROOM_HISTORY[room_id] = []
    
    ROOM_HISTORY[room_id].append(data)
    
    if len(ROOM_HISTORY[room_id]) > 3:
        ROOM_HISTORY[room_id] = ROOM_HISTORY[room_id][-3:]
        
    return ROOM_HISTORY[room_id]

def extract_features(history):
    if len(history) == 0:
        return None
        
    df = pd.DataFrame(history)
    current = df.iloc[-1].to_dict()
    
    feature_dict = {
        'Temperature_C': current['Temperature_C'],
        'Humidity_pct': current['Humidity_pct'],
        'CO2_Gas_PPM': current['CO2_Gas_PPM'],
        'Light_Lux': current['Light_Lux'],
        'Sound_dB': current['Sound_dB'],
        'Motion_PIR': current['Motion_PIR'],
        'temp_delta': 0.0,
        'hum_delta': 0.0,
        'co2_delta': 0.0,
        'temp_roll_mean': current['Temperature_C'],
        'co2_roll_mean': current['CO2_Gas_PPM'],
        'sound_roll_mean': current['Sound_dB'],
    }
    
    if len(df) > 1:
        feature_dict['temp_delta'] = current['Temperature_C'] - df['Temperature_C'].iloc[-2]
        feature_dict['hum_delta'] = current['Humidity_pct'] - df['Humidity_pct'].iloc[-2]
        feature_dict['co2_delta'] = current['CO2_Gas_PPM'] - df['CO2_Gas_PPM'].iloc[-2]
        
    feature_dict['temp_roll_mean'] = df['Temperature_C'].mean()
    feature_dict['co2_roll_mean'] = df['CO2_Gas_PPM'].mean()
    feature_dict['sound_roll_mean'] = df['Sound_dB'].mean()
    
    return pd.DataFrame([feature_dict])

@login_required
def index(request):
    return render(request, 'dashboard/index.html')

@login_required
def rooms(request):
    return render(request, 'dashboard/rooms.html')

@login_required
def analytics(request):
    return render(request, 'dashboard/analytics.html')

@login_required
def energy(request):
    return render(request, 'dashboard/energy.html')

@login_required
def alerts(request):
    return render(request, 'dashboard/alerts.html')

@login_required
def settings_view(request):
    return render(request, 'dashboard/settings.html')

@login_required
def simulator(request):
    return render(request, 'dashboard/simulator.html')

@csrf_exempt
def update_sensors(request):
    """RPC/API endpoint for ESP32 Simulator to send real-time data."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            room_id = str(data.get('room_id', '101'))
            
            sensor_data = {
                'Temperature_C': float(data.get('temperature_c', 22.0)),
                'Humidity_pct': float(data.get('humidity_pct', 50.0)),
                'CO2_Gas_PPM': float(data.get('co2_gas_ppm', 400)),
                'Light_Lux': float(data.get('light_lux', 300)),
                'Sound_dB': float(data.get('sound_db', 35.0)),
                'Motion_PIR': float(data.get('motion_pir', 0.0))
            }
            
            # Check for simulated sensor failures (-999 is our magic number for a failed sensor read)
            sensor_health = "OK"
            if sensor_data['Temperature_C'] == -999.0 or sensor_data['Humidity_pct'] == -999.0:
                sensor_health = "FAULTY"
                
            history = update_room_history(room_id, sensor_data)
            
            ai_alert = False
            predicted_class_name = 'Normal'
            
            label_map_inverse = {0: 'Normal', 1: 'Fire', 2: 'HVAC_Fail', 3: 'Party', 4: 'Window_Open'}
            
            # Only run AI if sensors are healthy
            if sensor_health == "OK" and rf_model and scaler and model_features and len(history) >= 1:
                features_df = extract_features(history)
                features_df = features_df[model_features]
                X_scaled = scaler.transform(features_df)
                
                pred = rf_model.predict(X_scaled)[0]
                predicted_class_name = label_map_inverse.get(pred, 'Unknown')
                
                if pred != 0:
                    ai_alert = True
                
            # Maintain existing control states unless overridden by AI
            current_state = LATEST_ROOM_STATUS.get(room_id, {})
            door_open = current_state.get('door_open', False)
            red_led = current_state.get('red_led', False)
            green_led = current_state.get('green_led', True)

            # AI Automation Override
            if ai_alert:
                green_led = False
                red_led = True
                if predicted_class_name == 'Fire':
                    door_open = True # Evacuation
                elif predicted_class_name == 'Party':
                    door_open = False # Containment/Security
            elif sensor_health == "OK":
                # Auto-reset if back to normal and healthy
                green_led = True
                red_led = False
                # Do not auto-close door if normal, let staff do it

            LATEST_ROOM_STATUS[room_id] = {
                'room': room_id,
                'temp': round(sensor_data['Temperature_C'], 1),
                'hum': round(sensor_data['Humidity_pct'], 1),
                'co2': int(sensor_data['CO2_Gas_PPM']),
                'light': int(sensor_data['Light_Lux']),
                'sound': round(sensor_data['Sound_dB'], 1),
                'pir': sensor_data['Motion_PIR'] == 1.0,
                'acOn': current_state.get('acOn', False),
                'door_open': door_open,
                'red_led': red_led,
                'green_led': green_led,
                'ai_alert': ai_alert,
                'predicted_class': predicted_class_name,
                'sensor_health': sensor_health,
                'last_updated': time.time()
            }

            return JsonResponse({"status": "success", "ai_alert": ai_alert, "predicted_class": predicted_class_name})
        except Exception as e:
            return JsonResponse({"status": "error", "msg": str(e)}, status=400)
    return JsonResponse({"status": "error", "msg": "POST required"}, status=400)

def get_room_status(request, room_id):
    """Old single-room endpoint."""
    room_id_str = str(room_id)
    if room_id_str in LATEST_ROOM_STATUS:
        return JsonResponse(LATEST_ROOM_STATUS[room_id_str])
    return JsonResponse({'error': 'Not found'}, status=404)

@login_required
def get_all_status(request):
    """New endpoint to return status of ALL active rooms."""
    response_data = []
    current_time = time.time()
    
    for room in ACTIVE_ROOMS:
        if room in LATEST_ROOM_STATUS:
            room_data = dict(LATEST_ROOM_STATUS[room])
            # Check if offline (>15 seconds without an update)
            if current_time - room_data['last_updated'] > 15:
                room_data['sensor_health'] = "OFFLINE"
            response_data.append(room_data)
        else:
            # Default offline state
            response_data.append({
                'room': room,
                'temp': '--',
                'hum': '--',
                'co2': '--',
                'light': '--',
                'sound': '--',
                'pir': False,
                'acOn': False,
                'ai_alert': False,
                'predicted_class': 'Offline',
                'sensor_health': 'OFFLINE',
                'last_updated': 0
            })
            
    return JsonResponse({'rooms': response_data})

@csrf_exempt
def toggle_ac(request, room_id):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            state = data.get('state')
            if str(room_id) in LATEST_ROOM_STATUS:
                 LATEST_ROOM_STATUS[str(room_id)]['acOn'] = state
            return JsonResponse({"status": "success", "ac_state": state})
        except Exception as e:
            return JsonResponse({"status": "error", "msg": str(e)}, status=400)
    return JsonResponse({"status": "error", "msg": "POST required"}, status=400)

@csrf_exempt
def control_device(request, room_id):
    """New endpoint to manually control door and LEDs."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            action = data.get('action')
            state = data.get('state')
            
            room_id_str = str(room_id)
            if room_id_str in LATEST_ROOM_STATUS:
                if action == 'door':
                    LATEST_ROOM_STATUS[room_id_str]['door_open'] = state
                elif action == 'red_led':
                    LATEST_ROOM_STATUS[room_id_str]['red_led'] = state
                elif action == 'green_led':
                    LATEST_ROOM_STATUS[room_id_str]['green_led'] = state
                
                return JsonResponse({
                    "status": "success", 
                    "door_open": LATEST_ROOM_STATUS[room_id_str]['door_open'],
                    "red_led": LATEST_ROOM_STATUS[room_id_str]['red_led'],
                    "green_led": LATEST_ROOM_STATUS[room_id_str]['green_led']
                })
            else:
                return JsonResponse({"status": "error", "msg": "Room not found"}, status=404)
        except Exception as e:
            return JsonResponse({"status": "error", "msg": str(e)}, status=400)
    return JsonResponse({"status": "error", "msg": "POST required"}, status=400)
