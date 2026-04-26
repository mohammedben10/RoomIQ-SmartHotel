import time
import requests
import random
import sys

URL = 'http://127.0.0.1:8000/api/v1/update_sensors'

def log(msg):
    print(f"[ESP32] {msg}")

def simulate():
    temp = 22.0
    gas = 310.0
    
    print("=========================================")
    print(" ESP32 HARDWARE SIMULATOR (ROOM 101)  ")
    print("=========================================")
    print(f"Targeting: {URL}\n")
    
    for i in range(1, 100):
        # Normal walk
        temp += random.uniform(-0.2, 0.2)
        gas += random.uniform(-10, 10)
        
        # Keep in reasonable bounds
        temp = max(18, min(temp, 25))
        gas = max(250, min(gas, 450))
        
        # Inject Fire Anomaly halfway through
        if i > 15 and i < 25:
            gas += 150
            temp += 2.5
            log(f"🔥 INJECTING FIRE ANOMALY...")
            
        # Cooldown
        if i >= 25 and i < 30:
            gas = max(310, gas - 100)
            temp = max(22.0, temp - 1.5)
            log(f"❄️ COOLDOWN...")
            
        data = {
            'room_id': '101',
            'temperature': round(temp, 2),
            'gas_level': round(gas, 1),
            'motion': bool(random.getrandbits(1))
        }
        
        try:
            resp = requests.post(URL, json=data, timeout=2)
            result = resp.json()
            is_alert = result.get('ai_alert', False)
            score = result.get('score', 0)
            
            alert_str = "🔴 AI ALERT" if is_alert else "🟢 OK"
            print(f"Sent: Temp={data['temperature']}°C, Gas={data['gas_level']}PPM | State: {alert_str} (Score: {score:.3f})")
            
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Is the Django server running?")
            sys.exit(1)
            
        time.sleep(2) # Send every 2 seconds

if __name__ == '__main__':
    simulate()
