import time
import requests
import random
import sys
import msvcrt
import threading

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

URL = 'http://127.0.0.1:8000/api/v1/update_sensors'

ROOMS = ['101', '102', '103', '104', '105', '106']

# Global state for all rooms
room_states = {}
for r in ROOMS:
    room_states[r] = {
        'temp': random.uniform(21.0, 23.0),
        'hum': random.uniform(45.0, 55.0),
        'co2': random.uniform(390, 410),
        'light': random.uniform(250, 350),
        'sound': random.uniform(30.0, 40.0),
        'motion': 0,
        'anomaly': None,
        'timer': 0,
        'fault': False
    }

lock = threading.Lock()
running = True

def log(msg):
    print(f"\r[ESP32 MULTI] {msg}")

def simulate_data_loop():
    while running:
        for r in ROOMS:
            with lock:
                state = room_states[r]
                
                # Apply normal drift
                state['temp'] += random.uniform(-0.1, 0.1)
                state['hum'] += random.uniform(-0.5, 0.5)
                state['co2'] += random.uniform(-5, 5)
                state['light'] += random.uniform(-10, 10)
                state['sound'] = 35.0 + random.uniform(-2, 2)
                state['motion'] = 1 if random.random() < 0.05 else 0
                
                # Keep bounds normal
                if state['anomaly'] is None:
                    state['temp'] = max(20, min(state['temp'], 24))
                    state['hum'] = max(40, min(state['hum'], 60))
                    state['co2'] = max(380, min(state['co2'], 450))
                    state['light'] = max(200, min(state['light'], 400))
                
                # Apply anomaly forces
                if state['anomaly'] == 'Fire':
                    state['temp'] += 1.5
                    state['co2'] += 150
                    state['sound'] = 85.0
                    state['motion'] = 1
                elif state['anomaly'] == 'HVAC_Fail':
                    state['temp'] += 0.3
                    state['hum'] += 1.0
                    state['motion'] = 1
                elif state['anomaly'] == 'Party':
                    state['co2'] += 20
                    state['sound'] = random.uniform(70, 85)
                    state['motion'] = 1
                    state['temp'] += 0.2
                elif state['anomaly'] == 'Window_Open':
                    state['temp'] -= 0.5
                    state['hum'] -= 1.0
                    state['co2'] = max(300, state['co2'] - 10)
                    state['sound'] = random.uniform(50, 60)
                    
                if state['anomaly']:
                    state['timer'] -= 1
                    if state['timer'] <= 0:
                        state['anomaly'] = None
                        log(f"Room {r}: Anomaly resolved.")

                data = {
                    'room_id': r,
                    'temperature_c': -999.0 if state['fault'] else round(state['temp'], 2),
                    'humidity_pct': -999.0 if state['fault'] else round(state['hum'], 1),
                    'co2_gas_ppm': int(state['co2']),
                    'light_lux': int(state['light']),
                    'sound_db': round(state['sound'], 1),
                    'motion_pir': state['motion']
                }
            
            try:
                requests.post(URL, json=data, timeout=1)
            except requests.exceptions.RequestException:
                pass # Silent fail if server is down, avoid spamming terminal
            
        # Random automatic anomaly generation (for demonstration purposes)
        # 5% chance every 2 seconds to trigger a random anomaly in a random room
        if random.random() < 0.05:
            random_room = random.choice(ROOMS)
            random_anomaly = random.choice(['Fire', 'HVAC_Fail', 'Party', 'Window_Open'])
            
            with lock:
                state = room_states[random_room]
                # Only inject if the room is currently normal
                if state['anomaly'] is None and not state['fault']:
                    state['anomaly'] = random_anomaly
                    state['timer'] = 20 # Lasts for 40 seconds (20 ticks * 2s)
                    log(f"⚡ AUTO-SCENARIO: Injecting {random_anomaly} into Room {random_room}!")

        time.sleep(2) # Send updates for all rooms every 2 seconds

def interactive_prompt():
    global running
    print("\n=========================================")
    print(" ESP32 FLOOR SIMULATOR (Rooms 101-106) ")
    print("=========================================")
    print(" [F] - Fire / Smoke")
    print(" [H] - HVAC Failure")
    print(" [P] - Unauthorized Party")
    print(" [W] - Window Open")
    print(" [X] - Hardware Sensor Fault")
    print(" [N] - Normal (Reset)")
    print(" [Q] - Quit")
    
    while running:
        if msvcrt.kbhit():
            key = msvcrt.getwch().upper()
            if key == 'Q':
                running = False
                break
            
            if key in ['F', 'H', 'P', 'W', 'X', 'N']:
                print(f"\nSelect Room (1-6) for anomaly '{key}': ", end="", flush=True)
                while True:
                    if msvcrt.kbhit():
                        r_key = msvcrt.getwch()
                        if r_key in ['1', '2', '3', '4', '5', '6']:
                            room_id = f"10{r_key}"
                            print(room_id)
                            with lock:
                                state = room_states[room_id]
                                if key == 'N':
                                    state['anomaly'] = None
                                    state['fault'] = False
                                    log(f"Resetting Room {room_id} to NORMAL.")
                                elif key == 'F':
                                    state['anomaly'] = 'Fire'
                                    state['timer'] = 15
                                    state['fault'] = False
                                    log(f"🔥 Room {room_id}: INJECTING FIRE...")
                                elif key == 'H':
                                    state['anomaly'] = 'HVAC_Fail'
                                    state['timer'] = 30
                                    state['fault'] = False
                                    log(f"❄️ Room {room_id}: INJECTING HVAC FAILURE...")
                                elif key == 'P':
                                    state['anomaly'] = 'Party'
                                    state['timer'] = 30
                                    state['fault'] = False
                                    log(f"🎉 Room {room_id}: INJECTING PARTY...")
                                elif key == 'W':
                                    state['anomaly'] = 'Window_Open'
                                    state['timer'] = 20
                                    state['fault'] = False
                                    log(f"🌬️ Room {room_id}: INJECTING OPEN WINDOW...")
                                elif key == 'X':
                                    state['fault'] = True
                                    state['anomaly'] = None
                                    log(f"💥 Room {room_id}: INJECTING HARDWARE FAULT (Temp/Hum -999)...")
                            break
                        else:
                            print("\nInvalid room. Cancelled.")
                            break
        time.sleep(0.1)

if __name__ == '__main__':
    t = threading.Thread(target=simulate_data_loop)
    t.daemon = True
    t.start()
    
    interactive_prompt()
