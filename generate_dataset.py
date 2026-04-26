import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- 1. Configuration ---
np.random.seed(42) # Pour garantir le même résultat à chaque exécution
num_records = 2000
start_time = datetime.now()

print("Génération des données capteurs pour RoomIQ (Chambre 101)...")

# --- 2. Données Normales (Baseline) ---
# Température moyenne ~22°C, Gaz MQ2 ~300 (air pur)
timestamps = [start_time + timedelta(minutes=i) for i in range(num_records)]
temperature = np.random.normal(loc=22.0, scale=0.5, size=num_records)
mq2_gas = np.random.normal(loc=300, scale=10, size=num_records)
labels = ["Normal"] * num_records

# --- 3. Injection d'Anomalies ---
# Scénario A : Client fume une cigarette (Légère hausse du gaz)
for i in range(400, 420):
    mq2_gas[i] = np.random.normal(loc=600, scale=20)
    labels[i] = "Cigarette"

# Scénario B : Début d'incendie (Forte hausse gaz + température)
for i in range(1200, 1215):
    mq2_gas[i] = np.random.normal(loc=950, scale=40)
    temperature[i] = np.random.normal(loc=35.0, scale=2.0)
    labels[i] = "Incendie"

# Scénario C : Client fume la nuit
for i in range(1750, 1780):
    mq2_gas[i] = np.random.normal(loc=580, scale=15)
    labels[i] = "Cigarette"

# --- 4. Création et Sauvegarde du Dataset ---
df = pd.DataFrame({
    'Timestamp': timestamps,
    'Room_ID': ["101"] * num_records,
    'Temperature_C': np.round(temperature, 1),
    'MQ2_Gas': np.round(mq2_gas, 0).astype(int),
    'True_Label': labels # Cette colonne sert juste pour vérifier si l'IA a juste
})

# Sauvegarde en CSV
filename = "roomiq_dataset.csv"
df.to_csv(filename, index=False)

print(f"Terminé ! Le fichier '{filename}' a été créé.")
print("\nRépartition des événements :")
print(df['True_Label'].value_counts())