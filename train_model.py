import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

def feature_engineering(df):
    """Derives rolling and delta features from raw signals."""
    # Ensure sorted by timestamp
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    df['gas_roll_mean'] = df['MQ2_Gas'].rolling(window=3, min_periods=1).mean()
    df['gas_roll_std'] = df['MQ2_Gas'].rolling(window=3, min_periods=1).std().fillna(0)
    df['temp_roll_mean'] = df['Temperature_C'].rolling(window=3, min_periods=1).mean()
    df['gas_delta'] = df['MQ2_Gas'].diff().fillna(0)
    df['temp_delta'] = df['Temperature_C'].diff().fillna(0)
    
    # risk_score = gas_level * (1 + clip(temp_delta, 0, \u221e))
    df['risk_score'] = df['MQ2_Gas'] * (1 + np.clip(df['temp_delta'], 0, None))
    
    return df

def main():
    print("Loading dataset 'roomiq_dataset.csv'...")
    df = pd.read_csv('roomiq_dataset.csv')
    
    print("Applying feature engineering...")
    df = feature_engineering(df)
    
    features = [
        'Temperature_C', 'MQ2_Gas', 'gas_roll_mean', 'gas_roll_std', 
        'temp_roll_mean', 'gas_delta', 'temp_delta', 'risk_score'
    ]
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    print("Training Isolation Forest model...")
    # Contamination based on dataset: 65 anomalies / 2000 records = 0.0325
    iso = IsolationForest(n_estimators=200, contamination=0.0325, random_state=42, n_jobs=-1)
    df['iso_preds'] = iso.fit_predict(X_scaled)
    df['anomaly_score'] = -iso.score_samples(X_scaled)
    
    # Save the trained model and scaler
    joblib.dump(iso, 'isolation_forest.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Model saved to 'isolation_forest.joblib'\nScaler saved to 'scaler.joblib'")
    
    # Evaluation
    # 1 (normal) -> 0, -1 (anomaly) -> 1
    df['pred_anomaly'] = (df['iso_preds'] == -1).astype(int)
    # Ground truth mapping: Normal (0), Cigarette/Incendie (1)
    df['is_true_anomaly'] = (df['True_Label'] != 'Normal').astype(int)
    
    print("\n--- Evaluation Report ---")
    print("Confusion Matrix:")
    print(confusion_matrix(df['is_true_anomaly'], df['pred_anomaly']))
    print("\nClassification Report:")
    print(classification_report(df['is_true_anomaly'], df['pred_anomaly']))

if __name__ == '__main__':
    main()
