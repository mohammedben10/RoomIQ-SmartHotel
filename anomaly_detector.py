import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import datetime
import warnings

# Suppress unimportant warnings for clean terminal output
warnings.filterwarnings('ignore')

def generate_mock_data():
    """Generates a realistic 1000-row dataset of sensor readings for Hotel Room 101."""
    n_rows = 1000
    
    # 5 days of data at 5-min intervals
    start_time = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - datetime.timedelta(days=5)
    timestamps = [start_time + datetime.timedelta(minutes=5 * i) for i in range(n_rows)]
    
    np.random.seed(42)
    
    # Temperature: Baseline ~22°C with sinusoidal 24h cycle + Gaussian noise (sigma=0.35)
    hours = np.array([ts.hour + ts.minute / 60.0 for ts in timestamps])
    temp_base = 22.0 + 2.0 * np.cos((hours - 16) * 2 * np.pi / 24) # Peak at 16:00
    temperature = temp_base + np.random.normal(0, 0.35, n_rows)
    
    # Gas level: Baseline 300–420 PPM, Gaussian noise (sigma=18)
    gas_level = np.random.normal(360, 18, n_rows)
    gas_level = np.clip(gas_level, 300, 420)
    
    # Motion: 0/1 PIR. 40% daytime (08-22h), 5% nighttime
    motion = np.zeros(n_rows, dtype=int)
    for i, ts in enumerate(timestamps):
        if 8 <= ts.hour <= 22:
            motion[i] = np.random.choice([0, 1], p=[0.6, 0.4])
        else:
            motion[i] = np.random.choice([0, 1], p=[0.95, 0.05])
            
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature,
        'gas_level': gas_level,
        'motion': motion,
        'true_label': np.zeros(n_rows, dtype=int)
    })
    
    # -- Inject 5 MILD anomalies (guest smoking) --
    # 12-sample window. Gas rises +260 to +370, Temp +0.6 to +1.4.
    mild_idx = np.random.choice(range(50, n_rows - 50), 5, replace=False)
    for idx in mild_idx:
        gas_add = np.linspace(0, np.random.uniform(260, 370), 12)
        gas_add[-4:] = gas_add[-5]  # Plateau
        temp_add = np.linspace(0, np.random.uniform(0.6, 1.4), 12)
        
        df.loc[idx:idx+11, 'gas_level'] += gas_add
        df.loc[idx:idx+11, 'temperature'] += temp_add
        df.loc[idx:idx+11, 'true_label'] = 1
        
    # -- Inject 2 SEVERE anomalies (fire / gas leak) --
    # 8-sample window. Fast concave spike (exponent 0.4). Gas +900 to +1400, Temp +5 to +9.
    # Exclude ranges near mild anomalies
    all_mild = set()
    for m in mild_idx:
        all_mild.update(range(m - 20, m + 20))
    avail_severe = [i for i in range(100, n_rows - 100) if i not in all_mild]
    severe_idx = np.random.choice(avail_severe, 2, replace=False)
    
    for idx in severe_idx:
        xs = np.linspace(0, 1, 8)
        gas_spike = np.random.uniform(900, 1400) * (xs ** 0.4)
        temp_spike = np.random.uniform(5, 9) * (xs ** 0.4)
        
        df.loc[idx:idx+7, 'gas_level'] += gas_spike
        df.loc[idx:idx+7, 'temperature'] += temp_spike
        df.loc[idx:idx+7, 'true_label'] = 2
        
    # Clip final values
    df['gas_level'] = np.clip(df['gas_level'], 200, 1800)
    df['temperature'] = np.clip(df['temperature'], 16, 45)
    
    return df

def feature_engineering(df):
    """Derives rolling and delta features from raw signals."""
    df['gas_roll_mean'] = df['gas_level'].rolling(window=3, min_periods=1).mean()
    df['gas_roll_std'] = df['gas_level'].rolling(window=3, min_periods=1).std().fillna(0)
    df['temp_roll_mean'] = df['temperature'].rolling(window=3, min_periods=1).mean()
    df['gas_delta'] = df['gas_level'].diff().fillna(0)
    df['temp_delta'] = df['temperature'].diff().fillna(0)
    
    # risk_score = gas_level * (1 + clip(temp_delta, 0, ∞))
    df['risk_score'] = df['gas_level'] * (1 + np.clip(df['temp_delta'], 0, None))
    
    return df

def train_and_evaluate(df):
    """Trains IsolationForest and prints eval report."""
    features = ['temperature', 'gas_level', 'gas_roll_mean', 'gas_roll_std', 
                'temp_roll_mean', 'gas_delta', 'temp_delta', 'risk_score']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    # Algorithm: IsolationForest
    iso = IsolationForest(n_estimators=200, contamination=0.04, random_state=42, n_jobs=-1)
    df['iso_preds'] = iso.fit_predict(X_scaled)
    
    # Anomaly score = negative of score_samples() (higher = more anomalous)
    df['anomaly_score'] = -iso.score_samples(X_scaled)
    
    # Predictions logic maps: 1 (normal) -> 0, -1 (anomaly) -> 1
    df['pred_anomaly'] = (df['iso_preds'] == -1).astype(int)
    # Ground truth mapping: 0 (normal), >0 (anomaly types) -> 1
    df['is_true_anomaly'] = (df['true_label'] > 0).astype(int)
    
    # Compute metrics
    total_samples = len(df)
    gt_anomalies = df['is_true_anomaly'].sum()
    pred_anomalies = df['pred_anomaly'].sum()
    
    cm = confusion_matrix(df['is_true_anomaly'], df['pred_anomaly'], labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    
    p = precision_score(df['is_true_anomaly'], df['pred_anomaly'], zero_division=0)
    r = recall_score(df['is_true_anomaly'], df['pred_anomaly'], zero_division=0)
    f = f1_score(df['is_true_anomaly'], df['pred_anomaly'], zero_division=0)
    
    # Formatted terminal report
    print("\n" + "="*40)
    print(" EVALUATION REPORT")
    print("="*40)
    print(f"Total samples:              {total_samples}")
    print(f"Ground-truth anomaly count: {gt_anomalies}")
    print(f"Predicted anomaly count:    {pred_anomalies}")
    print("-"*40)
    print(f"True Positives  (TP):       {tp}")
    print(f"False Positives (FP):       {fp}")
    print(f"False Negatives (FN):       {fn}")
    print(f"Precision:                  {p:.4f}")
    print(f"Recall:                     {r:.4f}")
    print(f"F1 Score:                   {f:.4f}")
    print("="*40 + "\n")
    
    return df

def generate_plots(df):
    """Generates a 3-panel dark-themed Matplotlib figure of the anomalies."""
    plt.style.use('dark_background')
    facecolor = '#0d1b2a'
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, 
                             gridspec_kw={'height_ratios': [3, 1, 2]})
    fig.patch.set_facecolor(facecolor)
    
    # Standard axis styler
    def style_ax(ax):
        ax.set_facecolor(facecolor)
        ax.grid(color='#112a52', linestyle='-', linewidth=0.5, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color('#112a52')
        ax.tick_params(colors='#94a3b8')

    date_fmt = mdates.DateFormatter("%d %b\n%H:%M")
    
    # Boolean masks
    anom_pred = df['pred_anomaly'] == 1
    mild_true = df['true_label'] == 1
    sev_true = df['true_label'] == 2

    # --- Panel 1: Gas Levels (Tall) ---
    ax1 = axes[0]
    style_ax(ax1)
    ax1.plot(df['timestamp'], df['gas_level'], color='#06b6d4', linewidth=1.5, label='MQ-2 Gas (PPM)')
    ax1.axhline(600, color='#ef4444', linestyle='--', label='Danger Threshold (600 PPM)', alpha=0.8)
    
    ax1.scatter(df['timestamp'][anom_pred], df['gas_level'][anom_pred], 
                color='#ef4444', label='IF Detected', zorder=5, s=30)
    ax1.scatter(df['timestamp'][mild_true], df['gas_level'][mild_true], 
                color='#fbbf24', marker='^', label='Injected Mild (Smoking)', zorder=4, s=50)
    ax1.scatter(df['timestamp'][sev_true], df['gas_level'][sev_true], 
                color='#f97316', marker='*', label='Injected Severe (Fire)', zorder=4, s=90)
    
    ax1.set_ylabel('Gas Level (PPM)', color='#cbd5e1')
    ax1.legend(loc='upper right', facecolor=facecolor, edgecolor='#112a52', labelcolor='#cbd5e1', ncol=2)
    
    # --- Panel 2: Anomaly Score (Short) ---
    ax2 = axes[1]
    style_ax(ax2)
    p96 = np.percentile(df['anomaly_score'], 96)
    
    ax2.fill_between(df['timestamp'], df['anomaly_score'], color='#2f7dd4', alpha=0.4)
    ax2.plot(df['timestamp'], df['anomaly_score'], color='#60a5e8', linewidth=1)
    ax2.axhline(p96, color='#fbbf24', linestyle='--', label='96th %ile Threshold', alpha=0.8)
    
    ax2.set_ylabel('IF Score', color='#cbd5e1')
    ax2.legend(loc='upper right', facecolor=facecolor, edgecolor='#112a52', labelcolor='#cbd5e1', fontsize='small')
    
    # --- Panel 3: Temperature (Medium) ---
    ax3 = axes[2]
    style_ax(ax3)
    ax3.plot(df['timestamp'], df['temperature'], color='#fbbf24', linewidth=1.5, label='DHT-11 Temp (°C)')
    ax3.scatter(df['timestamp'][anom_pred], df['temperature'][anom_pred], 
                color='#ef4444', zorder=5, s=30, label='IF Detected')
    
    ax3.set_ylabel('Temp (°C)', color='#cbd5e1')
    ax3.legend(loc='upper right', facecolor=facecolor, edgecolor='#112a52', labelcolor='#cbd5e1')
    
    ax3.xaxis.set_major_formatter(date_fmt)
    plt.xticks(rotation=0, ha='center', color='#94a3b8')
    
    plt.tight_layout()
    plt.savefig('roomiq_anomaly_detection.png', dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    print("Plot saved to 'roomiq_anomaly_detection.png'.")

def export_results(df):
    """Exports results DataFrame to CSV."""
    df.to_csv('roomiq_sensor_results.csv', index=False)
    print("Results exported to 'roomiq_sensor_results.csv'.")

def main():
    print("[1/5] Generating mock data...")
    df = generate_mock_data()
    
    print("[2/5] Engineering features (rolling means, deltas)...")
    df = feature_engineering(df)
    
    print("[3/5] Training IsolationForest model & Evaluating...")
    df = train_and_evaluate(df)
    
    print("[4/5] Generating diagnostic plot...")
    generate_plots(df)
    
    print("[5/5] Exporting CSV results...")
    export_results(df)
    
    print("\nPipeline execution complete. Ready for presentation.")

if __name__ == "__main__":
    main()
