"""Analyze discriminative power of each anomaly signal (recon, latent_pred, obs_pred)
in separating known anomaly windows from normal periods."""
import csv, json, os, sys
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load combined windows labels
windows_path = os.path.join(BASE, "labels", "combined_windows.json")
with open(windows_path) as f:
    windows = json.load(f)

datasets = [
    "realKnownCause/ssm_latent_jepa_ambient_temperature_system_failure.csv",
    "artificialWithAnomaly/ssm_latent_jepa_art_daily_jumpsdown.csv",
    "realAWSCloudwatch/ssm_latent_jepa_ec2_cpu_utilization_5f5533.csv",
    "realTraffic/ssm_latent_jepa_occupancy_t4013.csv",
]

def parse_ts(ts_str):
    """Normalize timestamp: strip fractional seconds, strip whitespace."""
    s = ts_str.strip()
    # Remove fractional seconds if present: "2013-12-15 07:00:00.000000" -> "2013-12-15 07:00:00"
    if '.' in s:
        s = s.split('.')[0]
    return s

for ds in datasets:
    path = os.path.join(BASE, "results", "ssm_latent_jepa", ds)
    if not os.path.exists(path):
        print(f"MISSING: {path}")
        continue

    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Find the matching window key
    # ds is like "realKnownCause/ssm_latent_jepa_ambient_temperature_system_failure.csv"
    parts = ds.split("/", 1)
    category = parts[0]
    # Remove detector prefix from filename
    filename = parts[1].replace("ssm_latent_jepa_", "")
    data_key = f"{category}/{filename}"

    # Find matching windows
    matching_keys = [k for k in windows.keys() if k.endswith(filename)]
    if not matching_keys:
        # Try without category prefix
        matching_keys = [k for k in windows.keys() if filename in k]
    if not matching_keys:
        print(f"\n=== {filename} === NO WINDOWS FOUND (keys: {list(windows.keys())[:5]}...)")
        continue
    
    data_key = matching_keys[0]
    anomaly_windows = windows[data_key]  # list of [start_idx, end_idx]

    timestamps = [parse_ts(r["timestamp"]) for r in rows]
    recon = np.array([float(r["recon_err_raw"]) for r in rows])
    latent = np.array([float(r["latent_pred_err_raw"]) for r in rows])
    obs_pred = np.array([float(r["obs_pred_err_raw"]) for r in rows])
    combined = np.array([float(r["anomaly_score"]) for r in rows])

    # Map anomaly windows to indices (NAB windows use timestamps)
    # Build ts -> idx mapping
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}

    anomaly_indices = set()
    for start_ts, end_ts in anomaly_windows:
        start_ts = parse_ts(start_ts)
        end_ts = parse_ts(end_ts)
        if start_ts in ts_to_idx and end_ts in ts_to_idx:
            for i in range(ts_to_idx[start_ts], ts_to_idx[end_ts] + 1):
                anomaly_indices.add(i)

    all_indices = set(range(len(rows)))
    normal_indices = all_indices - anomaly_indices

    # Skip probationary period (first 15%)
    probation_len = int(len(rows) * 0.15)
    anomaly_indices = {i for i in anomaly_indices if i >= probation_len}
    normal_indices = {i for i in normal_indices if i >= probation_len}

    if not anomaly_indices:
        print(f"\n=== {filename} === NO anomaly indices after probation filter")
        continue

    anom_recon = recon[list(anomaly_indices)]
    norm_recon = recon[list(normal_indices)]
    anom_latent = latent[list(anomaly_indices)]
    norm_latent = latent[list(normal_indices)]
    anom_obs = obs_pred[list(anomaly_indices)]
    norm_obs = obs_pred[list(normal_indices)]

    def separation_ratio(anom, norm):
        """Ratio of anomaly mean to normal mean. Higher = better separation."""
        a_mean = np.mean(anom)
        n_mean = np.mean(norm)
        n_std = np.std(norm)
        if n_mean < 1e-10:
            return 999
        # How many normal stds above normal mean?
        return (a_mean - n_mean) / max(n_std, 1e-10)

    def auc_like(anom, norm):
        """Simple AUC: fraction of anomaly points above normal median."""
        threshold = np.median(norm)
        tpr = np.mean(anom > threshold)
        fpr = np.mean(norm > threshold)
        return tpr, fpr, threshold

    print(f"\n=== {filename} ===")
    print(f"  Points: total={len(rows)}, anomaly={len(anomaly_indices)}, normal={len(normal_indices)}")
    
    for name, anom, norm in [
        ("recon_err      ", anom_recon, norm_recon),
        ("latent_pred_err", anom_latent, norm_latent),
        ("obs_pred_err   ", anom_obs, norm_obs),
    ]:
        sep = separation_ratio(anom, norm)
        tpr, fpr, thresh = auc_like(anom, norm)
        print(f"  {name}: norm_mean={np.mean(norm):.6f}, anom_mean={np.mean(anom):.6f}, "
              f"separation={sep:.2f}σ, TPR={tpr:.3f}, FPR={fpr:.3f} @median_thresh={thresh:.6f}")

    # Also print a few anomaly points vs their neighbors
    print(f"  Sample anomaly windows (first 3):")
    anom_list = sorted(anomaly_indices)
    shown = 0
    for ai in anom_list:
        if shown >= 3:
            break
        # Show context: 2 before, anomaly, 2 after
        start = max(0, ai - 2)
        end = min(len(rows) - 1, ai + 2)
        print(f"    idx={ai}:")
        for j in range(start, end + 1):
            marker = " <-- ANOMALY" if j in anomaly_indices else ""
            print(f"      [{j:5d}] recon={recon[j]:.6f}, latent={latent[j]:.6f}, obs={obs_pred[j]:.6f}, value={rows[j]['value']}{marker}")
        shown += 1
