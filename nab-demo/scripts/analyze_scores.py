import csv, sys, os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # nab-demo/

datasets = [
    "realKnownCause/ssm_latent_jepa_ambient_temperature_system_failure.csv",
    "artificialWithAnomaly/ssm_latent_jepa_art_daily_jumpsdown.csv",
    "realAWSCloudwatch/ssm_latent_jepa_ec2_cpu_utilization_5f5533.csv",
    "realTraffic/ssm_latent_jepa_occupancy_t4013.csv",
]

for ds in datasets:
    path = os.path.join(base, "results", "ssm_latent_jepa", ds)
    if not os.path.exists(path):
        print(f"MISSING: {path}")
        continue
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    scores = [float(r["anomaly_score"]) for r in rows]
    s_sorted = sorted(scores)
    nonzero = [(i, scores[i], rows[i]["value"]) for i in range(len(scores)) if scores[i] > 0.01]
    
    print(f"\n=== {ds.split('/')[-1]} ===")
    print(f"  Points: {len(scores)}")
    print(f"  Range: [{min(scores):.6f}, {max(scores):.6f}]")
    mid = len(s_sorted)//2
    print(f"  Percentiles: 50th={s_sorted[mid]:.6f}, 90th={s_sorted[int(len(s_sorted)*0.9)]:.6f}, 95th={s_sorted[int(len(s_sorted)*0.95)]:.6f}, 99th={s_sorted[int(len(s_sorted)*0.99)]:.6f}")
    print(f"  Scores > 0.99: {sum(1 for s in scores if s > 0.99)}")
    print(f"  Scores == 0.0: {sum(1 for s in scores if s == 0.0)}")
    
    # Show top non-zero scores
    nonzero_sorted = sorted(nonzero, key=lambda x: -x[1])
    print(f"  Top 10 non-zero scores:")
    for idx, s, v in nonzero_sorted[:10]:
        print(f"    idx={idx:5d}, score={s:.6f}, value={v}")
