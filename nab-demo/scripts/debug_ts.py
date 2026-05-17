import json, csv, os
os.chdir('nab-demo')
with open('labels/combined_windows.json') as f:
    w = json.load(f)
for k, v in w.items():
    if 'ambient' in k:
        print('KEY:', k)
        print('WINDOWS:', v[:2])
        break
with open('results/ssm_latent_jepa/realKnownCause/ssm_latent_jepa_ambient_temperature_system_failure.csv') as f:
    r = csv.DictReader(f)
    rows = list(r)
print('CSV ts[0]:', repr(rows[0]['timestamp']))
print('CSV ts[-1]:', repr(rows[-1]['timestamp']))
print('Total rows:', len(rows))
# Check if window timestamps match
for k, v in w.items():
    if 'ambient' in k:
        for start, end in v[:2]:
            print(f'Window start: {repr(start)}, end: {repr(end)}')
            # Check if these exist in CSV
            ts_set = set(r['timestamp'].strip() for r in rows)
            print(f'  start in csv: {start.strip() in ts_set}')
            print(f'  end in csv: {end.strip() in ts_set}')
