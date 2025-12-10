
import json, argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
args = parser.parse_args()

summaries = json.loads(Path(args.input).read_text())

# Simple proxy for review-time savings: length reduction
results = []
for s in summaries:
    original_len = 0  # unknown here; in production compute from original note
    summary_len = len(s['summary'].split())
    results.append({'note_id': s['note_id'], 'summary_words': summary_len})

print('Summary lengths (words):', results)
