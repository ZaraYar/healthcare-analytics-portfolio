
import os, json, argparse
from pathlib import Path

import pandas as pd

try:
    from openai import AzureOpenAI
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

SYSTEM_PROMPT = Path('prompts/system_prompt.txt').read_text()
USER_PROMPT_TPL = Path('prompts/user_prompt.txt').read_text()

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()

inp = Path(args.input)
outp = Path(args.output)
outp.parent.mkdir(exist_ok=True)

notes = json.loads(inp.read_text())

summaries = []

if HAS_OPENAI:
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    key = os.getenv('AZURE_OPENAI_KEY')
    deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT')
    client = AzureOpenAI(api_key=key, api_version='2024-06-01', azure_endpoint=endpoint)

for note in notes:
    user_prompt = USER_PROMPT_TPL.replace('<NOTE_TEXT>', note['text'])
    if HAS_OPENAI and endpoint and key and deployment:
        resp = client.chat.completions.create(model=deployment, messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ])
        content = resp.choices[0].message.content
    else:
        # Offline deterministic fallback: rule-based truncation
        content = note['text'][:600]
    summaries.append({"note_id": note['note_id'], "summary": content})

outp.write_text(json.dumps(summaries, indent=2))
print(f"Wrote {len(summaries)} summaries to {outp}")
