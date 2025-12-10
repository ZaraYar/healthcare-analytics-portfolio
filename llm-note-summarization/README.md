
# LLM Clinical Note Summarization (Azure OpenAI)

Summarize synthetic clinical notes into concise care-management summaries using Azure OpenAI. **No PHI** included.

## Highlights
- **Model:** Azure OpenAI (e.g., `gpt-4o-mini` or current equivalent)
- **Pipeline:** batch JSON input → prompt templating → summarization → JSON output
- **Evaluation:** simple utility to compare summary length vs. original and keyword coverage
- **Governance:** redaction utilities (for your data) and guidelines for PHI handling

## Setup
Set environment variables:
```bash
export AZURE_OPENAI_ENDPOINT="https://<your-endpoint>.openai.azure.com/"
export AZURE_OPENAI_KEY="<your-key>"
export AZURE_OPENAI_DEPLOYMENT="<your-deployment-name>"  # e.g., gpt-4o-mini
```

## Run
```bash
pip install -r requirements.txt
python src/summarize_notes.py --input data/sample_notes.json --output outputs/summaries.json
python src/evaluate_summaries.py --input outputs/summaries.json
```

## Data
`data/sample_notes.json` contains synthetic examples like:
```json
[{"note_id": 1, "text": "Pt with T2DM, HTN..."}]
```

## Requirements
See `requirements.txt`.

## License
MIT (see `LICENSE`).
