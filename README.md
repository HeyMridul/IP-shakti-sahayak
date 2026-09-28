# IP-sakti-sahayak
Multilingual RAG-based AI assistant for Ayurveda IP and regulatory guidance.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## API key (Gemini free tier)
Get a free key at https://aistudio.google.com/apikey, then create
`.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-key-here"
```
(or set the `GEMINI_API_KEY` environment variable). On Streamlit Cloud, add it
under App settings -> Secrets. Never commit the key to GitHub.

Run `python check_gemini.py` to list the models your key can use.
