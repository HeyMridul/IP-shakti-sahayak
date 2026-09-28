"""Lists the Gemini models your API key can use and tests one call."""
import tomllib
from pathlib import Path

from google import genai

secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"
with open(secrets_path, "rb") as f:
    key = tomllib.load(f)["GEMINI_API_KEY"]

client = genai.Client(api_key=key)

print("\nAVAILABLE GEMINI MODELS (generateContent):\n")
for m in client.models.list():
    if "generateContent" in (getattr(m, "supported_actions", None) or []):
        print(m.name)

print("\nTEST CALL:")
for model in ("gemini-2.5-flash", "gemini-2.5-flash-lite"):
    try:
        r = client.models.generate_content(model=model, contents="Say hello in one word.")
        print(f"{model}: OK -> {r.text.strip()}")
        break
    except Exception as e:
        print(f"{model}: FAILED -> {e}")
