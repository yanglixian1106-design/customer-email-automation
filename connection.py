"""HKUST GenAI's OpenAI-compatible gateway and required subscription header."""
from openai import OpenAI

BASE_URL = "https://hkust.azure-api.net/hkust-genai/v1"


def create_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key.strip(),
        base_url=BASE_URL,
        default_headers={"api-key": api_key.strip()},
        timeout=45,
        max_retries=1,
    )
