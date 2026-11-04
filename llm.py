"""Model client factory and health check for the local Docker Model Runner."""
import json
import urllib.request

import config


def get_model_client():
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    return OpenAIChatCompletionClient(
        model=config.MODEL, base_url=config.BASE_URL, api_key=config.API_KEY,
        # Capabilities must be declared for non-OpenAI model names. Tools are not used (see agents/).
        model_info={"vision": False, "function_calling": False, "json_output": False,
                    "family": "unknown", "structured_output": False},
        temperature=0.2, max_tokens=config.MAX_TOKENS, timeout=config.LLM_TIMEOUT, max_retries=1)


def health() -> tuple[bool, str]:
    """Is the local model server reachable and is the configured model pulled?"""
    try:
        with urllib.request.urlopen(config.BASE_URL.rstrip("/") + "/models", timeout=2) as r:
            ids = [m.get("id", "").lower() for m in json.load(r).get("data", [])]
    except Exception as exc:  # noqa: BLE001
        return False, f"Cannot reach {config.BASE_URL} ({type(exc).__name__}). Is Docker Model Runner TCP enabled on 12434?"
    want = config.MODEL.lower()
    if not any(want == i or want in i or i in want for i in ids if i):
        return False, f"Model not found. Run: docker model pull {config.MODEL}"
    return True, "ok"
