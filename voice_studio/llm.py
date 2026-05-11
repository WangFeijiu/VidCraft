"""LLM abstraction — OpenAI-compatible + Anthropic, via httpx."""
import base64
import json
from pathlib import Path

import httpx

CONFIG_PATH = Path(__file__).parent / "llm_config.json"

DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
}

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-6",
}

DEFAULT_VISION_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
}


def load_config():
    """Load config. Returns {configs: [...], active_idx: int}."""
    if CONFIG_PATH.exists():
        data = json.loads(CONFIG_PATH.read_text("utf-8"))
        # Migrate old single-config format
        if "configs" not in data:
            old = dict(data)
            data = {
                "configs": [
                    {
                        "name": old.get("name") or "默认配置",
                        "provider": old.get("provider", "openai"),
                        "api_key": old.get("api_key", ""),
                        "model": old.get("model", ""),
                        "base_url": old.get("base_url", ""),
                    }
                ],
                "active_idx": 0,
            }
        return data
    return {"configs": [], "active_idx": 0}


def save_config(cfg):
    """Save config, preserving original API keys when masked keys are received."""
    stored = load_config()
    stored_configs = stored.get("configs", [])
    new_configs = cfg.get("configs", [])
    for i, c in enumerate(new_configs):
        key = c.get("api_key", "")
        if "****" in key and i < len(stored_configs):
            c["api_key"] = stored_configs[i].get("api_key", "")
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), "utf-8")


def resolve_key(cfg_dict):
    """If api_key is masked, resolve it from stored config by name."""
    key = cfg_dict.get("api_key", "")
    if "****" in key:
        stored = load_config()
        for c in stored.get("configs", []):
            if c.get("name") == cfg_dict.get("name"):
                return dict(cfg_dict, api_key=c.get("api_key", ""))
    return cfg_dict


def mask_config(cfg):
    """Mask API keys for display."""
    out = {"configs": [], "active_idx": cfg.get("active_idx", 0)}
    for c in cfg.get("configs", []):
        mc = dict(c)
        key = mc.get("api_key", "")
        # Show first 4 and last 4 chars with **** in between
        if len(key) > 8:
            mc["api_key"] = key[:4] + "****" + key[-4:]
        elif len(key) > 4:
            mc["api_key"] = "****" + key[-4:]
        else:
            mc["api_key"] = "****" if key else ""
        out["configs"].append(mc)
    return out


def get_active_config(cfg=None):
    """Return the currently active config dict, or None."""
    if cfg is None:
        cfg = load_config()
    idx = cfg.get("active_idx", 0)
    configs = cfg.get("configs", [])
    if 0 <= idx < len(configs):
        return configs[idx]
    return configs[0] if configs else None


def call_llm(prompt, system=None, config=None, max_tokens=None):
    """Call LLM and return the text response."""
    if config is None:
        config = get_active_config()
    if not config:
        raise ValueError("LLM 未配置，请先在设置中添加配置")

    provider = config.get("provider", "openai")
    api_key = config.get("api_key", "")
    model = config.get("model") or DEFAULT_MODELS.get(provider, "")
    base_url = config.get("base_url") or DEFAULT_BASE_URLS.get(provider, "")

    if not api_key:
        raise ValueError(f"「{config.get('name','未命名')}」API Key 未配置")

    if provider == "anthropic":
        return _call_anthropic(base_url, api_key, model, prompt, system, max_tokens)
    else:
        return _call_openai(base_url, api_key, model, prompt, system)


def call_llm_vision(prompt, images_b64, system=None, config=None, timeout=None):
    """Call multimodal LLM with images."""
    if config is None:
        config = get_active_config()
    if not config:
        raise ValueError("LLM 未配置，请先在设置中添加配置")

    provider = config.get("provider", "openai")
    api_key = config.get("api_key", "")
    model = config.get("model") or DEFAULT_VISION_MODELS.get(provider, "")
    base_url = config.get("base_url") or DEFAULT_BASE_URLS.get(provider, "")

    if not api_key:
        raise ValueError(f"「{config.get('name','未命名')}」API Key 未配置")

    if provider == "anthropic":
        return _call_anthropic_vision(base_url, api_key, model, prompt, images_b64, system, timeout)
    else:
        return _call_openai_vision(base_url, api_key, model, prompt, images_b64, system, timeout)


def _retry_post(url, headers, body, timeout=120, max_retries=4):
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = httpx.post(url, headers=headers, json=body, timeout=timeout)
            resp.raise_for_status()
            return resp
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (500, 502, 503, 504, 429):
                last_err = e
                try:
                    body_text = e.response.text[:200]
                except Exception:
                    body_text = ""
                print(f"[llm] retry {attempt+1}/{max_retries} {e.response.status_code}: {body_text}")
                import time; time.sleep(2 ** attempt)
                continue
            try:
                err_body = e.response.text[:500]
            except Exception:
                err_body = ""
            raise RuntimeError(f"LLM API {e.response.status_code}: {err_body}") from e
    raise RuntimeError(f"LLM API failed after {max_retries} retries, last: {last_err.response.status_code}")

def _call_openai(base_url, api_key, model, prompt, system):
    url = f"{base_url.rstrip('/')}/chat/completions"
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = _retry_post(url,
                       headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                       body={"model": model, "messages": messages, "temperature": 0.3})
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _is_kimi(base_url):
    return "kimi.com" in base_url


def _is_third_party(base_url):
    return "anthropic.com" not in base_url


def _anthropic_headers(base_url, api_key):
    headers = {
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    if _is_kimi(base_url):
        # Kimi uses x-api-key even though it's third-party
        headers["x-api-key"] = api_key
    elif _is_third_party(base_url):
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        headers["x-api-key"] = api_key
    return headers


def _call_anthropic(base_url, api_key, model, prompt, system, max_tokens=None):
    base = base_url.rstrip('/')
    # Kimi base_url often omits /v1 (e.g. https://api.kimi.com/coding/)
    if _is_kimi(base_url) and not base.endswith('/v1'):
        base = base + '/v1'
    url = f"{base}/messages"
    # Kimi requires content as array format
    if _is_kimi(base_url):
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    else:
        messages = [{"role": "user", "content": prompt}]
    body = {"model": model, "max_tokens": max_tokens or 128000, "messages": messages}
    if system:
        body["system"] = system

    resp = _retry_post(url,
                       headers=_anthropic_headers(base_url, api_key),
                       body=body,
                       timeout=300)
    data = resp.json()
    return data["content"][0]["text"]


def _detect_media_type(b64):
    header = base64.b64decode(b64[:12])
    if header[:3] == b'\xff\xd8\xff':
        return "image/jpeg"
    if header[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    return "image/jpeg"


def _call_openai_vision(base_url, api_key, model, prompt, images_b64, system, timeout=None):
    url = f"{base_url.rstrip('/')}/chat/completions"
    content = []
    for img in images_b64:
        mt = _detect_media_type(img)
        content.append({"type": "image_url", "image_url": {"url": f"data:{mt};base64,{img}"}})
    content.append({"type": "text", "text": prompt})

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})

    resp = _retry_post(url,
                       headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                       body={"model": model, "messages": messages, "temperature": 0.3},
                       timeout=timeout or 180)
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _call_anthropic_vision(base_url, api_key, model, prompt, images_b64, system, timeout=None):
    base = base_url.rstrip('/')
    if _is_kimi(base_url) and not base.endswith('/v1'):
        base = base + '/v1'
    url = f"{base}/messages"
    content = []
    for img in images_b64:
        mt = _detect_media_type(img)
        content.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": img}})
    content.append({"type": "text", "text": prompt})

    body = {"model": model, "max_tokens": 64000, "messages": [{"role": "user", "content": content}]}
    if system:
        body["system"] = system

    _timeout = timeout or 300
    # Retry on transient errors (including 500 from overloaded upstream)
    last_err = None
    for attempt in range(4):
        try:
            resp = httpx.post(
                url,
                headers=_anthropic_headers(base_url, api_key),
                json=body,
                timeout=_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]
        except httpx.HTTPStatusError as e:
            # 500 from LLM upstream is often transient overload
            if e.response.status_code in (500, 502, 503, 504, 429):
                last_err = e
                try:
                    body_text = e.response.text[:300]
                except Exception:
                    body_text = ""
                print(f"[llm] attempt {attempt+1} failed with {e.response.status_code}: {body_text}")
                import time; time.sleep(2 ** attempt)
                continue
            # Surface the actual error body for debugging
            try:
                err_body = e.response.text[:500]
            except Exception:
                err_body = ""
            raise RuntimeError(f"LLM API {e.response.status_code}: {err_body}") from e
    raise RuntimeError(f"LLM API failed after 4 retries, last status: {last_err.response.status_code}")
