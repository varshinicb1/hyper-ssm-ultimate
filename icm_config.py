"""
ICM Configuration — loads from defaults, YAML, env vars, and CLI args.
"""

import os
import argparse
from dataclasses import dataclass, fields
from typing import Optional


@dataclass
class IcmConfig:
    model_name: str = "gpt2"
    embedder_name: str = "all-MiniLM-L6-v2"
    host: str = "0.0.0.0"
    port: int = 8000
    max_sessions: int = 100
    session_ttl: int = 3600
    save_dir: str = "icm_sessions"
    state_dim: int = 64
    num_scales: int = 4
    max_new_tokens: int = 512
    log_level: str = "INFO"
    stream_output: bool = True
    quantize_bits: Optional[int] = None
    sqlite_path: Optional[str] = None
    auth_enabled: bool = False
    auth_keys_path: Optional[str] = None
    rate_limit_rpm: int = 60
    memory_backend: str = "flat"


ENV_MAP = {
    "model_name": "ICM_MODEL_NAME",
    "embedder_name": "ICM_EMBEDDING_MODEL",
    "host": "ICM_HOST",
    "port": "ICM_PORT",
    "max_sessions": "ICM_MAX_SESSIONS",
    "session_ttl": "ICM_SESSION_TTL",
    "save_dir": "ICM_SAVE_DIR",
    "state_dim": "ICM_STATE_DIM",
    "num_scales": "ICM_NUM_SCALES",
    "max_new_tokens": "ICM_MAX_NEW_TOKENS",
    "log_level": "ICM_LOG_LEVEL",
    "stream_output": "ICM_STREAM_OUTPUT",
    "quantize_bits": "ICM_QUANTIZE_BITS",
    "sqlite_path": "ICM_SQLITE_PATH",
    "auth_enabled": "ICM_AUTH_ENABLED",
    "auth_keys_path": "ICM_AUTH_KEYS_PATH",
    "rate_limit_rpm": "ICM_RATE_LIMIT_RPM",
    "memory_backend": "ICM_MEMORY_BACKEND",
}


CLI_MAP = {
    "model_name": "model",
    "embedder_name": "embedder",
    "host": "host",
    "port": "port",
    "max_sessions": "max_sessions",
    "session_ttl": "session_ttl",
    "state_dim": "state_dim",
    "num_scales": "num_scales",
    "log_level": "log_level",
    "quantize_bits": "quantize_bits",
    "sqlite_path": "sqlite_path",
    "auth_enabled": "auth_enabled",
    "auth_keys_path": "auth_keys_path",
    "rate_limit_rpm": "rate_limit_rpm",
    "memory_backend": "memory_backend",
}


def _load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _load_env() -> dict:
    overrides = {}
    for key, env_var in ENV_MAP.items():
        val = os.environ.get(env_var)
        if val is not None:
            overrides[key] = val
    return overrides


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ICM Server")
    parser.add_argument("--model", "-m", help="Model name")
    parser.add_argument("--embedder", "-e", help="Embedding model")
    parser.add_argument("--host", help="Host to bind")
    parser.add_argument("--port", type=int, help="Port")
    parser.add_argument("--config", "-c", help="Path to config YAML")
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--session-ttl", type=int)
    parser.add_argument("--state-dim", type=int)
    parser.add_argument("--num-scales", type=int)
    parser.add_argument("--log-level", help="Logging level")
    parser.add_argument("--quantize-bits", type=int, choices=[4, 8], help="Quantize model to 4 or 8 bits (requires bitsandbytes)")
    parser.add_argument("--sqlite-path", help="Path to SQLite database for persistent sessions")
    parser.add_argument("--auth-enabled", action="store_true", help="Enable API key authentication")
    parser.add_argument("--auth-keys-path", help="Path to API keys JSON file")
    parser.add_argument("--rate-limit-rpm", type=int, help="Rate limit requests per minute per key")
    parser.add_argument("--memory-backend", choices=["flat", "tree"], default="flat", help="Memory backend: flat (O(1)) or tree (O(log N))")
    return parser


def _cast(key: str, value, field_type):
    if value is None:
        return None
    # Unwrap Optional[X] to get the inner type
    origin = getattr(field_type, '__origin__', None)
    if origin is not None:
        args = getattr(field_type, '__args__', ())
        for a in args:
            if a is not type(None) and callable(a):
                field_type = a
                break
    if field_type is bool:
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("1", "true", "yes")
    return field_type(value)


def get_config(args: Optional[list] = None) -> IcmConfig:
    config = IcmConfig()
    raw = {}

    # 1. YAML config file (via env var ICM_CONFIG or --config CLI)
    yaml_path = os.environ.get("ICM_CONFIG")
    if yaml_path and os.path.isfile(yaml_path):
        raw.update(_load_yaml(yaml_path))

    # 2. Environment variables
    raw.update(_load_env())

    # 3. CLI arguments
    parser = _build_parser()
    parsed, _ = parser.parse_known_args(args)

    # --config from CLI overrides env var
    if parsed.config and os.path.isfile(parsed.config):
        raw.update(_load_yaml(parsed.config))

    cli_overrides = {}
    for key, attr in CLI_MAP.items():
        val = getattr(parsed, attr, None)
        if val is not None:
            cli_overrides[key] = val
    raw.update(cli_overrides)

    # Apply overrides to config dataclass
    for f in fields(config):
        if f.name in raw:
            setattr(config, f.name, _cast(f.name, raw[f.name], f.type))

    return config
