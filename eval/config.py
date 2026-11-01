"""Connection settings. Values come from environment variables, falling back to docker/.env."""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()
    return values


@dataclass(frozen=True)
class PgConfig:
    host: str
    port: int
    ro_user: str
    ro_password: str
    admin_user: str
    admin_password: str


def secret(name: str) -> str:
    """An API key or password from the environment or docker/.env. Raises if missing; never prints it."""
    value = {**_read_dotenv(ROOT / "docker" / ".env"), **os.environ}.get(name, "")
    if not value:
        raise RuntimeError(f"{name} is not set (environment variable or docker/.env)")
    return value


def pg_config() -> PgConfig:
    env = {**_read_dotenv(ROOT / "docker" / ".env"), **os.environ}
    return PgConfig(
        host=env.get("PG_HOST", "127.0.0.1"),
        port=int(env.get("PG_PORT", "5433")),
        ro_user=env.get("PG_RO_USER", "sql_ro"),
        ro_password=env.get("PG_RO_PASSWORD", ""),
        admin_user=env.get("PG_ADMIN_USER", "postgres"),
        admin_password=env.get("PG_ADMIN_PASSWORD", ""),
    )
