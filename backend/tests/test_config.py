"""Configuration must survive the values the documentation tells people to use.

A settings field that only accepts JSON is a field nobody can configure from
a .env file, and the failure is a SettingsError at import time — the whole
application, not one endpoint.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent

# Settings is cached with lru_cache and read at import, so each case runs in a
# fresh interpreter with a controlled environment — the same way the API and
# the worker start.
_PROBE = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, %r)
    from app.core.config import Settings
    settings = Settings()
    print(repr(settings.cors_origin_list))
    """
) % str(BACKEND)


def _load(**env: str) -> str:
    environment = {
        k: v for k, v in os.environ.items() if k not in {"CORS_ORIGINS", "DATA_DIR"}
    }
    environment["DATA_DIR"] = env.pop("DATA_DIR", "/tmp/padel-config-test")
    environment.update(env)
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=environment,
        capture_output=True,
        text=True,
        cwd=str(BACKEND),
    )
    if result.returncode != 0:
        raise AssertionError(f"Settings() non caricabile:\n{result.stderr}")
    return result.stdout.strip()


def test_unset_cors_origins_means_same_origin_only():
    assert _load() == "[]"


def test_empty_cors_origins_is_accepted():
    """`CORS_ORIGINS=` is exactly what .env.example ships, and it used to
    abort startup with a SettingsError — in Docker too, since compose passes
    the same file through as environment variables."""
    assert _load(CORS_ORIGINS="") == "[]"


def test_a_plain_url_is_accepted():
    """The README tells you to set CORS_ORIGINS=http://localhost:5173 for the
    Vite dev server. A list-typed field would have demanded JSON."""
    assert _load(CORS_ORIGINS="http://localhost:5173") == "['http://localhost:5173']"


def test_comma_separated_origins_are_split_and_trimmed():
    assert _load(CORS_ORIGINS="http://a:1, http://b:2") == "['http://a:1', 'http://b:2']"


@pytest.mark.parametrize("value", ["   ", ",", " , "])
def test_blank_entries_are_dropped(value):
    assert _load(CORS_ORIGINS=value) == "[]"


def test_env_example_is_loadable_as_is(tmp_path):
    """Whatever ships as .env.example must start the app unchanged: it is
    what every install copies."""
    example = (BACKEND.parent / ".env.example").read_text()
    env_file = tmp_path / ".env"
    env_file.write_text(example)

    probe = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(BACKEND)!r})
        from pydantic_settings import SettingsConfigDict
        from app.core.config import Settings

        class FromExample(Settings):
            model_config = SettingsConfigDict(env_file={str(env_file)!r}, extra="ignore")

        settings = FromExample()
        print(settings.sample_hz, settings.cors_origin_list)
        """
    )
    environment = {
        k: v for k, v in os.environ.items() if not k.isupper() or k in {"PATH", "HOME"}
    }
    result = subprocess.run(
        [sys.executable, "-c", probe], env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, f".env.example non caricabile:\n{result.stderr}"
    assert "5.0" in result.stdout
