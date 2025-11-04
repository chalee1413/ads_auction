"""
Configuration Module

Loads environment variables from .env file for API keys and configuration.
"""

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")


def load_env():
    """Load environment variables from .env file."""
    if DOTENV_AVAILABLE:
        # Try multiple locations for .env file
        possible_paths = [
            Path(__file__).parent.parent / ".env",  # Project root
            Path(__file__).parent / ".env",  # Demo folder
            Path(".env"),  # Current directory
            Path(__file__).parent.parent / "demo" / ".env",  # Demo folder (absolute)
        ]

        for env_path in possible_paths:
            if env_path.exists():
                load_dotenv(env_path)
                print("Loaded environment variables from .env file")
                return

        print("No .env file found. Using system environment variables.")
    else:
        print("python-dotenv not available. Using system environment variables.")


def get_kaggle_credentials() -> dict:
    """
    Get Kaggle API credentials from environment variables or kaggle.json file.

    Returns:
        Dictionary with 'username' and 'key'
    """
    load_env()

    # Try environment variables first
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if username and key:
        return {"username": username, "key": key}

    # Try to read from kaggle.json file
    kaggle_json_paths = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path.home() / ".config" / "kaggle" / "kaggle.json",
        Path(".kaggle") / "kaggle.json",
    ]

    json_path_env = os.getenv("KAGGLE_JSON_PATH")
    if json_path_env:
        kaggle_json_paths.insert(0, Path(json_path_env).expanduser())

    for json_path in kaggle_json_paths:
        if json_path.exists():
            try:
                import json

                with open(json_path, "r") as f:
                    creds = json.load(f)
                    return {"username": creds.get("username"), "key": creds.get("key")}
            except Exception as e:
                print(f"Warning: Could not read {json_path}: {e}")
                continue

    return {}


def setup_kaggle_credentials():
    """
    Set up Kaggle credentials from environment variables.

    This creates or updates kaggle.json file with credentials from .env
    """
    creds = get_kaggle_credentials()

    if creds and creds.get("username") and creds.get("key"):
        # Create .kaggle directory if it doesn't exist
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)

        # Create kaggle.json
        kaggle_json = kaggle_dir / "kaggle.json"
        import json

        with open(kaggle_json, "w") as f:
            json.dump(creds, f, indent=2)

        # Set permissions
        os.chmod(kaggle_json, 0o600)

        print(f"Kaggle credentials set up in {kaggle_json}")
        return True
    else:
        print("Kaggle credentials not found in environment variables.")
        print("Please set KAGGLE_USERNAME and KAGGLE_KEY in .env file")
        return False


def get_api_key(api_name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get API key from environment variable.

    Args:
        api_name: Name of the API (e.g., 'EXCHANGE_RATE_API_KEY')
        default: Default value if not found

    Returns:
        API key or None
    """
    load_env()
    return os.getenv(api_name, default)


# Auto-load on import
if DOTENV_AVAILABLE:
    load_env()
