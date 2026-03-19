import yaml
from src.utils import paths


def load_config():
    try:
        with open(paths.CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError("Configuration file is empty.")

        return config

    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {paths.CONFIG_PATH}")

    except yaml.YAMLError as e:
        raise ValueError(f"Error while parsing YAML file: {e}")

    except Exception as e:
        raise RuntimeError(f"Failed to load configuration: {e}")
