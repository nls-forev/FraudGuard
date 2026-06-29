import yaml


def read_yaml_file(file_path: str):
    try:
        with open(file_path, "rb") as f:
            return yaml.safe_load(f)

    except Exception as e:
        raise e
