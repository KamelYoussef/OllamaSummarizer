import yaml

def load_config(file_path="web/config.yaml"):
    with open(file_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config