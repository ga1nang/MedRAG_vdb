import yaml

def load_config(path="./config/rag_config.yaml"):
    print(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)