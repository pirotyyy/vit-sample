import yaml

def load_config():
    with open('./config/config.yaml', encoding='utf-8') as f:
        conf = yaml.safe_load(f)

    return conf