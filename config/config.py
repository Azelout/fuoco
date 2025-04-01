import yaml

def load_config(config_path='K:/TIPE/fuoco/config/config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config()

if __name__ == "__main__":
    print(config)