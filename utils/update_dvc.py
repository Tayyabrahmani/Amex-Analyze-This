import yaml

def update_dvc_yaml():
    """Updates dvc.yaml with the latest model version from config.yaml."""

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model_version = config["model_version"]

    with open("dvc.yaml", "r") as f:
        dvc_data = f.read()

    # Replace placeholders with the actual version
    dvc_data = dvc_data.replace("${model_version}", model_version)

    with open("dvc.yaml", "w") as f:
        f.write(dvc_data)

    print(f"✅ Updated dvc.yaml with model version: {model_version}")

if __name__ == "__main__":
    update_dvc_yaml()
