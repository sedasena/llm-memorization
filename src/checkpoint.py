import os
import json 

CHECKPOINT_FILE = "/content/drive/MyDrive/checkpoint_{}.json"

def load_checkpoint(model_name):
    path = CHECKPOINT_FILE.format(model_name)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        print(f"Checkpoint loaded> {len(data)} samples")
        return data
    return []

def save_checkpoint(results, model_name):
    path = CHECKPOINT_FILE.format(model_name)
    with open(path, "w") as f:
        json.dump(results, f)