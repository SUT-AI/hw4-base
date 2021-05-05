import json

CONFIG_FILE = "tester_config.json"

config = {
    "version": 2,
    "packages": [{
        "name": f"accuracy > {i}",
        "score": 1,
        "tests": [f"test_sequence_{i:02d}_accuracy"],
    } for i in range(100)],
}

with open(CONFIG_FILE, 'w') as file:
    json.dump(config, file)
