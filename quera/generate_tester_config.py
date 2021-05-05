import json

CONFIG_FILE = "tester_config.json"

config = {
    "version": 2,
    "packages": [{
        "name": f"accuracy_{i}",
        "score": 1,
        "tests": [f"accuracy_{i}"],
    } for i in range(100)],
}

with open(CONFIG_FILE, 'w') as file:
    json.dump(config, file)
