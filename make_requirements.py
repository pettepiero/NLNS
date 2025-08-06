import os
import re
import yaml

site_packages = "/home/pettepiero/tirocinio/.venv/lib/python3.12/site-packages/"
output_file = "environment.yml"

packages = []

# Updated regex to capture full version (e.g., 3.0.6, 2024.10.3, etc.)
pattern = re.compile(r"(.+?)-((?:\d+\.)*\d+).*\.dist-info")

for item in os.listdir(site_packages):
    if item.endswith(".dist-info"):
        match = pattern.match(item)
        if match:
            name = match.group(1).replace("_", "-")
            version = match.group(2)
            packages.append(f"{name}={version}")

# Remove duplicates and sort
packages = sorted(set(packages))

env = {
    "name": "my-conda-env",
    "channels": ["defaults", "conda-forge"],
    "dependencies": packages
}

with open(output_file, "w") as f:
    yaml.dump(env, f, default_flow_style=False)

print(f"Written {len(packages)} packages to {output_file}")

