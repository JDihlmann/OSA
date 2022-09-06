#!/bin/bash

# blender environments/home/home_0.blend --background --python imageRender.py
# blender environments/living/living_0.blend --background --python avatarImageExporter.py
# blender environments/private-garage/private-garage_0.blend --background --python avatarImageExporter.py
# blender environments/research/research_0.blend --background --python avatarImageExporter.py
# blender environments/workspace/workspace_0.blend --background --python avatarImageExporter.py


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for environmentName in `ls $SCRIPT_DIR/../datasets/environments`; do
	for filePath in `find $SCRIPT_DIR/../datasets/environments/$environmentName -type f -name "*.blend"`; do
		blender $filePath --background --python avatarImageExporter.py
	done
done