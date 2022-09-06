#!/bin/bash

# python avatarSDFExporter.py
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for avatarType in `ls $SCRIPT_DIR/../datasets/avatars`; do
	for avatarName in `ls $SCRIPT_DIR/../datasets/avatars/$avatarType`; do
		python avatarSDFExporter.py $avatarType $avatarName
	done
done