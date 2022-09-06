#!/bin/bash

MODEL_CHECKPOINT_BEGIN=$1

if [ -z "$2"]; then
	MODEL_CHECKPOINT_END=$1
else
	MODEL_CHECKPOINT_END=$2
fi

mkdir "models_download"

# loop over model checkpoints
for MODELS in `ls models`; do
	mkdir "models_download/$MODELS"
	cp "models/$MODELS/checkpoint" "models_download/$MODELS/checkpoint"
	
	for i in $(seq $MODEL_CHECKPOINT_BEGIN $MODEL_CHECKPOINT_END); do 
		printf -v MODEL_NUMBER "%04d" $i
		cp "models/$MODELS/cp-$MODEL_NUMBER.ckpt.index" "models_download/$MODELS/cp-$MODEL_NUMBER.ckpt.index"
		cp "models/$MODELS/cp-$MODEL_NUMBER.ckpt.data-00000-of-00001" "models_download/$MODELS/cp-$MODEL_NUMBER.ckpt.data-00000-of-00001"
	done
	
done

zip -r "models.zip" "models_download"
# rm -r "models_download"