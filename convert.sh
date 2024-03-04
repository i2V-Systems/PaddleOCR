#!/bin/bash

# Setting up paths
TRAIN_MODEL_PATH="./output/v3_en_mobile/best_model/model"
SAVE_INFERENCE_DIR="./inference/rec_crnn"

# Step 1: Training model to inference model
python tools/export_model.py \
    -c configs/rec/custom_configs/rec_en_number_lite_train.yml \
    -o Global.pretrained_model="$TRAIN_MODEL_PATH" \
    Global.save_inference_dir="$SAVE_INFERENCE_DIR" 

# Step 2: Making input shape static
python ./PaddleUtils/paddle/paddle_infer_shape.py \
    --model_dir "$SAVE_INFERENCE_DIR" \
    --model_filename inference.pdmodel \
    --params_filename inference.pdiparams \
    --save_dir "$SAVE_INFERENCE_DIR" \
    --input_shape_dict="{'x': [-1, 3, 32, 100]}"

# Step 3: inference model to onnx model
paddle2onnx \
            --model_dir "$SAVE_INFERENCE_DIR" \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file "$SAVE_INFERENCE_DIR"/inference.onnx \
            --opset_version 12 \
            --enable_onnx_checker=True \
            --input_shape_dict="{'x': [-1, 3, 32, 100]}"

## setting permissions
# chmod +x convert.sh
