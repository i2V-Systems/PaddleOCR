# recommended paddle.__version__ == 2.0.0
python tools/train.py \
-c ./configs/rec/custom_configs/rec_en_number_lite_train.yml \
# -o Global.checkpoints=./output/v3_en_mobile/latest \
# -o Global.pretrained_model=../en_PP-OCRv3_rec_train/best_accuracy

# For training on multiple gpus
# python3 -m paddle.distributed.launch --log_dir=./debug/ --gpus '0,1'  tools/train.py \
# -c ./configs/rec/custom_configs/rec_en_number_lite_train.yml