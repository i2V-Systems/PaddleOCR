#!bin/bash

read -p "Enter work directory path : " work_dir_path # /home/i2v-admin/Documents/Raushan/OCR/PaddleOCR
                                                            
echo "work_dirs path is: "$work_dir_path

# check if work_dirs given by user exist
if [ ! -d $work_dir_path ] 
then
    echo "Directory $work_dir_path DOES NOT exists." 
    exit 1
fi

read -p "Enter dataset dir path : " data_dir_path # /home/i2v-admin/Documents/Raushan/OCR/dataset
                                                    
echo "dataset dir path is: "$data_dir_path

# check if dataset dir given by user exist
if [ ! -d $data_dir_path ] 
then
    echo "Directory $data_dir_path DOES NOT exists." 
    exit 1
fi

read -p "Enter inference & trained models dir path : " infer_and_trained_models # /home/i2v-admin/Documents/Raushan/OCR/models
                                                    
echo "inference & trained models dir path is: "$infer_and_trained_models

# check if pretrained models dir given by user exist
if [ ! -d $infer_and_trained_models ] 
then
    echo "Directory $infer_and_trained_models DOES NOT exists." 
    exit 1
fi


image_name="i2vdocker/paddle"
tag="1.0"

full_image_name="${image_name}:${tag}"
echo "image name: ${full_image_name}"

# for displaying window from docker (comment if not needed)
xhost +

# check if image exists locally
if [[ "$(docker images ${full_image_name} | grep ${image_name} 2> /dev/null)" == "" ]]; then
    echo "pulling ${full_image_name} from docker hub"
    docker pull ${full_image_name}
fi


nvidia-docker rm ppocr

nvidia-docker run -it -p 8888:8888 --name ppocr \
  --gpus all \
  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY -e XAUTHORITY -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v $work_dir_path:/home/PaddleOCR \
  -v $infer_and_trained_models:/home/models \
  -v $data_dir_path:/home/dataset \
  ${full_image_name}

# To train model, run:
# bash ./train.sh