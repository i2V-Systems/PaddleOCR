#!/bin/bash
echo "creating paddleocr docker image ..."

image_name="i2vdocker/paddle"
tag="1.0"

full_image_name="${image_name}:${tag}"

# docker rmi ${full_image_name}
nvidia-docker build -t ${full_image_name} -f ./deploy/docker/hubserving/gpu/Dockerfile .