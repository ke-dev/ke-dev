#!/bin/sh

image_name=mmsegmentation:0.24.1
container_name=mmsegmentation

#nvidia-docker run \
docker run \
--name $container_name \
-d \
--runtime=nvidia \
-v /etc/localtime:/etc/localtime \
-v /home/cdl/workspace/mmsegmentation/data/:/workspace/data/:rw \
-v /home/cdl/workspace/mmsegmentation/results/:/workspace/results/:rw \
-v /home/cdl/workspace/mmsegmentation/checkpoints/:/workspace/checkpoints/:rw \
-v /home/cdl/workspace/mmsegmentation/deploy/:/workspace/deploy/:rw \
-v /home/cdl/workspace/mmsegmentation/train/:/workspace/train/:rw \
-v /home/cdl/workspace/mmsegmentation/models/:/workspace/models/:rw \
-v /home/cdl/workspace/mmsegmentation/docker/:/workspace/docker/:rw \
-v /home/cdl/workspace/mmsegmentation/scripts/:/workspace/scripts/:rw \
-v /home/cdl/workspace/mmsegmentation/test_picture/:/workspace/test_picture/:rw \
--gpus '"device=0"' \
--ipc host \
--restart unless-stopped \
-t $image_name
