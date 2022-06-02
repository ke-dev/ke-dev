cd ../docker
image_name=mmsegmentation:0.24.1
docker build --progress=plain -t $image_name .
