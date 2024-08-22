git pull
docker build -t sam2 .
docker run --rm -it -v /tmp/.X11-unix:/tmp/.X11-unix  -e DISPLAY=$DISPLAY --gpus all --runtime=nvidia -p 8888:8888 sam2:latest bash -c "python3 -c 'import torch; print(torch.cuda.is_available(), torch.__version__);' && nvcc --version"
