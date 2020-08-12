##

## GPU, Tensorflow in FC31

Using virtual environment install of Tensorflow 2.1 without GPU support as it is problematic currently in FC31, see the different things tried.


### Docker install of tensorflow
podman pull tensorflow/tensorflow:latest-py3-jupyter
podman run -it -p 8888:8888 --name drawreader tensorflow/tensorflow:nightly-py3-jupyter
This is blocked, python 3.6 fails, probably Podman issue

### Docker Nvidia install
nvidia-docker not build for Fedora, centered on production systems (Ubuntu 18.xx)

### Local Cuda install
wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
sudo sh cuda_10.2.89_440.33.01_linux.run

This fails GCC version issues, only supports 8.2, wait till available?
