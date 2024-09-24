# Dockerfile
FROM tensorflow/tensorflow:2.12.0-gpu

# Instala las dependencias necesarias
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    qtbase5-dev \
    qtchooser \
    qt5-qmake \
    qtbase5-dev-tools \
    libsm6 \
    libxrender1 \
    software-properties-common \
    libnss3 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Instala Spyder y otras bibliotecas necesarias
RUN pip install numpy scipy Pillow cython matplotlib scikit-image opencv-python h5py imgaug IPython[all] spyder mediapipe
RUN pip uninstall jax jaxlib -y
RUN pip install jax jaxlib
# Establece el directorio de trabajo
WORKDIR /workspace
