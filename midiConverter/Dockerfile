FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# First install core numerical libraries
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy>=1.12.0 \
    pandas>=2.2.1

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.2.1 \
    torchvision==0.17.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install audio processing libraries
RUN pip install --no-cache-dir \
    librosa>=0.10.1 \
    pretty_midi>=0.2.10 \
    soundfile>=0.12.1 \
    audioread>=3.0.1

# Install ML and visualization libraries
RUN pip install --no-cache-dir \
    scikit-learn>=1.4.1.post1 \
    optuna>=3.5.0 \
    matplotlib>=3.8.3 \
    seaborn>=0.13.2 \
    plotly>=5.19.0 \
    tensorboard>=2.15.1

# Install development and utility tools
RUN pip install --no-cache-dir \
    python-dotenv>=1.0.1 \
    IPython>=8.21.0 \
    jupyter>=1.0.0 \
    pytest>=8.0.2 \
    flake8>=7.0.0 \
    tqdm>=4.66.2

# Install computer vision and documentation tools
RUN pip install --no-cache-dir \
    opencv-python>=4.9.0.80 \
    sphinx>=7.2.6 \
    sphinx_rtd_theme>=2.0.0

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# The container will start with a shell by default
CMD ["bash"] 