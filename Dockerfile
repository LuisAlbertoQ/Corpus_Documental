FROM python:3.10-slim

# Instalar dependencias del sistema para PyMuPDF (fitz) y CUDA
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        mupdf-tools \
        libmupdf-dev \
        libfreetype6-dev \
        libjpeg62-turbo-dev \
        libpng-dev \
        libtiff-dev \
        libopenjp2-7-dev \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-spa \
        # Utilidades para verificar GPU dentro del contenedor
        pciutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/jupyteruser/work

# >>> GPU: Instalar PyTorch con CUDA 12.1 ANTES de requirements.txt
# Esto sobrescribe la version CPU-only que pip instalaria desde PyPI.
# Compatible con drivers NVIDIA 525+ (RTX 30/40 series y la mayoria de GTX 16+).
# Para cambiar la version de CUDA, ver: https://pytorch.org/get-started/locally/
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Copiar requirements e instalar (sentence-transformers usara el torch CUDA ya instalado)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Crear usuario no-root
RUN useradd -m -s /bin/bash jupyteruser && \
    chown -R jupyteruser:jupyteruser /home/jupyteruser

USER jupyteruser

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''"]
