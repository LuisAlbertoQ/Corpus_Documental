FROM python:3.10-slim

# Instalar dependencias del sistema para PyMuPDF (fitz)
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario para evitar permisos de root
RUN useradd -m -s /bin/bash jupyteruser

WORKDIR /home/jupyteruser/work

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Dar permisos
RUN chown -R jupyteruser:jupyteruser /home/jupyteruser

USER jupyteruser

# Puerto de Jupyter
EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''"]