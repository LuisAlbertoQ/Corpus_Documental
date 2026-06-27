# Arquitectura de Corpus UPeU

Este proyecto implementa un sistema de procesamiento e indexación de un corpus de documentos universitarios de la Universidad Peruana Unión (UPeU). Convierte documentos PDF en un corpus de texto limpio, dividido en fragmentos (chunks), con embeddings vectoriales para búsqueda semántica utilizando ChromaDB.

## Descripción

El sistema sigue un pipeline de procesamiento:
1. **Extracción de texto**: Convierte PDFs a texto bruto.
2. **Limpieza**: Normaliza y limpia el texto extraído.
3. **Chunking**: Divide el texto en fragmentos con superposición.
4. **Embeddings**: Genera representaciones vectoriales usando modelos de transformers.
5. **Indexación**: Almacena los embeddings en una base vectorial para búsquedas semánticas.
6. **Verificación**: Evalúa la cobertura del corpus con consultas de prueba.

Está diseñado para documentos en español y categoriza el contenido en áreas como estatutos, estudiantes, investigación, etc.

## Requisitos Previos

- Docker y Docker Compose instalados.
- Al menos 4GB de RAM disponible (para procesamiento de embeddings).
- Espacio en disco: ~2GB para datos y modelos.

## Instalación

1. Clona este repositorio:
   ```bash
   git clone <url-del-repositorio>
   cd oe1_arquitectura_corpus
   ```

2. Construye la imagen Docker:
   ```bash
   docker-compose build
   ```

3. Coloca los documentos PDF en la carpeta `corpus_upeu/pdfs/` (esta carpeta está montada en el contenedor).

4. **(Opcional) GPU:** Si tienes NVIDIA Container Toolkit y driver 525+, las evaluaciones
   (notebooks 7-9) usarán GPU automáticamente. El `docker-compose.yml` ya incluye GPU
   passthrough (PyTorch 2.1.2 + CUDA 12.1 dentro del contenedor).

## Ejecución

1. Inicia el contenedor con Jupyter Notebook:
    ```bash
    docker-compose up
    ```

2. Abre tu navegador en `http://localhost:8888` (sin token, como configurado).

3. Ejecuta los notebooks en orden dentro de la carpeta `notebooks/`:
    - `1_extraccion_texto.ipynb`: Extrae texto de PDFs con `pdfplumber`.
    - `2_limpieza_corpus.ipynb`: Normaliza y limpia el texto extraído.
    - `3_chunking_embeddings.ipynb`: Chunking estructural (300 tokens, overlap 80).
    - `4_indexacion_vectorial.ipynb`: Indexa en ChromaDB con mpnet (768 dim).
    - `5_verificacion_cobertura.ipynb`: Verifica cobertura con umbrales 0.30/0.50.
    - `6_prueba_manual.ipynb`: Pruebas interactivas.
    - `7_evaluacion_miniLM_gpu.ipynb` (opcional): Evaluación MiniML-L12 con GPU.
    - `8_evaluacion_mpnet_gpu.ipynb` (opcional): Evaluación mpnet con GPU.
    - `9_evaluacion_bge_m3_gpu.ipynb` (opcional): Evaluación bge-m3 con GPU.

4. Los resultados se guardan en:
    - `corpus_upeu/metadatos/` — chunks, embeddings, informes de cobertura
    - `vector_store/` — índice principal (768 dim, mpnet)
    - `vector_store_miniLM/`, `vector_store_mpnet/`, `vector_store_bge_m3/` — índices de evaluación

## Estructura del Proyecto

```
oe1_arquitectura_corpus/
├── corpus_upeu/           # Datos del corpus
│   ├── pdfs/             # Documentos PDF originales (no en Git)
│   ├── txt_bruto/        # Texto extraído crudo
│   ├── txt_limpio/       # Texto limpiado
│   └── metadatos/        # Chunks, embeddings y reportes
├── notebooks/            # Jupyter notebooks del pipeline (1-6) y evaluación (7-9)
├── vector_store/         # Índice ChromaDB principal (768 dim, mpnet)
├── vector_store_miniLM/  # Índice ChromaDB para evaluación MiniLM (opcional)
├── vector_store_mpnet/   # Índice ChromaDB para evaluación mpnet (opcional)
├── vector_store_bge_m3/  # Índice ChromaDB para evaluación bge-m3 (opcional)
├── src/                  # Código fuente adicional
├── DIAGNOSTICO_CAMBIOS.md    # Documentación técnica de bugs y optimizaciones
├── RESUMEN_FINAL.md      # Resumen ejecutivo para stakeholders
├── Dockerfile            # Imagen Docker (PyTorch 2.1.2 + CUDA 12.1)
├── docker-compose.yml    # Configuración de contenedor (con GPU passthrough)
├── requirements.txt      # Dependencias Python
└── README.md             # Este archivo
```

## Notas

- Los documentos PDF en `corpus_upeu/pdfs/` no deben subirse a Git (están en .gitignore).
- El procesamiento puede tomar tiempo dependiendo del tamaño del corpus.
- Para búsquedas, usa el notebook de pruebas manuales o integra ChromaDB en tu aplicación.
- **Modelo de embeddings recomendado:** `paraphrase-multilingual-mpnet-base-v2` (768 dim, coseno).
  Se evaluaron 3 modelos en GPU: MiniLM (384 dim, 8/10), mpnet (768 dim, 10/10), bge-m3 (1024 dim, 10/10).
  mpnet fue el ganador: mejor distancia promedio (0.29) y mejor match por categoría (7/10 en pruebas, 8/10 en producción).
- **Métrica de distancia:** coseno (distancia = 1 - coseno). Umbrales: excelente < 0.30, aceptable < 0.50.

## Dependencias

Ver `requirements.txt` para las bibliotecas Python utilizadas.

## Resultados Finales (producción con mpnet)

| Métrica | Inicio | Final |
|---|---|---|
| Cobertura del banco de pruebas | 0/10 (falso positivo) | **10/10 (100%)** |
| Excelentes (d < 0.30) | 0 | **6/10 (60%)** |
| Categoría match | n/a | **8/10 (80%)** |
| Distancia promedio | 0.59 (ruido) | **0.2954** |
| Modelo de embeddings | all-MiniLM-L6-v2 (inglés, por defecto) | **paraphrase-multilingual-mpnet-base-v2** |

Ver `DIAGNOSTICO_CAMBIOS.md` para el análisis técnico completo de los 11 issues corregidos.
Ver `RESUMEN_FINAL.md` para un resumen ejecutivo del proyecto.
