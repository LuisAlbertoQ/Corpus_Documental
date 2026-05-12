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

## Ejecución

1. Inicia el contenedor con Jupyter Notebook:
   ```bash
   docker-compose up
   ```

2. Abre tu navegador en `http://localhost:8888` (sin token, como configurado).

3. Ejecuta los notebooks en orden dentro de la carpeta `notebooks/`:
   - `1_extraccion_texto.ipynb`: Extrae texto de PDFs.
   - `2_limpieza_corpus.ipynb`: Limpia el texto.
   - `3_chunking_embeddings.ipynb`: Genera chunks y embeddings.
   - `4_indexacion_vectorial.ipynb`: Indexa en ChromaDB.
   - `5_verificacion_cobertura.ipynb`: Verifica cobertura.
   - `6_prueba_manual.ipynb`: Pruebas interactivas.

4. Los resultados se guardan en `corpus_upeu/metadatos/` y `vector_store/`.

## Estructura del Proyecto

```
oe1_arquitectura_corpus/
├── corpus_upeu/           # Datos del corpus
│   ├── pdfs/             # Documentos PDF originales (no en Git)
│   ├── txt_bruto/        # Texto extraído crudo
│   ├── txt_limpio/       # Texto limpiado
│   └── metadatos/        # Chunks, embeddings y reportes
├── notebooks/            # Jupyter notebooks del pipeline
├── vector_store/         # Base de datos ChromaDB
├── src/                  # Código fuente adicional
├── Dockerfile            # Imagen Docker
├── docker-compose.yml    # Configuración de contenedor
├── requirements.txt      # Dependencias Python
└── README.md             # Este archivo
```

## Notas

- Los documentos PDF en `corpus_upeu/pdfs/` no deben subirse a Git (están en .gitignore).
- El procesamiento puede tomar tiempo dependiendo del tamaño del corpus.
- Para búsquedas, usa el notebook de pruebas manuales o integra ChromaDB en tu aplicación.
- Modelo de embeddings: `paraphrase-multilingual-MiniLM-L12-v2` (optimizado para español).

## Dependencias

Ver `requirements.txt` para las bibliotecas Python utilizadas.
