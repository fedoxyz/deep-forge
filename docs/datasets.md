# Datasets — Deep Forge

Gestiona, etiqueta y preprocesa imágenes para entrenar modelos de visión. Deep Forge soporta cuatro tipos de datasets, cada uno con su propio flujo de trabajo.

---

## Tipos de dataset

| Tipo | Uso | Formato de etiquetas |
|------|-----|----------------------|
| **Caption** | Fine-tuning de difusión (LoRA, DreamBooth) | `.txt` por imagen (texto libre) |
| **Classification** | Clasificadores CNN/ViT | Carpeta = clase |
| **Detection** | Detección de objetos | Forge `.txt` por imagen |
| **Segmentation** | Segmentación de instancias | Forge `.txt` por imagen |

Deep Forge infiere el tipo automáticamente al cargar. Puedes forzarlo con **Create New** eligiendo el tipo explícitamente.

---

## Cargar un dataset

1. Escribe o pega la ruta absoluta en el campo superior.
2. Pulsa **Load** o `Enter`.
3. El dataset aparece como pestaña. Varios datasets pueden estar cargados a la vez; ciérralos con `×`.

**Crear nuevo dataset vacío**

Pulsa **Create New**, elige nombre, directorio base y tipo. Se crea la carpeta y el archivo `.dataset_meta.json` de metadatos.

---

## Subir imágenes

Arrastra archivos sobre la zona de upload o haz clic para abrir el selector. Formatos aceptados: `png jpg jpeg webp bmp tiff`. También puedes subir archivos `.txt` de captions o etiquetas.

Para datasets de **clasificación**, sube imágenes organizadas en subcarpetas:
```
mi_dataset/
  gato/img1.jpg
  perro/img2.jpg
```
Deep Forge respeta la jerarquía y asigna la clase automáticamente.

---

## Navegación y filtros

| Control | Descripción |
|---------|-------------|
| **All / Captioned / Uncaptioned** | Filtrar por estado de etiqueta |
| **Buscador** | Filtra por texto en caption o nombre de archivo |
| **Grid size** | De 2 a 8 columnas |
| **Paginación** | 48 imágenes por página |

Selecciona imágenes con el checkbox (esquina superior izquierda de cada tarjeta). Con selección activa, el clic en la imagen también selecciona/deselecciona.

---

## Panel de captions (tipo Caption)

Cada imagen del dataset tiene un archivo `.txt` asociado con el mismo nombre base. El contenido es texto libre — una descripción de la imagen usada durante el entrenamiento.

Abre el panel derecho con el botón **Caption**. Muestra la imagen seleccionada y un editor de texto. Guarda con `Ctrl+S` o el botón **Save**.

**Batch captioning**: selecciona varias imágenes → botón **Caption Selected** → modal de generación masiva con IA.

**Análisis de conceptos** (pestaña **Analysis**): extrae n-gramas frecuentes de todos los captions. Haz clic en un concepto para filtrar las imágenes que lo contienen.

---

## Clasificación

Para datasets de tipo `classification`, cada tarjeta muestra el nombre de la clase asignada. El panel derecho muestra la distribución de clases con conteos.

---

## Herramienta de anotación (Detection / Segmentation)

Abre con el icono **□** en cada tarjeta (o al hacer clic en **Annotate**).

### Herramientas disponibles

| Tecla | Herramienta | Descripción |
|-------|-------------|-------------|
| `S` | **Select / Move** | Selecciona y arrastra anotaciones existentes |
| `B` | **Box** | Dibuja bounding boxes (YOLO detection) |
| `P` | **Polygon** | Polígono punto a punto; cierra haciendo clic cerca del primer punto o doble clic |
| `O` | **Ellipse** | Elipse convertida a máscara |
| `M` | **Paint** | Pincel de píxeles para máscaras libres |
| `E` | **Erase** | Borra píxeles de la máscara activa |

> Las herramientas Polygon, Ellipse, Paint y Erase solo están disponibles en datasets de tipo **Segmentation**.

### Flujo de trabajo con máscaras (Paint / Erase)

1. Selecciona **Paint** (`M`) y elige la clase.
2. Pinta sobre la imagen. Ajusta el tamaño del pincel con el slider **Brush**.
3. Para borrar partes de la máscara activa, cambia a **Erase** (`E`) y pinta.
4. También puedes hacer clic sobre una máscara ya guardada con Paint o Erase activo — la máscara se carga para edición y la anotación original se elimina del historial.
5. Pulsa **Commit Mask** para convertir la máscara pintada en una anotación guardada.

> Si cambias de clase con Paint activo, la máscara en curso se descarta. Haz Commit antes de cambiar.

### Editar una máscara existente

- Selecciona la anotación (lista inferior o `S` + clic).
- Con la anotación seleccionada, pulsa **Edit Mask** en la barra de herramientas — la máscara se carga en el pincel y puedes añadir o borrar píxeles.
- También puedes hacer clic directamente sobre la máscara con Paint o Erase activo para cargarla sin necesidad de seleccionarla antes.

### Atajos de teclado

| Atajo | Acción |
|-------|--------|
| `Ctrl+Z` | Deshacer |
| `Ctrl+Y` / `Ctrl+Shift+Z` | Rehacer |
| `Delete` / `Backspace` | Eliminar anotación seleccionada |
| `Escape` | Cancelar dibujo en curso / deseleccionar |

### Guardar etiquetas

Pulsa **Save Labels** en la cabecera del modal. Las etiquetas se guardan en **Forge Label Format** — un `.txt` junto a la imagen.

#### Forge Label Format

```
# Líneas con # son comentarios / metadatos

# Detection — coordenadas absolutas en píxeles (enteros)
D <class_name> <x1> <y1> <x2> <y2>

# Segmentation — bounding box del crop + RLE de la máscara
S <class_name> <crop_x1> <crop_y1> <crop_x2> <crop_y2> <rle...>
```

**RLE (Run-Length Encoding)**: pares intercalados `valor cuenta`. El píxel cubre solo la región del crop box, en orden row-major.

```
S perro 120 45 310 280 0 3600 1 892 0 1440 1 234 0 …
```
— significa: 3600 píxeles apagados, 892 encendidos, 1440 apagados, etc., dentro del recorte `(120,45)→(310,280)`.

Este formato es compacto, legible y parseable tanto desde Python como desde JavaScript sin dependencias externas.

---

## Crop to bucket (preprocesamiento)

Redimensiona y recorta imágenes a resoluciones óptimas para entrenamiento. Disponible en la barra **CropBar**.

| Preset | Rango | Paso | Uso típico |
|--------|-------|------|------------|
| SDXL | 768–1344 | 64 | Stable Diffusion XL |
| SD 1.5 | 512–768 | 64 | SD 1.5 / 2.x |
| Custom | configurable | — | Cualquier arquitectura |

- **Crop All**: procesa todas las imágenes del dataset.
- **Crop Selected**: procesa solo las imágenes seleccionadas.

Los originales se guardan automáticamente en `.originals/` antes de sobreescribir.

---

## Eliminar imágenes

- **Tarjeta individual**: icono de papelera (hover).
- **Selección múltiple**: selecciona → botón **Delete Selected** en la barra de selección.

Al borrar una imagen también se elimina su archivo de etiqueta/caption asociado.

---

## Descargar / exportar

Los archivos del dataset están en la carpeta del sistema de archivos que cargaste. Deep Forge escribe directamente allí; no hay paso de exportación extra. Para YOLO, los `.txt` están listos para usar con Ultralytics u otros frameworks compatibles.

---

## API REST (referencia rápida)

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/api/datasets/load` | Cargar directorio |
| `POST` | `/api/datasets/create` | Crear dataset vacío |
| `GET` | `/api/datasets/loaded` | Listar datasets cargados |
| `DELETE` | `/api/datasets/loaded/{id}` | Descargar de memoria |
| `GET` | `/api/datasets/{id}/entries` | Entradas paginadas (`offset`, `limit`, `filter`) |
| `GET` | `/api/datasets/{id}/thumbnails` | Miniaturas en lote |
| `PUT` | `/api/datasets/{id}/caption/{idx}` | Actualizar caption |
| `GET` | `/api/datasets/{id}/labels/{idx}` | Leer etiquetas (Forge format) |
| `PUT` | `/api/datasets/{id}/labels/{idx}` | Guardar etiquetas (Forge format) |
| `GET` | `/api/datasets/{id}/mask/{idx}` | Preview de máscara renderizada |
| `POST` | `/api/datasets/{id}/analyze` | Análisis de conceptos |
| `POST` | `/api/datasets/{id}/crop-to-bucket` | Redimensionar a buckets |
| `POST` | `/api/datasets/{id}/upload` | Subir archivos |
| `DELETE` | `/api/datasets/{id}/file/{filename}` | Borrar imagen |
| `POST` | `/api/datasets/{id}/delete-batch` | Borrar múltiples imágenes |
