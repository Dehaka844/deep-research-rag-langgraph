# Deep Research + RAG con LangGraph

Proyecto de aprendizaje para generar un informe de investigación en Markdown combinando
búsqueda web con Exa y documentos PDF internos indexados en Chroma. El flujo usa LangGraph
para planificar el informe, investigar varias secciones en paralelo, revisar su calidad y
redactar las secciones finales.

## Alcance y datos de ejemplo

El corpus de `rag/pdfs/` contiene dos memorandos sintéticos sobre oro y plata en 2025:

- [Análisis interno del oro](rag/pdfs/analisis_interno_oro_2025_ficticio.pdf)
- [Análisis interno de la plata](rag/pdfs/analisis_interno_plata_2025_ficticio.pdf)

El contenido y todas las cifras de esos PDFs son ficticios. Los valores representan puntos
de índices hipotéticos, no precios de mercado, datos históricos ni previsiones reales. Se
incluyó una advertencia dentro de cada documento. No uses esos valores para tomar decisiones
financieras.

## Requisitos

- Python 3.11 o posterior.
- Una clave de OpenAI para los modelos y embeddings.
- Una clave de Exa para la búsqueda web.
- Las dependencias declaradas en `requirements.txt`.

Chroma se ejecuta como base vectorial persistida en disco dentro de `rag/chroma_db/`; no
requiere un servicio Docker separado.

## Instalación y configuración

Activa el entorno virtual del proyecto e instala las dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Si `.env` todavía no existe, créalo a partir de `.env.example` y después añade las claves:

```powershell
# Ejecuta esta línea solo si .env aún no existe
Copy-Item .env.example .env
```

Edita `.env` y añade tus claves:

```dotenv
OPENAI_API_KEY=tu_clave_de_openai
EXA_API_KEY=tu_clave_de_exa
```

No compartas ni subas `.env` al repositorio. `.gitignore` excluye ese archivo y mantiene
`rag/chroma_db/` fuera de Git porque es una base generada.

## Crear la base vectorial

Desde la raíz del proyecto, ejecuta:

```powershell
python -m rag.create_db
```

El comando lee recursivamente los PDFs de `rag/pdfs/`, los divide en fragmentos, calcula
embeddings con OpenAI y persiste la colección `deep_research_documents` en
`rag/chroma_db/`. Si termina correctamente, imprime `{'status': 200}`.

Se pueden indicar otras rutas:

```powershell
python -m rag.create_db --pdf-dir "ruta\a\pdfs" --persist-dir "ruta\a\chroma_db"
```

Al volver a indexar los mismos archivos, elimina primero la base anterior si quieres evitar
duplicados en la colección.

## Ejecutar el flujo completo

Una vez creada Chroma, ejecuta:

```powershell
python test.py
```

El script usa el tema de prueba del enunciado, compila el grafo con `MemorySaver`, transmite
el progreso y reanuda automáticamente la interrupción de revisión del plan aprobándolo. Al
final imprime el informe Markdown. Antes de iniciar llamadas a servicios externos, comprueba
que existan PDFs y una base Chroma.

La configuración de prueba usa GPT-4.1 con OpenAI y Exa con los dominios `eleconomista.es`
y `expansion.com`. El plan solicita dos búsquedas iniciales; cada una de las tres secciones
de investigación puede hacer una búsqueda inicial y, si la reflexión la pide, una de
seguimiento. El máximo previsto es de ocho consultas a Exa para un plan válido. Una consulta
puede devolver hasta 15 resultados, configurados en `search_api_config`.

## Estructura

```text
config/
  configuration.py   Configuración del grafo y carga de .env
graph/
  graph.py           Grafo exterior y subgrafo de investigación
  nodes.py           Planificación, búsqueda, reflexión, RAG y redacción
prompts/
  prompts.py         Instrucciones para los modelos
tools/
  state.py           Modelos Pydantic y estados TypedDict
rag/
  pdfs/              PDFs de entrada
  chroma_db/         Base vectorial persistida, generada localmente
  create_db.py       Ingesta, división y vectorización de PDFs
  get_data.py        Recuperación por similitud con metadatos
test.py              Ejecución del flujo y aprobación automática del plan
```

## Flujo del grafo

1. Genera consultas de planificación y busca contexto web.
2. Propone las secciones del informe y pausa para revisión humana.
3. Investiga en paralelo las secciones marcadas con `research=True`.
4. Evalúa cada sección; si hace falta evidencia adicional, vuelve a buscar dentro del límite
   configurado.
5. Recupera fragmentos relevantes de Chroma y los añade como «Datos desde archivos».
6. Redacta en español las secciones finales sin investigación web y compila el informe en el
   orden del plan, con la sección RAG antes de las conclusiones.

## Módulos principales

- `retrieve_from_db(topic, persist_dir, k=10)` devuelve fragmentos con fuente, página y autor.
- `create_vector_db(pdf_dir, persist_dir)` permite parametrizar tanto la entrada PDF como la
  ruta de persistencia de Chroma.
- `ReportState.completed_sections` usa `operator.add` para acumular los resultados paralelos.
- `test.py` limita a una las consultas por sección y las consultas de seguimiento; el plan se
  valida antes de reanudar la investigación.

## Avisos conocidos

`PyPDFDirectoryLoader` procede de `langchain-community`, que puede mostrar una advertencia
de deprecación en la versión instalada. Esa advertencia no impidió la ingesta: el comando
terminó con `{'status': 200}`.

El `MemorySaver` de `test.py` registra explícitamente el modelo `Section` para la
serialización del checkpoint de LangGraph.
