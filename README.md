# RAG sobre contratos de arrendamiento

Flujo **RAG End-to-End** con **LangChain**, **ChromaDB** y **OpenAI**: indexa contratos de arrendamiento
en PDF en una base vectorial local y responde preguntas usando **exclusivamente** esa información.
La respuesta se valida con **Pydantic** e incluye las referencias (archivo y página) de donde sale cada dato.
Si la respuesta no está en los documentos, el modelo responde **"No lo sé."** en vez de inventar.

## Estructura

```
.
├── data/               # Los 5 contratos en PDF (el "cerebro" del RAG)
├── config.py           # Rutas, modelos, tamaño de chunk y top_k
├── ingest.py           # Ingesta: PDF -> limpieza -> chunks -> ChromaDB
├── schemas.py          # Pydantic: RAGResponse y Referencia
├── rag.py              # Retriever + prompt grounded + cadena LCEL + get_rag_response
├── main.py             # Las dos pruebas: pregunta en los documentos y pregunta trampa
├── .env.example        # Plantilla de variables de entorno
└── requirements.txt
```

`vectorstore/` (la base de ChromaDB) no se sube al repositorio: se genera al ejecutar la ingesta.

## Diseño

### Ingesta (`ingest.py`)

1. **Carga:** `pypdf` lee cada PDF de `data/` y crea un `Document` **por página**, con `source` (archivo)
   y `page` (desde 1) en la metadata, para poder citar "archivo, pág. N".
2. **Limpieza (`clean_text`):** normaliza los espacios especiales (` `, `\xa0`), quita los espacios
   que `pypdf` deja antes de cada salto de línea y une las líneas cortadas a mitad de frase
   (las que continúan en minúscula).
3. **Chunking:** `RecursiveCharacterTextSplitter` con `length_function` basada en `tiktoken`
   (`cl100k_base`, el tokenizador de `text-embedding-3-small`), así el tamaño se mide en **tokens** y no
   en caracteres. Chunks de hasta **600 tokens con 100 de overlap** (el enunciado pide al menos 500 / 50).
   `split_documents` copia la metadata a cada chunk. Resultado: 15 páginas -> 25 chunks.
4. **Persistencia:** `Chroma` en `./vectorstore`, con ids estables (`archivo-pN-i`) para que reindexar
   actualice los chunks en vez de duplicarlos.
5. **Sin reindexado innecesario:** `vectorstore_existe()` comprueba si la colección ya tiene chunks
   (no solo si existe la carpeta). Si los tiene, la ingesta se omite y no se gasta API.

### Mismo modelo de embeddings para indexar y consultar

`abrir_vectorstore()` es el **único** punto donde se crea `OpenAIEmbeddings(model=EMBEDDING_MODEL)`.
La ingesta y el retriever pasan por ahí, así que es imposible indexar con un modelo y consultar con otro.

### Recuperación y generación (`rag.py`)

Cadena **LCEL**:

```
{"contexto": retriever | formatear_docs, "pregunta": passthrough} | prompt | ChatOpenAI | PydanticOutputParser
```

- **Retriever:** búsqueda por similitud con `k=4` (dentro del rango 3–5 para evitar el "contexto infinito").
- **`formatear_docs`:** une los fragmentos y antepone a cada uno `Fuente: <archivo> | Página: <n>`,
  para que el modelo pueda citarlos.
- **Prompt de sistema ("filtro de veracidad"):** responder solo con el CONTEXTO; si no está, el campo
  `respuesta` debe ser exactamente "No lo sé.", con `respuesta_en_contexto=false` y sin referencias.
- **LLM:** `gpt-4o-mini`, `temperature=0`, con el **modo JSON** de OpenAI activado.
- **`PydanticOutputParser`:** valida la salida contra `RAGResponse`.
- **`get_rag_response(query)`:** función asíncrona (`ainvoke`) que ejecuta la cadena y devuelve un `RAGResponse`.

### Salida (`schemas.py`)

```python
class RAGResponse(BaseModel):
    respuesta: str                    # texto de la respuesta o "No lo sé."
    respuesta_en_contexto: bool       # False si el contexto no contenía la respuesta
    referencias: list[Referencia]     # archivo + página de cada fragmento usado
```

## Requisitos

- Python 3.14
- Una clave de API de OpenAI

## Instalación

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

En Linux / macOS: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

## Variables de entorno

```powershell
Copy-Item .env.example .env
```

| Variable | Obligatoria | Descripción |
|---|---|---|
| `OPENAI_API_KEY` | Sí | Clave de OpenAI (embeddings y LLM) |

El archivo `.env` está en `.gitignore`: la clave nunca se sube al repositorio.

## Uso

```powershell
# Pruebas completas (ingesta automática la primera vez)
.\.venv\Scripts\python.exe main.py

# Solo ingesta / reindexar desde cero (por ejemplo, tras cambiar los PDFs de data/)
.\.venv\Scripts\python.exe ingest.py
.\.venv\Scripts\python.exe ingest.py --forzar
```

> Si la consola de Windows da errores con las tildes, ejecuta antes `$env:PYTHONIOENCODING="utf-8"`.

## Pruebas

`main.py` lanza las dos consultas en paralelo y verifica el resultado (sale con código 0 si ambas pasan):

| Prueba | Pregunta | Resultado esperado |
|---|---|---|
| En los documentos | ¿Cuál es la renta mensual de la plaza de garaje? | 120 €, citando `CONTRATO DE ARRENDAMIENTO DE PLAZA DE GARAJE.pdf` |
| Pregunta trampa | ¿Se permite tener mascotas en la vivienda alquilada? | "No lo sé.", sin referencias (ningún contrato menciona mascotas ni animales) |

Salida real:

```
=== Prueba 1: respuesta en los documentos ===
Pregunta: ¿Cuál es la renta mensual de la plaza de garaje?
Respuesta: CIENTO VEINTE EUROS (120,00 €) mensuales.
¿En el contexto?: True
  - CONTRATO DE ARRENDAMIENTO DE PLAZA DE GARAJE.pdf, pág. 2

=== Prueba 2: pregunta trampa ===
Pregunta: ¿Se permite tener mascotas en la vivienda alquilada?
Respuesta: No lo sé.
¿En el contexto?: False

=== Resultado ===
Prueba 1 (responde con 120 € y cita el contrato de garaje): OK
Prueba 2 (dice 'No lo sé' sin referencias, no alucina):    OK
```

### Problema encontrado: salida que no es JSON

En una ejecución, ante la pregunta trampa, el modelo respondió `No lo sé.` como **texto plano** en vez de
JSON y el `PydanticOutputParser` falló (`OutputParserException`). Se corrigió de dos formas:

1. El prompt exige devolver **siempre** el JSON, y aclara que "No lo sé." va dentro del campo `respuesta`.
2. Se activó el modo JSON de OpenAI (`response_format={"type": "json_object"}`), que garantiza JSON
   válido; el `PydanticOutputParser` sigue validando el esquema.

Tras el cambio, las dos pruebas pasaron en 5 ejecuciones seguidas.

## Decisiones y limitaciones

- **Un `Document` por página:** permite citar la página, a cambio de que una cláusula partida entre dos
  páginas quede en chunks distintos.
- **Contratos parecidos:** hay dos de local de negocio y dos de vivienda; el retriever puede mezclar
  fragmentos de varios contratos. Por eso el prompt pide aclarar a qué contrato pertenece cada dato y
  la respuesta incluye las referencias.
- **Limpieza conservadora:** solo se unen las líneas que continúan en minúscula; los saltos antes de una
  mayúscula o un número (p. ej. `TRES MIL SEISCIENTOS\nEUROS`) se mantienen para no romper los títulos
  de las cláusulas.
