"""Módulo de ingesta: data/ -> chunks -> ChromaDB persistente en vectorstore/.

Uso: .venv\\Scripts\\python.exe ingest.py
"""
import re

import tiktoken
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
    VECTORSTORE_DIR,
)

# Tokenizador de text-embedding-3-small: así los 500 tokens se cuentan
# igual que los cuenta el modelo de embeddings.
TOKENIZER = tiktoken.get_encoding("cl100k_base")


def contar_tokens(texto: str) -> int:
    """Cuenta los tokens de un texto (la usa el splitter para medir los chunks)."""
    return len(TOKENIZER.encode(texto))


def clean_text(text: str) -> str:
    """Limpia la "basura" que deja pypdf al extraer el texto de los contratos."""

    # (a) Espacios especiales (espacio fino  , espacio duro \xa0) -> espacio normal.
    #     Aparecen, por ejemplo, en "1.º B".
    text = re.sub(r"[ \xa0]", " ", text)

    # (b) Varios espacios o tabulaciones seguidos -> un solo espacio.
    text = re.sub(r"[ \t]+", " ", text)

    # (c) Quitar los espacios pegados a un salto de línea: pypdf deja " \n"
    #     al final de cada línea.
    text = re.sub(r" *\n *", "\n", text)

    # (d) Unir las líneas cortadas a mitad de frase: si la línea siguiente empieza
    #     en minúscula, el salto lo puso el ancho de página, no el autor
    #     ("con domicilio a\nefectos" -> "con domicilio a efectos").
    text = re.sub(r"\n(?=[a-záéíóúüñ])", " ", text)

    # (e) Tres o más saltos seguidos -> dos. No borramos todos los saltos
    #     porque el splitter los usa para saber dónde cortar.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def cargar_documentos() -> list[Document]:
    """Lee los .pdf de DATA_DIR con pypdf y devuelve un Document por página.

    Cada Document lleva en metadata el archivo ("source") y la página ("page",
    contando desde 1) para poder citarlos en la respuesta.
    """
    documentos = []

    for ruta in sorted(DATA_DIR.glob("*.pdf")):
        lector = PdfReader(ruta)

        for numero, pagina in enumerate(lector.pages, start=1):
            texto = pagina.extract_text() or ""  # extract_text puede devolver None

            if not texto.strip():  # página vacía o escaneada: nada que indexar
                continue

            documentos.append(
                Document(
                    page_content=texto,
                    metadata={"source": ruta.name, "page": numero},
                )
            )

    if not documentos:
        raise FileNotFoundError(f"No se encontró texto en ningún PDF de {DATA_DIR}")

    return documentos


def fragmentar(documentos: list[Document]) -> list[Document]:
    """Limpia cada documento y lo corta en chunks de hasta CHUNK_SIZE tokens
    con CHUNK_OVERLAP de solapamiento. Cada chunk conserva source y page."""

    # 1. Limpiar el texto de cada documento (sin modificar los originales)
    limpios = [
        Document(page_content=clean_text(doc.page_content), metadata=doc.metadata)
        for doc in documentos
    ]

    # 2. Crear el splitter. Argumentos por nombre: en posición, el 1.º y el 2.º
    #    serían separators y keep_separator. length_function hace que mida en
    #    tokens; sin ella mediría en caracteres.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=contar_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator="end",  # el ". " queda al final del chunk, no al inicio del siguiente
    )

    # 3. Cortar: split_documents copia la metadata (source, page) a cada chunk
    chunks = splitter.split_documents(limpios)

    # 4. Resumen en vez de imprimir cada chunk entero
    tokens = [contar_tokens(c.page_content) for c in chunks]
    print(
        f"{len(documentos)} páginas -> {len(chunks)} chunks "
        f"(tokens: mín {min(tokens)}, máx {max(tokens)})"
    )

    return chunks


def abrir_vectorstore() -> Chroma:
    """Abre (o crea vacía) la colección persistente en VECTORSTORE_DIR.

    Es el ÚNICO sitio donde se elige el modelo de embeddings: ingesta y consulta
    pasan por aquí, así que es imposible que usen modelos distintos.
    Abrirla no llama a la API; solo se gasta al añadir documentos o al buscar.
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )


def vectorstore_existe() -> bool:
    """True si la colección ya está poblada, para no reindexar (ni pagar) de nuevo.

    No basta con mirar si existe la carpeta: puede haber quedado vacía o a medias.
    limit=1 evita traerse todos los chunks: solo importa si hay al menos uno.
    """
    return bool(abrir_vectorstore().get(limit=1)["ids"])


def generar_ids(chunks: list[Document]) -> list[str]:
    """Ids estables "archivo-pN-i": reingerir actualiza los chunks en vez de duplicarlos."""
    ids = []
    contador: dict[tuple[str, int], int] = {}

    for chunk in chunks:
        clave = (chunk.metadata["source"], chunk.metadata["page"])
        contador[clave] = contador.get(clave, 0) + 1
        ids.append(f"{clave[0]}-p{clave[1]}-{contador[clave]}")

    return ids


def ingestar(forzar: bool = False) -> None:
    """Puebla la colección COLLECTION_NAME en VECTORSTORE_DIR.

    Si ya está poblada y no se fuerza, no hace nada (ni llama a la API).
    Con forzar=True vacía la colección y la vuelve a crear desde data/.
    """
    if not forzar and vectorstore_existe():
        print(f"Vectorstore ya poblado en {VECTORSTORE_DIR}: se omite la ingesta.")
        return

    chunks = fragmentar(cargar_documentos())

    vectorstore = abrir_vectorstore()
    if forzar:
        vectorstore.reset_collection()  # borra también chunks de PDFs eliminados

    vectorstore.add_documents(chunks, ids=generar_ids(chunks))  # aquí se calculan los embeddings
    print(f"Ingesta completada: {len(chunks)} chunks guardados en {VECTORSTORE_DIR}")


if __name__ == "__main__":
    import sys

    ingestar(forzar="--forzar" in sys.argv)
