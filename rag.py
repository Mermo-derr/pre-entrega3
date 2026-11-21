"""Recuperación + generación grounded con una cadena LCEL."""
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from config import LLM_MODEL, TOP_K
from ingest import abrir_vectorstore
from schemas import RAGResponse

# Prompt de sistema: "filtro de veracidad". {format_instructions} lo rellena
# el PydanticOutputParser con el esquema JSON de RAGResponse.
SYSTEM_PROMPT = """Eres un asistente jurídico especializado en contratos de arrendamiento.

Reglas:
1. Responde ÚNICAMENTE con la información del CONTEXTO. No uses conocimientos propios ni supongas nada.
2. Si el CONTEXTO no contiene la respuesta, el campo "respuesta" debe ser exactamente "No lo sé.", con "respuesta_en_contexto" en false y "referencias" vacía.
3. Si la respuesta está en el CONTEXTO, indica en "referencias" el archivo y la página de cada fragmento que hayas usado.
4. Hay varios contratos parecidos: deja claro a qué contrato se refiere cada dato.
5. Devuelve SIEMPRE un objeto JSON con el formato indicado abajo, también cuando no sepas la respuesta. Nunca texto suelto.

{format_instructions}"""

HUMAN_PROMPT = """CONTEXTO:
{contexto}

PREGUNTA: {pregunta}"""


def formatear_docs(docs: list[Document]) -> str:
    """Une los chunks recuperados en un solo texto, cada uno con su fuente y página."""
    if not docs:
        return "(no se recuperó ningún fragmento)"

    fragmentos = []
    for i, doc in enumerate(docs, 1):
        cabecera = f"[Fragmento {i}] Fuente: {doc.metadata.get('source')} | Página: {doc.metadata.get('page')}"
        fragmentos.append(f"{cabecera}\n{doc.page_content}")

    return "\n\n".join(fragmentos)


def get_retriever():
    """Búsqueda por similitud sobre la colección de la ingesta (mismos embeddings)."""
    return abrir_vectorstore().as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})


@lru_cache(maxsize=1)
def build_chain() -> Runnable:
    """Cadena LCEL: retriever -> formateo de docs -> prompt -> LLM -> PydanticOutputParser.

    Se construye una sola vez (lru_cache) y se reutiliza en cada consulta.
    """
    parser = PydanticOutputParser(pydantic_object=RAGResponse)

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
    ).partial(format_instructions=parser.get_format_instructions())

    # Modo JSON de OpenAI: garantiza JSON válido, que luego valida el PydanticOutputParser.
    # Sin él, el modelo a veces contesta "No lo sé." en texto plano y el parser falla.
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0).bind(response_format={"type": "json_object"})

    return (
        {
            "contexto": get_retriever() | RunnableLambda(formatear_docs),  # a. similitud
            "pregunta": RunnablePassthrough(),
        }
        | prompt  # b. prompt con los fragmentos
        | llm  # c. LLM (asíncrono vía ainvoke)
        | parser  # d. JSON -> RAGResponse
    )


async def get_rag_response(query: str) -> RAGResponse:
    """a. búsqueda de similitud, b. prompt con fragmentos,
    c. llamada asíncrona al LLM, d. parseo a RAGResponse."""
    return await build_chain().ainvoke(query)
