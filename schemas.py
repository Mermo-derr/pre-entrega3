"""Modelo Pydantic de la respuesta del RAG"""
from pydantic import BaseModel, Field


class Referencia(BaseModel):
    """Origen de un dato usado en la respuesta."""

    archivo: str = Field(description="Nombre exacto del archivo, tal como aparece en 'Fuente' del contexto")
    pagina: int = Field(description="Número de página, tal como aparece en 'Página' del contexto")


class RAGResponse(BaseModel):
    """Respuesta final: el texto generado y las referencias usadas."""

    respuesta: str = Field(
        description="Respuesta a la pregunta basada solo en el contexto, o exactamente 'No lo sé.' si no está en él"
    )
    respuesta_en_contexto: bool = Field(
        description="true si el contexto contiene la respuesta; false si respondiste 'No lo sé.'"
    )
    referencias: list[Referencia] = Field(
        default_factory=list,
        description="Fragmentos del contexto de los que sale la respuesta; lista vacía si respondiste 'No lo sé.'",
    )
