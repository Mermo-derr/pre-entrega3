"""Las dos pruebas del enunciado.

Uso: .venv\\Scripts\\python.exe main.py
"""
import asyncio
import sys

from ingest import ingestar
from rag import get_rag_response
from schemas import RAGResponse

# Está en CONTRATO DE ARRENDAMIENTO DE PLAZA DE GARAJE.pdf: 120 € al mes
PREGUNTA_EN_DOCS = "¿Cuál es la renta mensual de la plaza de garaje?"

# Plausible en un contrato de alquiler, pero ningún contrato menciona mascotas ni animales
PREGUNTA_TRAMPA = "¿Se permite tener mascotas en la vivienda alquilada?"


def mostrar(titulo: str, pregunta: str, r: RAGResponse) -> None:
    print(f"\n=== {titulo} ===")
    print(f"Pregunta: {pregunta}")
    print(f"Respuesta: {r.respuesta}")
    print(f"¿En el contexto?: {r.respuesta_en_contexto}")
    for ref in r.referencias:
        print(f"  - {ref.archivo}, pág. {ref.pagina}")


async def main() -> int:
    ingestar()  # no reindexa si la colección ya existe

    # Las dos consultas son independientes: se lanzan en paralelo
    en_docs, trampa = await asyncio.gather(
        get_rag_response(PREGUNTA_EN_DOCS),
        get_rag_response(PREGUNTA_TRAMPA),
    )

    mostrar("Prueba 1: respuesta en los documentos", PREGUNTA_EN_DOCS, en_docs)
    mostrar("Prueba 2: pregunta trampa", PREGUNTA_TRAMPA, trampa)

    # Verificaciones
    ok_1 = en_docs.respuesta_en_contexto and "120" in en_docs.respuesta and any(
        "GARAJE" in ref.archivo for ref in en_docs.referencias
    )
    ok_2 = (
        not trampa.respuesta_en_contexto
        and "no lo sé" in trampa.respuesta.lower()
        and not trampa.referencias
    )

    print("\n=== Resultado ===")
    print(f"Prueba 1 (responde con 120 € y cita el contrato de garaje): {'OK' if ok_1 else 'FALLA'}")
    print(f"Prueba 2 (dice 'No lo sé' sin referencias, no alucina):    {'OK' if ok_2 else 'FALLA'}")

    return 0 if ok_1 and ok_2 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
