import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(
    title="Oráculo del Álgebra de Lie - API",
    description="Motor de cálculo de conmutadores para SU(3) con explicación teórica."
)

# Permitir solicitudes desde Neocities y entornos locales
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://oscar7.neocities.org",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar cliente de Gemini con la API Key configurada en variables de entorno
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Definición de las 8 matrices de Gell-Mann (Generadores de SU(3))
i = 1j
GELL_MANN = {
    1: np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex),
    2: np.array([[0, -i, 0], [i, 0, 0], [0, 0, 0]], dtype=complex),
    3: np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex),
    4: np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex),
    5: np.array([[0, 0, -i], [0, 0, 0], [i, 0, 0]], dtype=complex),
    6: np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex),
    7: np.array([[0, 0, 0], [0, -i, 0], [0, i, 0]], dtype=complex),
    8: (1 / np.sqrt(3)) * np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex)
}

class ConmutadorRequest(BaseModel):
    operador_a: int
    operador_b: int

def serializar_matriz_compleja(matriz: np.ndarray) -> list:
    return [
        [{"real": round(float(x.real), 4), "imag": round(float(x.imag), 4)} for x in fila]
        for fila in matriz
    ]

def generar_explicacion_llm(index_a: int, index_b: int, matriz_res: np.ndarray, es_nulo: bool) -> str:
    prompt = f"""
    Actúa como un físico teórico experto en Teoría de Campos Cuánticos y Álgebras de Lie.
    El usuario ha evaluado el conmutador [\\lambda_{index_a}, \\lambda_{index_b}] en el álgebra SU(3) de la Cromodinámica Cuántica (QCD).
    
    Datos numéricos calculados:
    - Operador A: \\lambda_{index_a}
    - Operador B: \\lambda_{index_b}
    - Resultado de [\\lambda_{index_a}, \\lambda_{index_b}]: {matriz_res.tolist()}
    - ¿Es conmutador nulo?: {es_nulo}

    Instrucciones:
    1. Explica brevemente la implicación física en la simetría de color y gluones.
    2. Expresa las ecuaciones matemáticas formateadas en LaTeX ($ para inline, $$ para bloques).
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="Responde únicamente con análisis físico y fórmulas en Markdown/LaTeX.",
            temperature=0.2
        )
    )
    return response.text

@app.post("/api/v1/conmutador/su3")
def calcular_conmutador(req: ConmutadorRequest):
    if req.operador_a not in GELL_MANN or req.operador_b not in GELL_MANN:
        raise HTTPException(status_code=400, detail="Los operadores deben estar entre el 1 y el 8.")

    A = GELL_MANN[req.operador_a]
    B = GELL_MANN[req.operador_b]

    # Cálculo algebraico exacto con NumPy
    conmutador = np.dot(A, B) - np.dot(B, A)
    es_nulo = np.allclose(conmutador, 0)

    # Explicación conceptual generada por IA
    explicacion_latex = generar_explicacion_llm(req.operador_a, req.operador_b, conmutador, es_nulo)

    return {
        "operadores": {"A": f"λ{req.operador_a}", "B": f"λ{req.operador_b}"},
        "conmuta": bool(es_nulo),
        "matriz_resultado": serializar_matriz_compleja(conmutador),
        "desglose_latex": explicacion_latex
    }
