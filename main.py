from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Cargar el modelo y el encoder
modelo = joblib.load("modelo_entrenado.pkl")
encoder = joblib.load("encoder.pkl")

# Crear la app
app = FastAPI()

# Clase para recibir los datos de entrada
class DatosEntrada(BaseModel):
    genre: str
    year_of_release: float

# Ruta principal
@app.get("/")
def root():
    return {"mensaje": "API para predecir si un videojuego tendrá ventas altas"}

# Ruta para predecir
@app.post("/predict")
def predecir(datos: DatosEntrada):
    # Codificar el género usando el encoder
    genero_codificado = encoder.transform([[datos.genre]])
    entrada = np.hstack((genero_codificado, [[datos.year_of_release]]))
    prediccion = modelo.predict(entrada)
    return {"ventas_altas": int(prediccion[0])}

