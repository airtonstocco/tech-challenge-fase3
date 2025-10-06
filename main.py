from fastapi import FastAPI
from collector import coletar_e_enviar_para_s3

app = FastAPI(title="API Coletora B3")

@app.get("/")
def home():
    return {"mensagem": "API Coletora de dados B3 está online."}

@app.post("/coletar/")
def coletar():
    resultados = coletar_e_enviar_para_s3()
    return {"resultado": resultados}