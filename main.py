from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from collector import coletar_e_enviar_para_s3

app = FastAPI(
    title = "tech-challenge-3",
    version = "1.0.0",
    description = "Tech Challenge 3"
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.post("/coletar/")
def coletar():
    resultados = coletar_e_enviar_para_s3()
    return {"resultado": resultados}
