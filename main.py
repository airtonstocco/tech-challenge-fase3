from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from collector import coletar_e_enviar_para_s3, coletar_dados_hoje, get_recomendacao_modelo

app = FastAPI(
    title = "tech-challenge-3",
    version = "1.0.0",
    description = "Tech Challenge 3"
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.post("/post_atualiza_historico/")
def post_historico():
    resultados = coletar_e_enviar_para_s3()
    return {"resultado": resultados}

""" @app.get("/get_dados_hoje")
def get_hoje():
    dados_atuais = coletar_dados_hoje()
    return {"status": "ok", "quantidade_tickers": len(dados_atuais), "dados": dados_atuais} """

@app.get("/get_recomendacao_modelo")
def get_recomendacao():
    acoes_recomendadas = get_recomendacao_modelo()
    return {"status": "ok", "dados": acoes_recomendadas}