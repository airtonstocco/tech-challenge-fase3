import yfinance as yf
import boto3
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
import os
from datetime import datetime, timezone, timedelta
import time
import pyarrow as pa
import pyarrow.parquet as pq

load_dotenv()

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION")
)

lambda_client = boto3.client(
    "lambda",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION")
)

BUCKET_NAME = "dados-fase-3"

# Lista fixa dos tickers B3
tickers_b3 = [
    "ABEV3.SA", "ALPA4.SA", "AMER3.SA", "ASAI3.SA", "AZUL4.SA",
    "B3SA3.SA", "BBAS3.SA", "BBDC3.SA", "BBDC4.SA", "BBSE3.SA",
    "BPAC11.SA", "BRAP4.SA", "BRFS3.SA", "BRKM5.SA", "BRML3.SA",
    "CCRO3.SA", "CIEL3.SA", "CMIG4.SA", "COGN3.SA", "CPFE3.SA",
    "CPLE6.SA", "CRFB3.SA", "CSAN3.SA", "CSNA3.SA", "CYRE3.SA",
    "ELET3.SA", "ELET6.SA", "EMBR3.SA", "ENGI11.SA", "EQTL3.SA",
    "EZTC3.SA", "GGBR4.SA", "GOAU4.SA", "HAPV3.SA", "HYPE3.SA",
    "IGTI11.SA", "ITSA4.SA", "ITUB4.SA", "JBSS3.SA", "KLBN11.SA",
    "LREN3.SA", "MGLU3.SA", "MRFG3.SA", "MRVE3.SA", "MULT3.SA",
    "NTCO3.SA", "PETR3.SA", "PETR4.SA", "PRIO3.SA", "RADL3.SA",
    "RAIL3.SA", "RENT3.SA", "RRRP3.SA", "SANB11.SA", "SBSP3.SA",
    "SMTO3.SA", "SUZB3.SA", "TAEE11.SA", "TIMS3.SA", "UGPA3.SA",
    "USIM5.SA", "VALE3.SA", "VBBR3.SA", "VIIA3.SA", "VIVT3.SA", "WEGE3.SA", "YDUQ3.SA"
]

def obter_ultima_data_ingestao_por_ticker(bucket_name, ticker):
    prefix = "raw/"
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    datas = []

    if "Contents" not in response:
        return None

    for obj in response["Contents"]:
        key = obj["Key"]
        if f"{ticker}_" in key and "ingestion_date=" in key:
            try:
                data = key.split("ingestion_date=")[1].split("/")[0]
                datas.append(data)
            except Exception:
                continue

    if not datas:
        return None

    return max(datas)  # retorna a última data em formato 'YYYY-MM-DD'

def coletar_e_enviar_para_s3():
    resultados = []
    hoje = datetime.now(timezone.utc).date()
    end_date = hoje - timedelta(days=1)
    end_date_str = end_date.strftime("%Y-%m-%d")

    for ticker in tickers_b3:
        try:
            start_date = obter_ultima_data_ingestao_por_ticker(BUCKET_NAME, ticker)

            if not start_date:
                print(f"{ticker} será coletado pela primeira vez (desde 2022-01-01).")
                start_date = "2022-01-01"

            if start_date >= end_date_str:
                resultados.append({ticker: "Sem novos dados"})
                continue

            print(f"Baixando {ticker} de {start_date} até {end_date_str}...")

            dados = yf.download(ticker, start=start_date, end=end_date_str, auto_adjust=True)
            dados = dados.reset_index()
            dados.rename(columns={"Date": "date"}, inplace=True)
            dados["date"] = dados["date"].dt.strftime("%Y-%m-%d")
            dados["ingestion_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            dados["ticker"] = ticker

            if dados.empty:
                resultados.append({ticker: "Sem dados"})
                continue

            dados.columns = [col[0] if isinstance(col, tuple) else col for col in dados.columns]
            table = pa.Table.from_pandas(dados, preserve_index=False)
            buffer = BytesIO()
            pq.write_table(table, buffer, coerce_timestamps="us", use_deprecated_int96_timestamps=False)

            ingestion_date = datetime.now().strftime("%Y-%m-%d")
            filename = f"raw/ingestion_date={ingestion_date}/{ticker}_{datetime.now().strftime('%H-%M-%S')}.parquet"

            s3.put_object(
                Bucket=BUCKET_NAME,
                Key=filename,
                Body=buffer.getvalue()
            )

            print(f"{ticker} enviado com ingestion_date={ingestion_date}")
            resultados.append({ticker: "Enviado"})

            time.sleep(1)

        except Exception as e:
            print(f"Erro em {ticker}: {e}")
            resultados.append({ticker: f"Erro: {str(e)}"})

    # Após o loop de coleta de dados, invoca a função Lambda
    invocar_lambda()

    return resultados

def invocar_lambda():
    try:
        response = lambda_client.invoke(
            FunctionName='fase-3-glue-job',  # Substitua pelo nome da sua função Lambda
            InvocationType='Event',  # 'Event' para execução assíncrona
            Payload='{"message": "Coleta de dados B3 finalizada."}'
        )
        print(f"Função Lambda invocada com sucesso: {response['StatusCode']}")
    except Exception as e:
        print(f"Erro ao invocar Lambda: {e}")

def coletar_dados_hoje():
    resultados = []
    hoje = datetime.now(timezone.utc).date()
    hoje_str = hoje.strftime("%Y-%m-%d")

    for ticker in tickers_b3:
        try:
            print(f"Coletando dados de hoje para {ticker} ({hoje_str})...")

            # Baixa dados do dia atual
            dados = yf.download(
                ticker,
                start=hoje_str,
                end=hoje_str,
                auto_adjust=True,
                interval="1d"
            )

            if dados.empty:
                print(f"Sem dados para {ticker} hoje.")
                continue

            dados.columns = [col[0] if isinstance(col, tuple) else col for col in dados.columns]

            dados = dados.reset_index()
            dados.rename(columns={"Date": "date"}, inplace=True)
            dados["date"] = dados["date"].dt.strftime("%Y-%m-%d")
            dados["ticker"] = ticker

            # Transforma em dicionários para retorno JSON-friendly
            registros = dados.to_dict(orient="records")
            resultados.extend(registros)

        except Exception as e:
            print(f"Erro ao coletar {ticker}: {e}")

    return resultados
