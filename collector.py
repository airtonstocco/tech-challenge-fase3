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
import pytz
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

load_dotenv()

TZ_BR = pytz.timezone("America/Sao_Paulo")

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
    "USIM5.SA", "VALE3.SA", "VBBR3.SA", "VIIA3.SA", "VIVT3.SA",
    "WEGE3.SA", "YDUQ3.SA", "^BVSP"
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
            dados.rename(columns={"Date": "trading_date"}, inplace=True)
            dados["trading_date"] = dados["trading_date"].dt.strftime("%Y-%m-%d")
            dados["ingestion_date"] = datetime.now(TZ_BR).strftime("%Y-%m-%d %H:%M:%S")
            dados["ticker"] = ticker

            if dados.empty:
                resultados.append({ticker: "Sem dados"})
                continue

            dados.columns = [col[0] if isinstance(col, tuple) else col for col in dados.columns]
            table = pa.Table.from_pandas(dados, preserve_index=False)
            buffer = BytesIO()
            pq.write_table(table, buffer, coerce_timestamps="us", use_deprecated_int96_timestamps=False)

            ingestion_date = datetime.now(TZ_BR).strftime("%Y-%m-%d")
            filename = f"raw/ingestion_date={ingestion_date}/{ticker}_{datetime.now(TZ_BR).strftime('%H-%M-%S')}.parquet"

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

    return resultados

def coletar_dados_hoje():
    resultados = []
    hoje_br = datetime.now(TZ_BR).date()
    hoje_str = hoje_br.strftime("%Y-%m-%d")

    for ticker in tickers_b3:
        try:
            print(f"Coletando dados de hoje ({hoje_str}) para {ticker}...")

            # Usa period em vez de start/end — mais confiável para intraday
            dados = yf.download(
                ticker,
                period="1d",
                interval="30m",
                auto_adjust=True,
                progress=False
            )

            # Se não vier nada, tenta últimos 5 dias (fallback)
            if dados.empty:
                print(f"Nenhum dado retornado para {ticker}, tentando últimos 5 dias...")
                dados = yf.download(
                    ticker,
                    period="5d",
                    interval="30m",
                    auto_adjust=True,
                    progress=False
                )

            if dados.empty:
                print(f"Sem dados para {ticker} nem nos últimos 5 dias.")
                continue

            # Ajusta timezone (Yahoo costuma retornar UTC)
            idx = dados.index
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            dados.index = idx.tz_convert(TZ_BR)

            # Filtra apenas o dia de hoje no fuso horário da B3
            dados = dados[dados.index.date == hoje_br]

            if dados.empty:
                print(f"{ticker}: sem candles para {hoje_str} (fora do horário de pregão?).")
                continue

            # Corrige colunas caso venham como MultiIndex
            dados.columns = [col[0] if isinstance(col, tuple) else col for col in dados.columns]

            # Prepara DataFrame para saída
            dados = dados.reset_index().rename(columns={"Datetime": "datetime"})
            dados["datetime"] = dados["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
            dados["ticker"] = ticker
            dados["trading_date"] = hoje_str

            # Converte para dicionários serializáveis
            registros = dados.to_dict(orient="records")
            resultados.extend(registros)

            print(f"{ticker}: {len(dados)} registros coletados.")

        except Exception as e:
            print(f"Erro ao coletar {ticker}: {e}")

    return resultados

def get_recomendation():    
    bucket_name = "dados-fase-3"
    prefix = "raw/"
    dfs = []

    # Lista todos os objetos da pasta raw/
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if "Contents" not in response:
        print("Nenhum arquivo encontrado.")
        return pd.DataFrame()

    for obj in response["Contents"]:
        key = obj["Key"]
        if key.endswith(".parquet"):
            print(f"Lendo {key}...")
            data = s3.get_object(Bucket=bucket_name, Key=key)['Body'].read()
            df = pq.read_table(BytesIO(data)).to_pandas()
            dfs.append(df)

    raw_data = pd.concat(dfs, ignore_index=True)
    print("Dados brutos carregados:", raw_data.shape)

    # Retorno do índice de referência
    reference_index_data = raw_data[raw_data["ticker"] == "^BVSP"].copy()
    reference_index_return = reference_index_data["Close"].pct_change().dropna().reset_index(drop=True)

    # Métricas por ticker
    keys = ["ticker","cumulative_return_period","return_std_deviation","volume_avg","drawdown","sharpe_ratio","correlation_index"] 
    metrics = {key: [] for key in keys}

    tickers_only = [t for t in raw_data["ticker"].unique() if t != "^BVSP"]

    for ticker in tickers_only:
        df_ticker = raw_data[raw_data["ticker"] == ticker].copy().reset_index(drop=True)

        close_values = df_ticker["Close"]
        volume = df_ticker["Volume"]

        cumulative_return_period = (close_values.iloc[-1] / close_values.iloc[0]) - 1
        daily_return = close_values.pct_change().dropna()
        return_std_deviation = daily_return.std()
        volume_avg = volume.mean()
        drawdown = (close_values / close_values.cummax() - 1).min()
        sharpe_ratio = cumulative_return_period / return_std_deviation if return_std_deviation > 0 else 0
        correlation_index = daily_return.corr(reference_index_return)

        metrics["ticker"].append(ticker)
        metrics["cumulative_return_period"].append(cumulative_return_period)
        metrics["return_std_deviation"].append(return_std_deviation)
        metrics["volume_avg"].append(volume_avg)
        metrics["drawdown"].append(drawdown)
        metrics["sharpe_ratio"].append(sharpe_ratio)
        metrics["correlation_index"].append(correlation_index)

    # Converte para DataFrame
    metrics_df = pd.DataFrame(metrics)
    print("Métricas calculadas:", metrics_df.shape)

    # Features e target
    features = ["cumulative_return_period", "return_std_deviation", "drawdown", "volume_avg", "correlation_index"]
    target = "sharpe_ratio"

    X = metrics_df[features]
    y = metrics_df[target]

    # Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.35, random_state=42)

    # Grid de parâmetros
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 1.0]
    }

    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=XGBRegressor(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    optimized_parameters = grid_search.best_params_
    print("Parâmetros otimizados:", optimized_parameters)

    # Modelo final
    modelo = XGBRegressor(
        **optimized_parameters,
        colsample_bytree=0.9,
        random_state=42
    )
    modelo.fit(X_train, y_train)

    # Avaliação
    y_pred = modelo.predict(X_test)
    print(f"R²: {r2_score(y_test, y_pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

    # Score predito no DataFrame completo
    metrics_df["score_predito"] = modelo.predict(X_scaled)

    # Top 5 recomendadas
    df_sorted = metrics_df.sort_values(by=["score_predito", "ticker"], ascending=[False, True]).reset_index(drop=True)
    carteira_recomendada = df_sorted.head(5)

    print("\nAções recomendadas:")
    print(carteira_recomendada[["ticker", "cumulative_return_period", "return_std_deviation", "drawdown", "score_predito"]])

    # Salva no S3 com versionamento via datetime
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_key = f"recommendations/recommendation_{now_str}.parquet"
    buffer = BytesIO()
    table = pa.Table.from_pandas(carteira_recomendada)
    pq.write_table(table, buffer)
    buffer.seek(0)
    s3.put_object(Bucket=bucket_name, Key=file_key, Body=buffer.getvalue())
    print(f"Carteira recomendada salva no S3: {file_key}")

    return carteira_recomendada.to_dict(orient="records")