# -*- coding: utf-8 -*-
"""
TCC - Modelo preditivo para tempo de travessia na Ponte Rio-Niteroi

Este script faz o fluxo completo:
1. Le os tweets brutos da Ecovias Ponte.
2. Extrai variaveis de data, sentido, condicao, acidentes, locais e tempo.
3. Cruza os tweets com dados horarios do INMET.
4. Treina modelos de regressao para prever `tempo_travessia_min`.
5. Compara Arvore de Decisao, Random Forest e XGBoost.
6. Usa One-Hot Encoding e binarizacao de `locais_mencionados`.
7. Gera graficos e resumo dos locais mais associados a incidentes.

Arquivos esperados na mesma pasta:
- ecoviasponte_tweets_completo.csv
- inmet_niteroi_2025_2026.csv

Dependencias:
pip install pandas numpy matplotlib scikit-learn xgboost
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


# =============================================================================
# 1. Configuracoes gerais
# =============================================================================

RANDOM_STATE = 42
TARGET = "tempo_travessia_min"

INPUT_TWEETS = Path("ecoviasponte_tweets_completo.csv")
INPUT_INMET = Path("inmet_niteroi_2025_2026.csv")

OUTPUT_DADOS_ORGANIZADOS = Path("ecoviasponte_modelagem_com_inmet.csv")
OUTPUT_RESUMO_LOCAIS = Path("locais_incidentes_resumo.csv")
OUTPUT_GRAFICO_PREVISOES = Path("grafico_real_vs_previsto_modelos.png")
OUTPUT_GRAFICO_IMPORTANCIA_RIO = Path("grafico_importancia_xgboost_rio.png")
OUTPUT_GRAFICO_IMPORTANCIA_NITEROI = Path("grafico_importancia_xgboost_niteroi.png")

DIAS_PT = {
    0: "Segunda",
    1: "Terca",
    2: "Quarta",
    3: "Quinta",
    4: "Sexta",
    5: "Sabado",
    6: "Domingo",
}

LOCAIS = [
    "Reta do Cais",
    "Grande Reta",
    "Vao Central",
    "Vão Central",
    "Jansen de Melo",
    "Avenida Brasil",
    "Av. Brasil",
    "Mocangue",
    "Mocanguê",
    "Gasometro",
    "Gasômetro",
    "Pedagio",
    "Pedágio",
    "Portico",
    "Pórtico",
    "Alameda",
    "Rodoviaria",
    "Rodoviária",
    "INTO",
    "Caju",
    "Charitas",
    "Icarai",
    "Icaraí",
]

LOCAIS_NORMALIZADOS = {
    "Vão Central": "Vao Central",
    "Mocanguê": "Mocangue",
    "Gasômetro": "Gasometro",
    "Pedágio": "Pedagio",
    "Pórtico": "Portico",
    "Rodoviária": "Rodoviaria",
    "Icaraí": "Icarai",
    "Av. Brasil": "Avenida Brasil",
}

LOCAIS_RE = [(loc, re.compile(re.escape(loc), re.I)) for loc in LOCAIS]


# =============================================================================
# 2. Organizacao dos tweets
# =============================================================================

def parse_datetime(raw: str):
    """Converte data bruta do X/Twitter para timestamp UTC."""
    try:
        return pd.to_datetime(raw, format="%a %b %d %H:%M:%S +0000 %Y", utc=True)
    except Exception:
        return pd.NaT


def periodo_dia(hora: int) -> str:
    if 0 <= hora < 6:
        return "madrugada"
    if 6 <= hora < 12:
        return "manha"
    if 12 <= hora < 18:
        return "tarde"
    return "noite"


def classificar_tipo(texto: str) -> str:
    t = str(texto).lower()

    tem_ocorrencia = "ocorrencia" in t or "ocorrência" in t
    ocorrencia_encerrada = tem_ocorrencia and any(
        p in t for p in ["finalizada", "finalizado", "liberada", "liberado", "liberacao", "liberação"]
    )

    if any(
        p in t
        for p in [
            "acidente",
            "colisao",
            "colisão",
            "capotamento",
            "atropelamento",
            "engavetamento",
            "pane mecanica",
            "pane mecânica",
            "pane seca",
        ]
    ) or (tem_ocorrencia and not ocorrencia_encerrada):
        return "acidente"

    if any(p in t for p in ["interdicao", "interdição", "interditad", "bloqueio", "fechada", "fechado"]):
        return "interdicao"

    if any(
        p in t
        for p in [
            "liberada",
            "liberado",
            "liberacao",
            "liberação",
            "normalizado",
            "retomado",
            "ocorrencia finalizada",
            "ocorrência finalizada",
            "faixas liberadas",
        ]
    ):
        return "liberacao"

    if any(
        p in t
        for p in [
            "atualizacao do transito",
            "atualização do trânsito",
            "atualizacao de transito",
            "atualização de trânsito",
        ]
    ):
        return "atualizacao_trafego"

    if any(p in t for p in ["sentido rio", "sentido niteroi", "sentido niterói"]):
        return "atualizacao_trafego"

    if any(p in t for p in ["dica", "evite", "atencao", "atenção", "lembre", "respeite", "velocidade"]):
        return "informativo"

    return "outro"


def extrair_sentido(texto: str):
    t = str(texto).lower()
    tem_rio = "sentido rio" in t
    tem_niteroi = "sentido niteroi" in t or "sentido niterói" in t

    if tem_rio and tem_niteroi:
        return "ambos"
    if tem_rio:
        return "Rio"
    if tem_niteroi:
        return "Niteroi"
    return None


def extrair_tempo_travessia(texto: str):
    padrao = r"(?:travessia\s+(?:de|em|media de|média de)\s+)?(\d+)\s+minutos?"
    matches = re.findall(padrao, str(texto).lower())
    if matches:
        return max(int(m) for m in matches)
    return None


def classificar_condicao(texto: str):
    t = str(texto).lower()
    if any(p in t for p in ["congestionamento", "muito lento", "trafego intenso", "tráfego intenso", "🔴"]):
        return "congestionamento"
    if any(p in t for p in ["lentidao", "lentidão", "fluxo lento", "🟡"]):
        return "lentidao"
    if any(p in t for p in ["fluxo normal", "fluxo bom", "🟢"]):
        return "fluxo normal"
    return None


def detectar_acidente(texto: str) -> int:
    t = str(texto).lower()
    return int(
        any(
            p in t
            for p in [
                "acidente",
                "colisao",
                "colisão",
                "capotamento",
                "atropelamento",
                "engavetamento",
                "queda de moto",
                "pane mecanica",
                "pane mecânica",
                "pane seca",
                "pane",
            ]
        )
    )


def normalizar_local(local: str) -> str:
    return LOCAIS_NORMALIZADOS.get(local, local)


def extrair_locais(texto: str) -> str:
    encontrados = []
    for loc, pat in LOCAIS_RE:
        if pat.search(str(texto)):
            encontrados.append(normalizar_local(loc))

    encontrados = sorted(set(encontrados))
    return "; ".join(encontrados) if encontrados else ""


def carregar_tweets(caminho_csv: Path) -> pd.DataFrame:
    print(f"Lendo tweets: {caminho_csv}")
    df = pd.read_csv(caminho_csv, low_memory=False, usecols=["createdAt", "text"])

    df["_ts"] = df["createdAt"].apply(parse_datetime)
    df["_ts_brt"] = df["_ts"].dt.tz_convert("America/Sao_Paulo")

    df["data"] = df["_ts_brt"].dt.date
    df["hora_postagem"] = df["_ts_brt"].dt.strftime("%H:%M")
    df["dia_semana"] = df["_ts_brt"].dt.dayofweek.map(DIAS_PT)
    df["fim_de_semana"] = df["_ts_brt"].dt.dayofweek.isin([5, 6]).astype(int)
    df["periodo_dia"] = df["_ts_brt"].dt.hour.apply(periodo_dia)
    df["mes"] = df["_ts_brt"].dt.month
    df["ano"] = df["_ts_brt"].dt.year
    df["texto"] = df["text"].str.strip()

    df["tipo_tweet"] = df["texto"].apply(classificar_tipo)
    df["sentido"] = df["texto"].apply(extrair_sentido)
    df["tempo_travessia_min"] = df["texto"].apply(extrair_tempo_travessia)
    df["condicao_label"] = df["texto"].apply(classificar_condicao)
    df["tem_acidente"] = df["texto"].apply(detectar_acidente)
    df["locais_mencionados"] = df["texto"].apply(extrair_locais)

    colunas = [
        "data",
        "hora_postagem",
        "dia_semana",
        "fim_de_semana",
        "periodo_dia",
        "mes",
        "ano",
        "tipo_tweet",
        "sentido",
        "tempo_travessia_min",
        "condicao_label",
        "tem_acidente",
        "locais_mencionados",
        "texto",
    ]

    df_final = df[colunas].sort_values(["data", "hora_postagem"]).reset_index(drop=True)
    print(f"Tweets organizados: {len(df_final):,} linhas")
    return df_final


# =============================================================================
# 3. Cruzamento com dados meteorologicos do INMET
# =============================================================================

def cruzar_inmet(df: pd.DataFrame, caminho_inmet: Path) -> pd.DataFrame:
    print(f"Lendo INMET: {caminho_inmet}")
    inmet = pd.read_csv(caminho_inmet)

    inmet["datetime_brt"] = pd.to_datetime(inmet["datetime_brt"], errors="coerce")
    if inmet["datetime_brt"].dt.tz is None:
        inmet["datetime_brt"] = inmet["datetime_brt"].dt.tz_localize("America/Sao_Paulo")
    else:
        inmet["datetime_brt"] = inmet["datetime_brt"].dt.tz_convert("America/Sao_Paulo")

    inmet["_chave"] = inmet["datetime_brt"].dt.floor("h")
    inmet = inmet.drop_duplicates(subset="_chave")

    aux = pd.to_datetime(
        df["data"].astype(str) + " " + df["hora_postagem"],
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    ).dt.tz_localize("America/Sao_Paulo").dt.floor("h")

    clima_cols = ["_chave", "precipitacao_mm", "temperatura_C", "vento_rajada_ms"]
    clima = aux.rename("_chave").to_frame().merge(inmet[clima_cols], on="_chave", how="left")
    clima = clima.drop(columns="_chave")

    df_final = pd.concat([df.reset_index(drop=True), clima.reset_index(drop=True)], axis=1)
    print(f"Dados com INMET: {len(df_final):,} linhas")
    return df_final


# =============================================================================
# 4. Preparacao para modelagem
# =============================================================================

NUMERIC_FEATURES = [
    "hora",
    "fim_de_semana",
    "mes",
    "ano",
    "tem_acidente",
    "temperatura_C",
    "precipitacao_mm",
    "vento_rajada_ms",
]

CATEGORICAL_FEATURES = [
    "dia_semana",
    "periodo_dia",
]

LOCAL_FEATURE = "locais_mencionados"


class LocaisBinarizer(BaseEstimator, TransformerMixin):
    """Transforma 'Local A; Local B' em colunas binarias, uma por local."""

    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    @staticmethod
    def split_locais(valor: object) -> list[str]:
        if pd.isna(valor) or str(valor).strip() == "":
            return []
        return [item.strip() for item in str(valor).split(";") if item.strip()]

    def fit(self, X, y=None):
        serie = pd.Series(np.asarray(X).ravel())
        self.mlb.fit(serie.apply(self.split_locais))
        return self

    def transform(self, X):
        serie = pd.Series(np.asarray(X).ravel())
        return self.mlb.transform(serie.apply(self.split_locais))

    def get_feature_names_out(self, input_features=None):
        return np.array([f"local_{classe}" for classe in self.mlb.classes_])


def preparar_dados_modelagem(df_final: pd.DataFrame, sentido: str) -> pd.DataFrame:
    df = df_final[
        (df_final["sentido"] == sentido) &
        df_final[TARGET].notna()
    ].copy()

    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["hora"] = df["hora_postagem"].str[:2].astype(int)
    df["mes"] = df["mes"].astype(int)
    df["ano"] = df["ano"].astype(int)

    for col in ["temperatura_C", "precipitacao_mm", "vento_rajada_ms"]:
        if col not in df.columns:
            df[col] = np.nan

    if LOCAL_FEATURE not in df.columns:
        df[LOCAL_FEATURE] = ""

    df = df.dropna(subset=["data"]).sort_values(["data", "hora"]).reset_index(drop=True)
    return df


def split_temporal(df: pd.DataFrame, test_size: float = 0.2):
    corte = int(len(df) * (1 - test_size))
    treino = df.iloc[:corte].copy()
    teste = df.iloc[corte:].copy()
    return treino, teste


def montar_preprocessador() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    locais_pipeline = Pipeline(
        steps=[
            ("binarizador", LocaisBinarizer()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("locais", locais_pipeline, [LOCAL_FEATURE]),
        ],
        remainder="drop",
    )


def montar_pipeline(modelo) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessamento", montar_preprocessador()),
            ("modelo", modelo),
        ]
    )


# =============================================================================
# 5. Avaliacao e treinamento
# =============================================================================

def calcular_metricas(y_real, y_pred) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_real, y_pred),
        "RMSE": mean_squared_error(y_real, y_pred) ** 0.5,
        "R2": r2_score(y_real, y_pred),
    }


def imprimir_metricas(nome: str, metricas: dict[str, float]) -> None:
    print(f"\n{nome}")
    print(f"  MAE : {metricas['MAE']:.3f} min")
    print(f"  RMSE: {metricas['RMSE']:.3f} min")
    print(f"  R2  : {metricas['R2']:.4f}")


def avaliar_modelo(nome: str, pipeline: Pipeline, X_treino, X_teste, y_treino, y_teste):
    pipeline.fit(X_treino, y_treino)
    pred_treino = pipeline.predict(X_treino)
    pred_teste = pipeline.predict(X_teste)

    metricas_treino = calcular_metricas(y_treino, pred_treino)
    metricas_teste = calcular_metricas(y_teste, pred_teste)

    imprimir_metricas(f"{nome} - treino", metricas_treino)
    imprimir_metricas(f"{nome} - teste", metricas_teste)

    return pipeline, pred_teste, metricas_teste


def baseline_mediana(y_treino, y_teste):
    pred = np.repeat(y_treino.median(), len(y_teste))
    return calcular_metricas(y_teste, pred)


def rodar_modelos_por_sentido(df_final: pd.DataFrame, sentido: str):
    df = preparar_dados_modelagem(df_final, sentido)
    treino, teste = split_temporal(df)

    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [LOCAL_FEATURE]
    X_treino = treino[features]
    y_treino = treino[TARGET]
    X_teste = teste[features]
    y_teste = teste[TARGET]

    print("=" * 70)
    print(f"SENTIDO {sentido.upper()}")
    print("=" * 70)
    print(f"Treino cronologico: {len(X_treino):,} | Teste cronologico: {len(X_teste):,}")

    imprimir_metricas("Baseline mediana - teste", baseline_mediana(y_treino, y_teste))

    modelos = {
        "Arvore de Decisao": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
    }

    resultados = {}
    predicoes = {}

    for nome, modelo in modelos.items():
        pipeline, pred, metricas = avaliar_modelo(
            nome,
            montar_pipeline(modelo),
            X_treino,
            X_teste,
            y_treino,
            y_teste,
        )
        resultados[nome] = {"pipeline": pipeline, "metricas": metricas}
        predicoes[nome] = pred

    params = {
        "modelo__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "modelo__max_depth": [3, 4, 6, 8],
        "modelo__n_estimators": [100, 200, 300],
        "modelo__colsample_bytree": [0.6, 0.8, 1.0],
        "modelo__subsample": [0.6, 0.8, 1.0],
        "modelo__reg_lambda": [1, 5, 10],
        "modelo__reg_alpha": [0, 0.1, 1],
        "modelo__min_child_weight": [1, 3, 5],
    }

    busca = RandomizedSearchCV(
        estimator=montar_pipeline(
            xgb.XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                verbosity=0,
            )
        ),
        param_distributions=params,
        n_iter=20,
        cv=TimeSeriesSplit(n_splits=4),
        scoring="neg_root_mean_squared_error",
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    busca.fit(X_treino, y_treino)
    print(f"\nMelhores parametros XGBoost - {sentido}:")
    print(busca.best_params_)
    print(f"RMSE medio CV temporal: {-busca.best_score_:.3f} min")

    xgb_ajustado, pred_ajustado, metricas_ajustado = avaliar_modelo(
        "XGBoost ajustado",
        busca.best_estimator_,
        X_treino,
        X_teste,
        y_treino,
        y_teste,
    )

    resultados["XGBoost ajustado"] = {"pipeline": xgb_ajustado, "metricas": metricas_ajustado}
    predicoes["XGBoost ajustado"] = pred_ajustado

    return {
        "sentido": sentido,
        "y_teste": y_teste,
        "predicoes": predicoes,
        "resultados": resultados,
    }


# =============================================================================
# 6. Analise de locais e graficos
# =============================================================================

def analisar_locais_incidentes(df_final: pd.DataFrame) -> pd.DataFrame:
    incidentes = df_final[
        (df_final["tem_acidente"] == 1) |
        (df_final["tipo_tweet"].isin(["acidente", "interdicao", "liberacao"]))
    ].copy()

    linhas = []
    for _, row in incidentes.iterrows():
        locais = LocaisBinarizer.split_locais(row.get(LOCAL_FEATURE, ""))
        for local in locais:
            linhas.append(
                {
                    "local": local,
                    "sentido": row.get("sentido"),
                    "tipo_tweet": row.get("tipo_tweet"),
                    "tempo_travessia_min": row.get(TARGET),
                }
            )

    if not linhas:
        return pd.DataFrame(columns=["local", "qtd_tweets", "tempo_medio_travessia"])

    locais_df = pd.DataFrame(linhas)
    resumo = (
        locais_df.groupby("local", dropna=False)
        .agg(
            qtd_tweets=("local", "size"),
            tempo_medio_travessia=("tempo_travessia_min", "mean"),
        )
        .sort_values("qtd_tweets", ascending=False)
        .reset_index()
    )
    return resumo


def plotar_reais_vs_previstos(resultado_rio: dict, resultado_niteroi: dict) -> None:
    modelos = list(resultado_rio["predicoes"].keys())
    fig, axes = plt.subplots(2, len(modelos), figsize=(4.5 * len(modelos), 8))

    for col, nome_modelo in enumerate(modelos):
        for row, resultado in enumerate([resultado_rio, resultado_niteroi]):
            ax = axes[row][col]
            y_real = resultado["y_teste"]
            y_pred = resultado["predicoes"][nome_modelo]
            sentido = resultado["sentido"]

            ax.scatter(y_real, y_pred, alpha=0.35, s=10)
            lim = [min(y_real.min(), y_pred.min()) - 1, max(y_real.max(), y_pred.max()) + 1]
            ax.plot(lim, lim, "r--", linewidth=1.2, label="Previsao perfeita")
            ax.set_xlabel("Valor real (min)")
            ax.set_ylabel("Valor previsto (min)")
            ax.set_title(f"{sentido} - {nome_modelo}", fontsize=10)
            ax.grid(linestyle="--", alpha=0.35)
            ax.legend(fontsize=8)

    plt.suptitle("Valores reais vs previstos - tempo de travessia", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_GRAFICO_PREVISOES, dpi=150, bbox_inches="tight")
    plt.show()


def plotar_importancia_xgboost(resultado: dict, caminho_saida: Path) -> None:
    pipeline = resultado["resultados"]["XGBoost ajustado"]["pipeline"]
    modelo = pipeline.named_steps["modelo"]
    preprocessador = pipeline.named_steps["preprocessamento"]

    nomes_features = preprocessador.get_feature_names_out()
    importancias = (
        pd.Series(modelo.feature_importances_, index=nomes_features)
        .sort_values(ascending=False)
        .head(20)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    importancias.sort_values().plot(kind="barh", ax=ax)
    ax.set_title(f"Top 20 variaveis - XGBoost ajustado ({resultado['sentido']})")
    ax.set_xlabel("Importancia")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# 7. Execucao principal
# =============================================================================

def main() -> None:
    df_tweets = carregar_tweets(INPUT_TWEETS)
    df_final = cruzar_inmet(df_tweets, INPUT_INMET)
    df_final.to_csv(OUTPUT_DADOS_ORGANIZADOS, index=False, encoding="utf-8-sig")

    print("\nResumo geral da base:")
    print(df_final.info())
    print("\nDistribuicao de condicao:")
    print(df_final["condicao_label"].value_counts(dropna=False))
    print("\nDistribuicao de sentido:")
    print(df_final["sentido"].value_counts(dropna=False))

    resumo_locais = analisar_locais_incidentes(df_final)
    resumo_locais.to_csv(OUTPUT_RESUMO_LOCAIS, index=False, encoding="utf-8-sig")
    print("\nLocais mais mencionados em tweets de incidentes/ocorrencias:")
    print(resumo_locais.head(15).to_string(index=False))

    resultado_rio = rodar_modelos_por_sentido(df_final, "Rio")
    resultado_niteroi = rodar_modelos_por_sentido(df_final, "Niteroi")

    plotar_reais_vs_previstos(resultado_rio, resultado_niteroi)
    plotar_importancia_xgboost(resultado_rio, OUTPUT_GRAFICO_IMPORTANCIA_RIO)
    plotar_importancia_xgboost(resultado_niteroi, OUTPUT_GRAFICO_IMPORTANCIA_NITEROI)

    print("\nArquivos gerados:")
    print(f"- {OUTPUT_DADOS_ORGANIZADOS}")
    print(f"- {OUTPUT_RESUMO_LOCAIS}")
    print(f"- {OUTPUT_GRAFICO_PREVISOES}")
    print(f"- {OUTPUT_GRAFICO_IMPORTANCIA_RIO}")
    print(f"- {OUTPUT_GRAFICO_IMPORTANCIA_NITEROI}")


if __name__ == "__main__":
    main()
