# -*- coding: utf-8 -*-


import pandas as pd
import re

# ── CONFIGURAÇÃO ───────────────────────────────────────────────────────────────
INPUT_CSV  = "ecoviasponte_tweets_completo.csv"   # altere se necessário
OUTPUT_CSV = "ecoviasponte_organizado.csv"        # opcional — só salva se rodar direto


# ── DICIONÁRIOS ────────────────────────────────────────────────────────────────
DIAS_PT = {
    0: "Segunda", 1: "Terça",  2: "Quarta",
    3: "Quinta",  4: "Sexta",  5: "Sábado", 6: "Domingo"
}

# Locais conhecidos na Ponte Rio-Niterói (do mais específico ao mais genérico)
LOCAIS = [
    "Reta do Cais", "Grande Reta", "Vão Central",
    "Jansen de Melo", "Avenida Brasil", "Av. Brasil",
    "Mocanguê", "Gasômetro", "Pedágio",
    "Pórtico", "Alameda", "Rodoviária", "INTO",
    "Caju", "Charitas", "Icaraí",
]
LOCAIS_RE = [(loc, re.compile(re.escape(loc), re.I)) for loc in LOCAIS]


# ── FUNÇÕES AUXILIARES ─────────────────────────────────────────────────────────

def parse_datetime(raw: str):
    """Converte 'Wed Sep 24 21:56:02 +0000 2025' → Timestamp UTC."""
    try:
        return pd.to_datetime(raw, format="%a %b %d %H:%M:%S +0000 %Y", utc=True)
    except Exception:
        return pd.NaT


def periodo_dia(hora: int) -> str:
    if 0  <= hora < 6:  return "madrugada"
    if 6  <= hora < 12: return "manhã"
    if 12 <= hora < 18: return "tarde"
    return "noite"


def classificar_tipo(texto: str) -> str:
    t = str(texto).lower()
    # "ocorrência" sozinha = acidente; acompanhada de encerramento = outro
    tem_ocorrencia = "ocorrência" in t
    ocorrencia_encerrada = tem_ocorrencia and any(p in t for p in [
        "finalizada", "finalizado", "liberada", "liberado", "liberação"
    ])
    if any(p in t for p in ["acidente", "colisão", "colisao", "capotamento",
                             "atropelamento", "engavetamento",
                             "pane mecânica", "pane seca"]) or        (tem_ocorrencia and not ocorrencia_encerrada):
        return "acidente"
    if any(p in t for p in ["interdição", "interdicao", "interditad",
                             "bloqueio", "fechada", "fechado"]):
        return "interdição"
    if any(p in t for p in ["liberada", "liberado", "liberação", "normalizado",
                             "retomado", "ocorrência finalizada", "faixas liberadas"]):
        return "liberação"
    if any(p in t for p in ["atualização do trânsito", "atualizacao do transito",
                             "atualização de trânsito"]):
        return "atualização_tráfego"
    if any(p in t for p in ["sentido rio", "sentido niterói", "sentido niteroi"]):
        return "atualização_tráfego"
    if any(p in t for p in ["dica", "evite", "atenção", "lembre", "respeite",
                             "velocidade", "fiscalização", "dirigir"]):
        return "informativo"
    return "outro"


def extrair_sentido(texto: str):
    t = str(texto).lower()
    tem_rio     = "sentido rio" in t
    tem_niteroi = "sentido niterói" in t or "sentido niteroi" in t
    if tem_rio and tem_niteroi:
        return "ambos"
    if tem_rio:
        return "Rio"
    if tem_niteroi:
        return "Niterói"
    return None


def extrair_tempo_travessia(texto: str):
    """
    Captura padrões como:
      '13 minutos' / 'travessia de 13 minutos' / 'travessia em 17 minutos'
    Quando há dois sentidos no mesmo tweet, retorna o maior valor.
    """
    padrao = r"(?:travessia\s+(?:de|em)\s+)?(\d+)\s+minutos?"
    matches = re.findall(padrao, str(texto).lower())
    if matches:
        return max(int(m) for m in matches)
    return None


def classificar_condicao(texto: str):
    """
    Retorna o label da pior condição mencionada no tweet.
    Usa emojis 🟢🟡🔴 além de palavras-chave.
    """
    t = str(texto).lower()
    # Ordem do mais grave ao mais leve — retorna o primeiro encontrado
    if any(p in t for p in ["congestionamento", "🔴", "muito lento",
                             "tráfego intenso", "trafego intenso"]):
        return "congestionamento"
    if any(p in t for p in ["lentidão", "lentidao", "fluxo lento", "🟡"]):
        return "lentidão"
    if any(p in t for p in ["fluxo normal", "fluxo bom", "🟢"]):
        return "fluxo normal"
    return None


def detectar_acidente(texto: str) -> int:
    t = str(texto).lower()
    return int(any(p in t for p in [
        "acidente", "colisão", "colisao", "capotamento",
        "atropelamento", "engavetamento", "queda de moto",
        "pane mecânica", "pane seca", "pane",
    ]))


def extrair_locais(texto: str) -> str:
    encontrados = [loc for loc, pat in LOCAIS_RE if pat.search(texto)]
    return "; ".join(encontrados) if encontrados else ""


# ── FUNÇÃO PRINCIPAL ───────────────────────────────────────────────────────────

def carregar_tweets(caminho_csv: str) -> pd.DataFrame:
    """
    Lê o CSV bruto do scraper e retorna um DataFrame processado,
    pronto para modelagem.

    Parâmetros
    ----------
    caminho_csv : str
        Caminho para o arquivo CSV (ex: 'ecoviasponte_tweets_completo.csv').

    Retorna
    -------
    pd.DataFrame com as colunas:
        data, hora_postagem,
        dia_semana, fim_de_semana, periodo_dia,
        mes, ano,
        tipo_tweet, sentido,
        tempo_travessia_min, condicao_label,
        tem_acidente, locais_mencionados,
        texto
    """

    print(f"📂 Lendo {caminho_csv}...")
    df = pd.read_csv(caminho_csv, low_memory=False, usecols=["createdAt", "text"])
    print(f"   {len(df):,} tweets carregados")

    # ── Data / hora ──────────────────────────────────────────────────────────
    print("🕐 Convertendo datas para BRT...")
    df["_ts"]     = df["createdAt"].apply(parse_datetime)
    df["_ts_brt"] = df["_ts"].dt.tz_convert("America/Sao_Paulo")

    df["data"]          = df["_ts_brt"].dt.date
    df["hora_postagem"] = df["_ts_brt"].dt.strftime("%H:%M")
    df["dia_semana"]    = df["_ts_brt"].dt.dayofweek.map(DIAS_PT)
    df["fim_de_semana"] = df["_ts_brt"].dt.dayofweek.isin([5, 6]).astype(int)
    df["periodo_dia"]   = df["_ts_brt"].dt.hour.apply(periodo_dia)
    df["mes"]           = df["_ts_brt"].dt.strftime("%m")
    df["ano"]           = df["_ts_brt"].dt.strftime("%Y")

    # ── Conteúdo ─────────────────────────────────────────────────────────────
    print("🔍 Extraindo features dos tweets...")
    df["texto"] = df["text"].str.strip()

    df["tipo_tweet"]          = df["texto"].apply(classificar_tipo)
    df["sentido"]             = df["texto"].apply(extrair_sentido)
    df["tempo_travessia_min"] = df["texto"].apply(extrair_tempo_travessia)
    df["condicao_label"]      = df["texto"].apply(classificar_condicao)
    df["tem_acidente"]        = df["texto"].apply(detectar_acidente)
    df["locais_mencionados"]  = df["texto"].apply(extrair_locais)

    # ── Colunas finais ────────────────────────────────────────────────────────
    colunas = [
        "data", "hora_postagem",
        "dia_semana", "fim_de_semana", "periodo_dia",
        "mes", "ano",
        "tipo_tweet", "sentido",
        "tempo_travessia_min", "condicao_label",
        "tem_acidente", "locais_mencionados",
        "texto",
    ]
    df_final = df[colunas].sort_values("data").reset_index(drop=True)

    # ── Resumo ────────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"✅ DataFrame pronto  |  {len(df_final):,} linhas  ×  {len(df_final.columns)} colunas")
    print(f"   Período: {df_final['data'].min()}  →  {df_final['data'].max()}")
    print(f"\n📊 Tipo de tweet:")
    print(df_final["tipo_tweet"].value_counts().to_string())
    print(f"\n🚗 Sentido:")
    print(df_final["sentido"].value_counts(dropna=False).to_string())
    print(f"\n🚦 Condição:")
    print(df_final["condicao_label"].value_counts(dropna=False).to_string())
    print(f"\n⚠️  Tweets com acidente : {df_final['tem_acidente'].sum():,}")
    print(f"{'='*55}\n")

    return df_final


# ── EXECUÇÃO DIRETA ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = carregar_tweets(INPUT_CSV)

    # Salvar CSV (opcional)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"💾 CSV salvo em: {OUTPUT_CSV}")

    print("\nPrimeiras linhas:")
    print(df.head(3).to_string())


# ── MERGE COM DADOS METEOROLÓGICOS (INMET) ─────────────────────────────────────
INMET_CSV = "inmet_niteroi_2025_2026.csv"


def cruzar_inmet(df: pd.DataFrame, caminho_inmet: str) -> pd.DataFrame:
    """
    Recebe o DataFrame de tweets já processado e adiciona as colunas
    meteorológicas do INMET, associando cada tweet à hora cheia correspondente.

    Lógica: tweet às 15:03, 15:27, 15:45 → dados do INMET das 15:00.

    Parâmetros
    ----------
    df            : DataFrame retornado por carregar_tweets()
    caminho_inmet : caminho para o CSV do INMET

    Retorna
    -------
    df_final : novo DataFrame com todas as colunas de df
               + precipitacao_mm, temperatura_C, vento_rajada_ms
    """

    print("📂 Lendo dados do INMET...")
    inmet = pd.read_csv(caminho_inmet)
    inmet["datetime_brt"] = pd.to_datetime(inmet["datetime_brt"]).dt.tz_convert("America/Sao_Paulo")
    inmet["_chave"] = inmet["datetime_brt"].dt.floor("h")
    inmet = inmet.drop_duplicates(subset="_chave")
    print(f"   {len(inmet):,} registros horários carregados")

    # Chave de merge: hora cheia de cada tweet
    aux = pd.to_datetime(
        df["data"].astype(str) + " " + df["hora_postagem"],
        format="%Y-%m-%d %H:%M"
    ).dt.tz_localize("America/Sao_Paulo").dt.floor("h").rename("_chave")

    cols_inmet = ["_chave", "precipitacao_mm", "temperatura_C", "vento_rajada_ms"]
    clima = aux.to_frame().merge(inmet[cols_inmet], on="_chave", how="left").drop(columns="_chave")

    df_final = pd.concat([df.reset_index(drop=True), clima.reset_index(drop=True)], axis=1)

    sem_dados = df_final["temperatura_C"].isna().sum()
    print(f"\n{'='*55}")
    print(f"✅ df_final pronto  |  {len(df_final):,} tweets")
    print(f"   Com dados meteorológicos : {len(df_final) - sem_dados:,}")
    print(f"   Sem correspondência INMET: {sem_dados:,} (falha da estação)")
    print(f"{'='*55}\n")

    return df_final


# ── EXECUÇÃO DIRETA ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df       = carregar_tweets(INPUT_CSV)
    df_final = cruzar_inmet(df, INMET_CSV)

    print("df      :", df.shape)
    print("df_final:", df_final.shape)
    print("\nAmostra df_final:")
    print(df_final[["data", "hora_postagem", "temperatura_C",
                     "precipitacao_mm", "vento_rajada_ms"]].head(5).to_string())
    
#%% Análise descritiva


