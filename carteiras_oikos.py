import io
import re
import glob
import os
import zipfile

import requests
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from PIL import Image

from Assets import BR_BENCH_TICKERS
from comparador import (
    benchmarks,
    safe_download,
    _get_br_slice,
    _calcular_metricas,
    _exibir_metricas,
    _graficos_comparacao,
    _tabela_rentabilidade_mensal,
    _load_cdi_bcb,
)

CARTEIRAS_FOLDER = "Carteiras Oikos/"
FUNDS_PARQUET = (
    "/Users/thiagocosta/Documents/Python Scripts/Otimizador Geral/fundos_inf_diario.parquet"
)

MESES_PT = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8,
    "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}


def _parse_month_year(filename):
    """Extrai (mês, ano) de 'Carteira Oikos Abril 2026.xlsx'."""
    base = os.path.basename(filename).lower().replace(".xlsx", "")
    for pt, num in MESES_PT.items():
        if pt in base:
            m = re.search(r"\d{4}", base)
            year = int(m.group()) if m else None
            return num, year
    return None, None


def _load_portfolio_file(path):
    """
    Lê todas as abas do Excel e retorna {nome_aba: DataFrame(Ativo, CNPJ, Peso)}.
    Detecta automaticamente as colunas — suporta nomes variáveis ("Peso"/"Porcentagem"),
    CNPJ opcional (abas de ETFs/Ações não têm CNPJ), e ignora colunas "Unnamed:" extras.
    """
    xl = pd.ExcelFile(path)
    result = {}
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        if df.empty or len(df.columns) < 2:
            continue
        cols = df.columns.tolist()

        # Coluna de peso: "Peso", "Porcentagem", ou qualquer variação
        peso_col = next(
            (c for c in cols if any(w in str(c).lower() for w in ["peso", "porcentagem", "pct"])),
            None,
        )
        if peso_col is None:
            continue

        # Coluna CNPJ (opcional — abas de ETFs/Ações BR não têm)
        cnpj_col = next((c for c in cols if "cnpj" in str(c).lower()), None)

        # Coluna de nome do ativo: primeira coluna nomeada que não seja CNPJ nem Peso
        # e que não seja "Unnamed:"
        ativo_col = next(
            (c for c in cols
             if c not in (cnpj_col, peso_col) and not str(c).startswith("Unnamed")),
            None,
        )
        if ativo_col is None:
            continue

        keep = [ativo_col, cnpj_col, peso_col] if cnpj_col else [ativo_col, peso_col]
        df = df[keep].copy()

        if cnpj_col:
            df.columns = ["Ativo", "CNPJ", "Peso"]
        else:
            df.columns = ["Ativo", "Peso"]
            df["CNPJ"] = ""

        df["Ativo"] = df["Ativo"].astype(str).str.strip()
        df["CNPJ"] = df["CNPJ"].fillna("").astype(str).str.strip().replace("nan", "")
        df["Peso"] = pd.to_numeric(df["Peso"], errors="coerce")
        df = df.dropna(subset=["Peso"])
        df = df[(df["Peso"] > 0) & (df["Ativo"] != "nan") & (df["Ativo"] != "")]
        if df.empty:
            continue
        df["Peso"] = df["Peso"] / df["Peso"].sum()
        result[sheet] = df[["Ativo", "CNPJ", "Peso"]].reset_index(drop=True)
    return result


@st.cache_resource(show_spinner="Carregando dados de fundos…")
def _load_funds_parquet():
    """Carrega o parquet de cotas de fundos gerado pelo funds_loader do Otimizador Geral."""
    df = pd.read_parquet(FUNDS_PARQUET)
    if df.index.name != "Date":
        df = df.rename_axis("Date")
    if "CNPJ" not in df.columns:
        for col in ("CNPJ_FUNDO_CLASSE", "CNPJ_CLASSE", "CNPJ_FUNDO"):
            if col in df.columns:
                df["CNPJ"] = df[col].astype(str).str.replace(r"\D", "", regex=True).str.zfill(14)
                break
    return df


@st.cache_resource(show_spinner=False)
def _load_cvm_zip(yyyymm):
    """Baixa e cacheia o arquivo de cotas da CVM para um mês (YYYYMM). Fallback para meses
    mais recentes que o parquet. Suporta tanto o formato antigo (CNPJ_FUNDO) quanto o novo
    (CNPJ_FUNDO_CLASSE), que passou a ser usado pela CVM a partir de 2026."""
    url = (
        f"https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/"
        f"inf_diario_fi_{yyyymm}.zip"
    )
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, sep=";", dtype=str, encoding="latin-1")
    df["DT_COMPTC"] = pd.to_datetime(df["DT_COMPTC"], errors="coerce")
    df["VL_QUOTA"] = pd.to_numeric(df["VL_QUOTA"], errors="coerce")
    df = df.dropna(subset=["DT_COMPTC", "VL_QUOTA"])
    # Normaliza o CNPJ — prioridade: CNPJ_FUNDO_CLASSE > CNPJ_CLASSE > CNPJ_FUNDO
    for col in ("CNPJ_FUNDO_CLASSE", "CNPJ_CLASSE", "CNPJ_FUNDO"):
        if col in df.columns:
            df["CNPJ"] = df[col].astype(str).str.replace(r"\D", "", regex=True).str.zfill(14)
            break
    return df.set_index("DT_COMPTC")


def _get_fund_returns(cnpj, start, end):
    """Retorna série de retornos diários de um fundo.
    Usa o parquet local (histórico até ~ago/2025) e faz fallback para download CVM
    nos meses mais recentes que o parquet não cobre.

    Usa MEDIANA por data para deduplicar cotas — a partir da Resolução CVM 175
    (2023-2024) o mesmo CNPJ pode ter múltiplas classes com cotas em escalas
    diferentes; a mediana descarta as classes outliers e mantém a dominante."""
    clean = re.sub(r"[.\-/]", "", cnpj).zfill(14)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    frames = []

    # ── 1. Parquet (rápido, histórico) ───────────────────────────────────
    info = _load_funds_parquet()
    parquet_end = info.index.max()
    if start_ts <= parquet_end:
        sub = info.loc[
            (info.index >= start_ts) & (info.index <= min(end_ts, parquet_end))
            & (info["CNPJ"] == clean), "VL_QUOTA"
        ].dropna()
        if not sub.empty:
            frames.append(sub.groupby(level=0).median().sort_index())

    # ── 2. CVM download para meses após o parquet ─────────────────────────
    cvm_from = max(start_ts, parquet_end + pd.Timedelta(days=1))
    if cvm_from <= end_ts:
        for m in pd.period_range(start=cvm_from, end=end_ts, freq="M"):
            try:
                df = _load_cvm_zip(m.strftime("%Y%m"))
                sub = df.loc[df["CNPJ"] == clean, "VL_QUOTA"].dropna()
                if not sub.empty:
                    frames.append(sub.groupby(level=0).median().sort_index())
            except Exception:
                pass

    if not frames:
        return pd.Series(dtype=float)
    quotas = pd.concat(frames).sort_index()
    # Mediana final cobre o caso em que parquet e CVM têm datas sobrepostas
    quotas = quotas.groupby(level=0).median()
    quotas = quotas[(quotas.index >= start_ts) & (quotas.index <= end_ts)]
    return quotas.pct_change().dropna()


def _parse_cdi_beta(asset_name):
    """Retorna o beta se o nome for do tipo 'X% CDI', senão None."""
    m = re.match(r"(\d+\.?\d*)\s*%\s*CDI", str(asset_name).strip(), re.IGNORECASE)
    return float(m.group(1)) / 100 if m else None


def _compute_portfolio_returns(portfolio_df, month, year):
    """
    Calcula retornos diários da carteira para o mês/ano informado.
    Retorna (pd.Series retornos diários, lista de warnings).
    """
    start = pd.Timestamp(year, month, 1)
    end = start + pd.offsets.MonthEnd(0)
    warns = []

    br_full = _get_br_slice(start=str(start.date()), end=str(end.date()))
    cdi_series = (
        br_full["CDI"].dropna()
        if (br_full is not None and not br_full.empty and "CDI" in br_full.columns)
        else None
    )

    asset_returns = {}

    for _, row in portfolio_df.iterrows():
        ativo = str(row["Ativo"]).strip()
        cnpj = str(row["CNPJ"]).strip()

        # ── CDI sintético: "X% CDI" ──────────────────────────────────────
        beta = _parse_cdi_beta(ativo)
        if beta is not None:
            if cdi_series is not None and not cdi_series.empty:
                asset_returns[ativo] = beta * cdi_series
            else:
                try:
                    cdi_s = _load_cdi_bcb(str(start.date()), str(end.date()))
                    if not cdi_s.empty:
                        asset_returns[ativo] = beta * cdi_s
                    else:
                        warns.append(f"CDI não disponível para '{ativo}'")
                except Exception:
                    warns.append(f"CDI não disponível para '{ativo}'")
            continue

        # ── Fundo com CNPJ → CVM ─────────────────────────────────────────
        if cnpj and cnpj not in ("", "nan"):
            s = _get_fund_returns(cnpj, start.date(), end.date())
            if s.empty:
                warns.append(f"Sem dados CVM para '{ativo}' ({cnpj})")
            else:
                asset_returns[ativo] = s
            continue

        # ── Ticker yfinance ───────────────────────────────────────────────
        def _yf_close(ticker):
            raw = yf.download(ticker, start=start, end=end + pd.Timedelta(days=3),
                              progress=False, auto_adjust=True)
            if raw.empty:
                return pd.Series(dtype=float)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            close = raw["Close"].squeeze() if "Close" in raw.columns else raw.squeeze()
            s = pd.Series(close).pct_change().dropna()
            return s[(s.index >= start) & (s.index <= end + pd.Timedelta(days=3))]

        try:
            s = _yf_close(ativo)
            # Fallback: adiciona .SA para tickers B3 sem sufixo (ex: PETR4 → PETR4.SA)
            if s.empty and "." not in ativo and not ativo.startswith("^") and not ativo.endswith("=X"):
                s = _yf_close(ativo + ".SA")
            if not s.empty:
                asset_returns[ativo] = s
            else:
                warns.append(f"Sem dados para '{ativo}'")
        except Exception as e:
            warns.append(f"Erro ao baixar '{ativo}': {e}")

    if not asset_returns:
        return pd.Series(dtype=float), warns

    returns_df = pd.DataFrame(asset_returns)
    ref_idx = returns_df.dropna(how="all").index
    returns_df = returns_df.reindex(ref_idx).ffill().dropna(how="all")

    weights = portfolio_df.set_index("Ativo")["Peso"]
    available = [t for t in returns_df.columns if t in weights.index]
    if not available:
        return pd.Series(dtype=float), warns

    w = np.array([weights.get(t, 0.0) for t in available])
    w = w / w.sum()
    port_ret = (returns_df[available] @ w).dropna()
    return port_ret, warns



@st.cache_resource(show_spinner=False)
def _compute_portfolios_cached(files_key: str):
    """Calcula os retornos de todas as carteiras para todos os meses.
    Usa files_key (nomes dos arquivos ordenados) como chave — recomputa só se
    novos arquivos forem adicionados à pasta. Persiste no servidor até reinicialização."""
    files = sorted(
        glob.glob(CARTEIRAS_FOLDER + "*.xlsx"),
        key=lambda f: (_parse_month_year(f)[1] or 0, _parse_month_year(f)[0] or 0),
    )
    all_results: dict = {}
    all_warns: list = []
    for f in files:
        m, y = _parse_month_year(f)
        if m is None or y is None:
            continue
        for sheet_name, portfolio_df in _load_portfolio_file(f).items():
            ret, warns = _compute_portfolio_returns(portfolio_df, m, y)
            all_warns.extend(warns)
            if not ret.empty:
                all_results.setdefault(sheet_name, []).append(ret)
    final = {}
    for sheet, chunks in all_results.items():
        combined = pd.concat(chunks).sort_index()
        final[sheet] = combined[~combined.index.duplicated(keep="last")]
    return final, all_warns


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_benchmark_series(benchmark_choice, start_str, end_str):
    """Retorna série de retornos diários do benchmark selecionado."""
    if benchmark_choice == "Nenhum":
        return pd.Series(dtype=float)
    ticker = benchmarks.get(benchmark_choice)
    if ticker is None:
        return pd.Series(dtype=float)
    if ticker in BR_BENCH_TICKERS:
        br = _get_br_slice(start=start_str, end=end_str)
        if ticker in br.columns:
            return br[ticker].dropna()
        if ticker == "CDI":
            try:
                return _load_cdi_bcb(start_str, end_str)
            except Exception:
                return pd.Series(dtype=float)
        return pd.Series(dtype=float)
    prices = safe_download(ticker, start=start_str, end=end_str)
    if prices.empty:
        return pd.Series(dtype=float)
    return pd.Series(prices).pct_change().dropna()


def show_carteiras():
    try:
        try:
            logo = Image.open("Logo Oikos Horizontal Colorido.png")
            st.image(logo, use_container_width=False, width=800)
        except Exception:
            pass

        st.title("Carteiras Oikos — Desempenho")

        files = sorted(
            glob.glob(CARTEIRAS_FOLDER + "*.xlsx"),
            key=lambda f: (
                _parse_month_year(f)[1] or 0,
                _parse_month_year(f)[0] or 0,
            ),
        )

        if not files:
            st.error(f"Nenhum arquivo Excel encontrado em '{CARTEIRAS_FOLDER}'.")
            return

        # Carrega todos os arquivos em ordem cronológica
        all_data = []
        sheet_names_ordered = []
        for f in files:
            m, y = _parse_month_year(f)
            if m is None or y is None:
                continue
            portfolios = _load_portfolio_file(f)
            if portfolios:
                all_data.append((m, y, portfolios))
                for s in portfolios:
                    if s not in sheet_names_ordered:
                        sheet_names_ordered.append(s)

        if not all_data:
            st.error("Nenhum arquivo válido encontrado.")
            return

        MESES_NOME = {v: k.capitalize() for k, v in MESES_PT.items()}
        first_m, first_y, _ = all_data[0]
        last_m,  last_y,  _ = all_data[-1]
        periodo = f"{MESES_NOME[first_m]} {first_y} → {MESES_NOME[last_m]} {last_y}"
        st.caption(f"Período: {periodo}  |  {len(all_data)} mês(es) carregado(s)")

        # Cálculo com cache no servidor — só recomputa se a lista de arquivos mudar
        files_key = ",".join(sorted(os.path.basename(f) for f in files))
        with st.spinner("Calculando desempenho das carteiras…"):
            results, all_warns = _compute_portfolios_cached(files_key)

        for w in all_warns:
            st.warning(w)

        if not sheet_names_ordered:
            return

        # Benchmark selectbox
        bm_options = ["Nenhum"] + list(benchmarks.keys())
        benchmark_choice = st.selectbox(
            "Benchmark para comparação:",
            options=bm_options,
            index=bm_options.index("CDI (Taxa DI)") if "CDI (Taxa DI)" in bm_options else 0,
            key="carteiras_benchmark",
        )

        period_start = str(pd.Timestamp(first_y, first_m, 1).date())
        period_end   = str((pd.Timestamp(last_y, last_m, 1) + pd.offsets.MonthEnd(0)).date())

        bm_series = pd.Series(dtype=float)
        if benchmark_choice != "Nenhum":
            with st.spinner(f"Carregando {benchmark_choice}…"):
                bm_series = _fetch_benchmark_series(benchmark_choice, period_start, period_end)
            if bm_series.empty:
                st.warning(f"Não foi possível carregar o benchmark: {benchmark_choice}")

        # Composição mais recente por estratégia (último mês com dados para aquela aba)
        latest_composition: dict = {}
        for m, y, portfolios in reversed(all_data):
            for sn, pdf in portfolios.items():
                if sn not in latest_composition:
                    latest_composition[sn] = (m, y, pdf)

        tabs = st.tabs(sheet_names_ordered)

        for tab, sheet_name in zip(tabs, sheet_names_ordered):
            with tab:
                # ── Composição da Carteira ──────────────────────────────────
                if sheet_name in latest_composition:
                    m_comp, y_comp, pdf_comp = latest_composition[sheet_name]
                    mes_nome_comp = MESES_NOME.get(m_comp, str(m_comp))
                    with st.expander(f"Composição da Carteira — {mes_nome_comp} {y_comp}", expanded=True):
                        disp = pdf_comp[["Ativo", "Peso"]].copy()
                        disp["Tipo"] = pdf_comp.apply(
                            lambda r: "Fundo (CVM)" if r["CNPJ"] else
                                      ("CDI Sint." if _parse_cdi_beta(r["Ativo"]) is not None else "ETF / Ação"),
                            axis=1,
                        )
                        disp = disp[["Ativo", "Tipo", "Peso"]].sort_values("Peso", ascending=False)
                        disp["Peso"] = disp["Peso"].apply(lambda x: f"{x:.2%}")
                        st.dataframe(disp, use_container_width=True, hide_index=True)

                if sheet_name not in results:
                    st.info("Sem dados de desempenho disponíveis para esta estratégia.")
                    continue

                ret_series = results[sheet_name]

                # Alinha benchmark ao período da carteira
                bm_aligned = pd.Series(dtype=float)
                if not bm_series.empty:
                    bm_aligned = bm_series.reindex(
                        bm_series.index[
                            (bm_series.index >= ret_series.index.min()) &
                            (bm_series.index <= ret_series.index.max())
                        ]
                    ).dropna()

                st.markdown("### Métricas de Desempenho")
                metricas = [_calcular_metricas(ret_series, rf=0.0, label=sheet_name)]
                if not bm_aligned.empty:
                    metricas.append(_calcular_metricas(bm_aligned, rf=0.0, label=benchmark_choice))
                _exibir_metricas(metricas)

                series_chart = {sheet_name: ret_series}
                if not bm_aligned.empty:
                    series_chart[benchmark_choice] = bm_aligned
                _graficos_comparacao(series_chart)

                _tabela_rentabilidade_mensal(
                    {sheet_name: ret_series},
                    benchmark_series=bm_aligned if not bm_aligned.empty else None,
                    benchmark_label=benchmark_choice if not bm_aligned.empty else None,
                )

    except Exception as e:
        st.error("Ocorreu um erro inesperado na página Carteiras Oikos.")
        st.text(f"Detalhes técnicos: {e}")
