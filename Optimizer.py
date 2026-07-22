import streamlit as st
from PIL import Image
import pandas as pd
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import plotly.graph_objects as go
from pypfopt import risk_models, expected_returns, EfficientFrontier
import empyrical 
from scipy.optimize import minimize
import time
import plotly.figure_factory as ff

from Assets import symbol_real, symbol_dolar, symbol_euro, symbol_br_indices, load_br_indices, BR_BENCH_TICKERS
from comparador import _tabela_rentabilidade_mensal, _calcular_metricas, _exibir_metricas, _compute_synthetic_returns, _get_rf_rate, _render_synth_expander, _render_constraints_expander, _build_opt_constraints, _render_report_ui, _yc_layout, _REPORT_PALETTE, _load_cdi_bcb


# ---------------------------------------------------------------------------
# Textos pré-prontos para o relatório PPTX do Optimizer — mesma estrutura de
# _TEXTOS_PADRAO em comparador.py, mas comparando a carteira otimizada com o
# benchmark escolhido (o Optimizer não tem uma "carteira atual" para comparar).
# Volatilidade e Drawdown ficam com texto genérico, sem comparação direta com
# o benchmark: contra um benchmark como o CDI essa comparação é trivial (o
# CDI praticamente não tem drawdown/volatilidade) e não agrega à análise.
# ---------------------------------------------------------------------------
_TEXTOS_PADRAO_OPT = {
    "fronteira": {
        "Explicação": (
            "A *fronteira eficiente* é o conjunto de carteiras que oferecem o máximo retorno esperado "
            "para cada nível de risco aceitável, ou equivalentemente, o mínimo risco para cada nível "
            "de retorno desejado. Construída através da teoria moderna de portfolio (Média-Variância), "
            "ela representa as combinações ótimas de ativos que eliminam o risco não-sistemático através "
            "da diversificação. Qualquer carteira localizada abaixo da fronteira é considerada *ineficiente*, "
            "pois oferece menor retorno para o mesmo risco ou maior risco para o mesmo retorno."
        ),
        "Análise do Gráfico": (
            "A carteira selecionada se posiciona na fronteira eficiente com retorno de {port_ret} e "
            "risco de {port_risco}, representando a combinação matematicamente ótima entre os ativos "
            "analisados para o nível de risco assumido. Para referência, o benchmark {benchmark_label} "
            "apresentou retorno de {bench_ret} no mesmo período. *A escolha do ponto da fronteira deve "
            "refletir o apetite de risco do investidor*, já que pontos mais à direita da curva oferecem "
            "maior retorno esperado em troca de maior volatilidade."
        ),
    },
    "metricas": {
        "Explicação": (
            "As métricas de desempenho consolidam a avaliação quantitativa da carteira analisada. "
            "O *Retorno Anualizado* mede o crescimento médio gerado ao longo do período, enquanto a "
            "*Volatilidade Anualizada* mensura a dispersão dos retornos como proxy de risco total. "
            "O *Índice de Sharpe* e o *Índice de Sortino* quantificam o retorno por unidade de risco total "
            "e de queda, respectivamente — valores superiores indicam melhor compensação pelo risco "
            "assumido. O *Drawdown Máximo* revela a maior perda acumulada de pico a vale no período."
        ),
        "Análise Comparativa": (
            "A carteira otimizada apresenta retorno anualizado de *{ret_port}* frente a {ret_bench} do "
            "benchmark {benchmark_label}, com volatilidade de {vol_port} (vs. {vol_bench}). O índice de "
            "Sharpe da carteira é de *{sharpe_port}*, e o drawdown máximo observado no período foi de "
            "{dd_port}. *Esses números consolidam o desempenho da estratégia de alocação ao longo do "
            "período analisado*, servindo de referência para o acompanhamento futuro da carteira."
        ),
    },
    "volatilidade": {
        "Explicação": (
            "A volatilidade móvel de 30 dias mensura a oscilação dos retornos em janelas rolantes de "
            "um mês, expressando o risco de forma dinâmica ao longo do tempo. Períodos de maior "
            "volatilidade sinalizam incerteza elevada no mercado ou eventos específicos que afetam "
            "os ativos da carteira, enquanto ciclos de menor volatilidade indicam estabilidade relativa. "
            "A análise temporal desta métrica permite identificar regimes de risco distintos e avaliar "
            "se a carteira amplifica ou atenua a volatilidade do mercado em episódios de estresse."
        ),
        "Análise do Gráfico": (
            "O gráfico acompanha a volatilidade da carteira em janelas móveis de 30 dias ao longo do "
            "período analisado, evidenciando os momentos de maior e menor oscilação dos retornos. A "
            "comparação com o benchmark selecionado ajuda a contextualizar o nível de risco assumido "
            "pela carteira frente a uma referência de mercado. *Picos de volatilidade tendem a coincidir "
            "com períodos de maior incerteza ou estresse nos mercados*, sendo esperado que a intensidade "
            "dessas oscilações varie conforme a composição e o grau de diversificação da carteira."
        ),
    },
    "drawdown": {
        "Explicação": (
            "O gráfico de drawdown representa a queda acumulada da carteira em relação ao seu pico "
            "histórico mais recente. Quanto maior a profundidade e a duração do drawdown, maior o impacto "
            "sobre o patrimônio e mais exigente o processo de recuperação. Drawdowns prolongados ou severos "
            "indicam exposição elevada a fatores de risco sistemático ou concentração em ativos "
            "correlacionados. A análise permite avaliar o quão exigente pode ser o processo de "
            "recuperação da carteira em períodos de turbulência do mercado."
        ),
        "Análise do Gráfico": (
            "O gráfico mostra a evolução do drawdown da carteira ao longo do tempo, isto é, a distância "
            "entre o valor atual e o pico mais recente atingido. Momentos de queda mais acentuada refletem "
            "períodos de maior estresse de mercado ou concentração em ativos correlacionados, enquanto a "
            "velocidade de recuperação após esses eventos indica a resiliência da alocação escolhida. "
            "*A comparação com o benchmark selecionado serve como referência* para avaliar se o perfil de "
            "risco da carteira está alinhado ao objetivo da estratégia."
        ),
    },
    "correlacao": {
        "Explicação": (
            "A matriz de correlação quantifica o grau de co-movimento entre os retornos dos ativos. "
            "Células em amarelo indicam *alta correlação (≥ 0,7)*, sugerindo diversificação limitada "
            "entre os pares. Células em azul refletem *correlação moderada (0,4–0,7)*, e células em "
            "vermelho (≤ −0,3) sinalizam *ativos com tendência de movimentação oposta*, que contribuem "
            "mais efetivamente para a redução do risco por diversificação. A diagonal em azul escuro "
            "representa a autocorrelação de cada ativo consigo mesmo (sempre igual a 1). Carteiras "
            "com baixa correlação média tendem a apresentar *menor volatilidade e drawdowns mais controlados*."
        ),
    },
}


def _preencher_analise_fronteira_opt(texto, m):
    """Fill {placeholders} in the Optimizer frontier analysis text."""
    def _pct(v):
        return f"{v * 100:.1f}".replace(".", ",") + "%"
    vals = {
        "port_ret":        _pct(m["port_ret"]),
        "port_risco":      _pct(m["port_risco"]),
        "bench_ret":       _pct(m["bench_ret"]) if pd.notna(m.get("bench_ret")) else "N/D",
        "benchmark_label": m.get("benchmark_label", "benchmark"),
    }
    try:
        return texto.format(**vals)
    except KeyError:
        return texto


def _preencher_analise_metricas_opt(texto, m):
    """Fill {placeholders} in the Optimizer metrics analysis text."""
    def _pct(v):
        return f"{v * 100:.2f}".replace(".", ",") + "%" if pd.notna(v) else "N/D"
    def _f2(v):
        return f"{v:.2f}".replace(".", ",") if pd.notna(v) else "N/D"
    vals = {
        "ret_port":        _pct(m["ret_port"]),
        "ret_bench":       _pct(m.get("ret_bench")),
        "vol_port":        _pct(m["vol_port"]),
        "vol_bench":       _pct(m.get("vol_bench")),
        "sharpe_port":     _f2(m["sharpe_port"]),
        "dd_port":         _pct(abs(m["dd_port"]) if pd.notna(m["dd_port"]) else np.nan),
        "dd_bench":        _pct(abs(m["dd_bench"]) if pd.notna(m.get("dd_bench")) else np.nan),
        "benchmark_label": m.get("benchmark_label", "benchmark"),
    }
    try:
        return texto.format(**vals)
    except KeyError:
        return texto


@st.cache_resource(show_spinner="Carregando índices BR (CDI, IMA-B, IHFA...)...")
def _get_br_full_opt():
    return load_br_indices()


def _get_br_slice_opt(start=None, end=None):
    full = _get_br_full_opt()
    if full.empty:
        return full
    result = full.copy()
    if start is not None:
        result = result[result.index >= pd.to_datetime(start)]
    if end is not None:
        result = result[result.index <= pd.to_datetime(end)]
    return result


def show_optimizer():

    try:

        # --- CONFIG PAGE ---
        #st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

        # --- LOGO ---
        logo = Image.open("logo_final.png")
        st.image(logo, use_container_width=False, width=800)


        # === SYMBOLS LIST ===
        # Benchmarks disponíveis
        benchmarks = {
                "Ibovespa (Brazil)": "^BVSP",
                "S&P 500 (SPY)": "SPY",
                "S&P 500 Index (^GSPC)": "^GSPC",
                "Dow Jones (^DJI)": "^DJI",
                "Nasdaq 100 (^NDX)": "^NDX",
                "Russell 2000 (^RUT)": "^RUT",
                "EURO STOXX 50": "^STOXX50E",
                "STOXX Europe 600": "^STOXX",
                "DAX (Germany)": "^GDAXI",
                "CAC 40 (France)": "^FCHI",
                "FTSE MIB (Italy)": "FTSEMIB.MI",
                "IBEX 35 (Spain)": "^IBEX",
                "FTSE 100 (UK)": "^FTSE",
                "Euro Corporate Bonds (ETF - IEAC.L)": "IEAC.L",
                "U.S. Treasury Bill ETF (Cash Equivalent)": "BIL",
                "CDI (Taxa DI)": "CDI",
                "IMA-B (NTN-B Total)": "IMA-B",
                "IHFA (Hedge Funds)": "IHFA",
            }

        symbol_info = symbol_real | symbol_dolar | symbol_euro | symbol_br_indices


        st.title("Otimizador de Portfólio")

        st.header("Selecione os Ativos para Otimização")

        available_symbols = sorted(symbol_info.keys(), key=lambda x: symbol_info.get(x, x))
        _synth_opt = st.session_state.get("synthetic_assets", {})

        # Sintéticos entram diretamente na lista de opções do multiselect
        all_opt_symbols = available_symbols + list(_synth_opt.keys())

        def _format_opt(s):
            if s in _synth_opt:
                _si = _synth_opt[s]
                return f"{s} (Sintético β={_si['beta']:.2g}×{_si['ref_ticker']}) [{s}]"
            return f"{symbol_info.get(s, s)} [{s}]"

        ticker_options_opt = [_format_opt(s) for s in all_opt_symbols]
        ticker_lookup_opt  = {_format_opt(s): s for s in all_opt_symbols}

        _render_synth_expander("opt", ticker_options_opt, ticker_lookup_opt)

        selected_symbols = st.multiselect(
            "Selecione os ativos para otimizar:",
            options=all_opt_symbols,
            default=["PETR4.SA", "TAEE11.SA", "WEGE3.SA", "BND", "TLT", "GLD"],
            format_func=_format_opt,
        )

        benchmark_choice = st.selectbox("Selecione o Benchmark:", list(benchmarks.keys()), index=0)
        currency_choice = st.selectbox("Selecione a Moeda:", ["Real", "Dollar", "Euro"], index=0)

        if len(selected_symbols) < 2:
            st.warning("Selecione pelo menos dois ativos.")
            st.stop()

        # --- Download de dados ---
        progress_bar_tab1 = st.progress(0, text="Starting data download...")

        today = pd.to_datetime('today').normalize()

        def safe_download(symbol, start, end, max_retries=10, wait=2):
            if symbol in symbol_br_indices:
                return pd.Series(dtype=float)
            for attempt in range(max_retries):
                try:
                    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
                    if 'Close' in df and not df['Close'].dropna().empty:
                        return df['Close']
                except Exception:
                    pass
                time.sleep(wait)
            return pd.Series(dtype=float)

        # Pré-carrega índices BR se algum foi selecionado
        br_selected = [s for s in selected_symbols if s in symbol_br_indices]
        br_idx = _get_br_slice_opt(start="2000-01-01", end=str(today)) if br_selected else None

        data = pd.DataFrame()
        valid_symbols = []
        total_assets_tab1 = len(selected_symbols)
        for i, symbol in enumerate(selected_symbols, start=1):
            if symbol in symbol_br_indices:
                if br_idx is not None and symbol in br_idx.columns:
                    valid_symbols.append(symbol)
                else:
                    st.warning(f"⚠️ Dados não disponíveis para: {symbol}")
            elif symbol in _synth_opt:
                valid_symbols.append(symbol)
            else:
                close_data = safe_download(symbol, start="2000-01-01", end=today)
                if not close_data.empty:
                    data[symbol] = close_data
                    valid_symbols.append(symbol)
                else:
                    st.warning(f"⚠️ Failed to download data for: {symbol}")
            progress_value = int((i / max(total_assets_tab1, 1)) * 70)
            progress_bar_tab1.progress(progress_value, text=f"Downloading selected assets... {i}/{total_assets_tab1}")

        if len(valid_symbols) < 2:
            st.error("Poucos ativos válidos para otimizar. Ajuste sua seleção.")
            st.stop()

        data_benchmark = pd.DataFrame()
        total_benchmarks_tab1 = len(benchmarks)
        for j, (name, ticker) in enumerate(benchmarks.items(), start=1):
            if ticker not in BR_BENCH_TICKERS:
                data_benchmark[name] = safe_download(ticker, start="2000-01-01", end=today)
            progress_value = 70 + int((j / max(total_benchmarks_tab1, 1)) * 20)
            progress_bar_tab1.progress(progress_value, text=f"Downloading benchmarks... {j}/{total_benchmarks_tab1}")

        # #Baixando o CDI
        # var_ok = 0 
        # while var_ok != 1:
        #     try:
        #         url = 'http://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?formato=json'
        #         cdi = pd.read_json(url)
        #         cdi['data'] = pd.to_datetime(cdi['data'], dayfirst=True)
        #         cdi.rename({'data': 'Date','valor':'CDI'}, axis='columns', inplace= True)
        #         cdi.set_index('Date', inplace = True)
        #         cdi['CDI']=cdi['CDI']/100
        #         var_ok = 1 
        #     except:
        #         0

        # #Baixando o IMA-B
        # var_ok = 0
        # while var_ok != 1:
        #     try:
        #         url2 = 'http://api.bcb.gov.br/dados/serie/bcdata.sgs.12466/dados?formato=json'
        #         imab = pd.read_json(url2)
        #         imab['data'] = pd.to_datetime(imab['data'], dayfirst=True)
        #         imab.rename({'data': 'Date','valor':'Ima-B'}, axis='columns', inplace= True)
        #         imab.set_index('Date', inplace = True)
        #         var_ok = 1
        #     except:
        #         0

        # data_benchmark['IMA-B'] = imab['Ima-B']

        progress_bar_tab1.progress(95, text="Calculating returns...")
        returns = np.log(data / data.shift(1)).dropna() if not data.empty else pd.DataFrame()

        # Mescla retornos dos índices BR selecionados como ativos
        br_asset_cols = [s for s in valid_symbols if s in symbol_br_indices]
        if br_idx is not None and br_asset_cols:
            if returns.empty:
                returns = br_idx[br_asset_cols].dropna()
            else:
                for s in br_asset_cols:
                    if s in br_idx.columns:
                        returns[s] = br_idx[s]
                avail_br = [s for s in br_asset_cols if s in returns.columns]
                if avail_br:
                    returns = returns.dropna(subset=avail_br)

        # Ativos sintéticos: adiciona ao returns antes da otimização
        for _sn in [s for s in selected_symbols if s in _synth_opt]:
            _si = _synth_opt[_sn]
            _sr = _compute_synthetic_returns(_si["ref_ticker"], "2000-01-01", str(today))
            if _sr is not None:
                returns[_sn] = _si["beta"] * _sr
                valid_symbols.append(_sn)
            else:
                st.warning(f"⚠️ Não foi possível calcular retornos para o ativo sintético '{_sn}' (referência: {_si['ref_ticker']}).")

        returns_benchmark_all = np.log(data_benchmark / data_benchmark.shift(1)).dropna() if not data_benchmark.empty else pd.DataFrame()

        # Mescla benchmarks BR (CDI / IMA-B / IHFA)
        if br_idx is None:
            br_idx = _get_br_slice_opt(start="2000-01-01", end=str(today))
        for bm_name, bm_ticker in benchmarks.items():
            if bm_ticker in BR_BENCH_TICKERS and bm_ticker in br_idx.columns:
                returns_benchmark_all[bm_name] = br_idx[bm_ticker]
        
        dollar_brl = safe_download("USDBRL=X", start="2000-01-01", end=today)
        euro_brl = safe_download("EURBRL=X", start="2000-01-01", end=today)
        usd_eur = safe_download("USDEUR=X", start="2000-01-01", end=today) 

    # --- Ajustar retorno para moeda escolhida ---
        if currency_choice == "Real":
            for symbol in returns.columns:
                if symbol in symbol_dolar:
                    aligned = np.log(dollar_brl / dollar_brl.shift(1)).dropna()
                    aligned = aligned.reindex(returns.index).dropna()
                    returns = returns.loc[aligned.index]
                    returns[symbol] = (1 + returns[symbol]) * (1 + aligned["USDBRL=X"]) - 1
                elif symbol in symbol_euro:
                    aligned = np.log(euro_brl / euro_brl.shift(1)).dropna()
                    aligned = aligned.reindex(returns.index).dropna()
                    returns = returns.loc[aligned.index]
                    returns[symbol] = (1 + returns[symbol]) * (1 + aligned["EURBRL=X"]) - 1

        elif currency_choice == "Dollar":
            for symbol in returns.columns:
                if symbol in symbol_real:
                    aligned = -np.log(dollar_brl / dollar_brl.shift(1)).dropna()
                    aligned = aligned.reindex(returns.index).dropna()
                    returns = returns.loc[aligned.index]
                    returns[symbol] = (1 + returns[symbol]) * (1 + aligned["USDBRL=X"]) - 1
                elif symbol in symbol_euro:
                    aligned = np.log(usd_eur / usd_eur.shift(1)).dropna()
                    aligned = aligned.reindex(returns.index).dropna()
                    returns = returns.loc[aligned.index]
                    returns[symbol] = (1 + returns[symbol]) * (1 + aligned["USDEUR=X"]) - 1

        elif currency_choice == "Euro":
            for symbol in returns.columns:
                if symbol in symbol_real:
                    aligned = -np.log(euro_brl / euro_brl.shift(1)).dropna()
                    aligned = aligned.reindex(returns.index).dropna()
                    returns = returns.loc[aligned.index]
                    returns[symbol] = (1 + returns[symbol]) * (1 + aligned["EURBRL=X"]) - 1
                elif symbol in symbol_dolar:
                    aligned = -np.log(usd_eur / usd_eur.shift(1)).dropna()
                    aligned = aligned.reindex(returns.index).dropna()
                    returns = returns.loc[aligned.index]
                    returns[symbol] = (1 + returns[symbol]) * (1 + aligned["USDEUR=X"]) - 1

        cov_matrix = np.asarray(252 * returns.cov())
        annualized_returns = np.asarray(252 * returns.mean())

        first_available_date = returns.index.min()
        last_available_date = returns.index.max()
        
        risk_measure = st.selectbox(
            "Selecione o Método de Construção da Fronteira",
            ["Mean-Variance", "Mean-Semivariance", "Mean Absolute Deviation (MAD)", "CVaR / Expected Shortfall"],
            index=0,
            key="risk_measure_tab1"
        )

        confidence_level = 0.95
        if risk_measure == "CVaR / Expected Shortfall":
            confidence_level = st.slider(
                "Nível de Confiança",
                min_value=0.90,
                max_value=0.99,
                value=0.95,
                step=0.01,
                key="confidence_level_tab1"
            )

        progress_bar_tab1.progress(100, text="Download completed.")
        progress_bar_tab1.empty()
        st.success(f"Data downloaded successfully! Available from {first_available_date.date()} to {last_available_date.date()}.")

        # --- Restrições de Alocação ---
        g_min_opt, g_max_opt, per_asset_opt, groups_opt = _render_constraints_expander(
            list(returns.columns), "opt_main"
        )

        # --- Otimização (Fronteira Eficiente) ---
        st.header("Fronteira Eficiente")

        n_opt = len(returns.columns)
        bounds = [per_asset_opt.get(t, (g_min_opt, g_max_opt)) for t in returns.columns]
        initial_guess = [1 / n_opt] * n_opt
        rent_alvo = np.arange(0.01, 1.5, 0.005)

        def portfolio_variance(w):
            return np.dot(w.T, np.dot(cov_matrix, w))

        def portfolio_return(w):
            return np.dot(w, annualized_returns)

        def portfolio_returns_series(w):
            return returns @ np.asarray(w)

        def portfolio_semivariance(w):
            port_rets = portfolio_returns_series(w)
            downside = np.minimum(port_rets, 0)
            return 252 * np.mean(np.square(downside))

        def portfolio_mad(w):
            port_rets = portfolio_returns_series(w)
            mean_ret = np.mean(port_rets)
            return np.mean(np.abs(port_rets - mean_ret)) * np.sqrt(252)

        def portfolio_var(w, alpha):
            port_rets = portfolio_returns_series(w)
            if len(port_rets) == 0:
                return np.nan
            return -np.quantile(port_rets, 1 - alpha)

        def portfolio_cvar(w, alpha):
            port_rets = portfolio_returns_series(w)
            if len(port_rets) == 0:
                return np.nan
            threshold = np.quantile(port_rets, 1 - alpha)
            tail_losses = port_rets[port_rets <= threshold]
            if len(tail_losses) == 0:
                return -threshold
            return -tail_losses.mean()

        def portfolio_risk(w):
            if risk_measure == "Mean-Variance":
                return np.sqrt(portfolio_variance(w))
            elif risk_measure == "Mean-Semivariance":
                return np.sqrt(portfolio_semivariance(w))
            elif risk_measure == "Mean Absolute Deviation (MAD)":
                return portfolio_mad(w)
            elif risk_measure == "CVaR / Expected Shortfall":
                return portfolio_cvar(w, alpha=confidence_level)
            return np.sqrt(portfolio_variance(w))

        def minimize_portfolio_risk(min_return):
            base_cons = [
                {'type': 'eq',   'fun': lambda w: np.sum(w) - 1},
                {'type': 'ineq', 'fun': lambda w: portfolio_return(w) - min_return},
            ]
            grp_cons = _build_opt_constraints(list(returns.columns), groups_opt)
            return minimize(portfolio_risk, initial_guess,
                            method='SLSQP',
                            bounds=bounds,
                            constraints=base_cons + grp_cons)

        frontier_points = []
        weights_frontier = pd.DataFrame()

        for r_target in rent_alvo:
            result = minimize_portfolio_risk(r_target)
            if result.success:
                w_opt = result.x
                risk_value = portfolio_risk(w_opt)
                ret = portfolio_return(w_opt)
                if np.isfinite(risk_value) and np.isfinite(ret):
                    frontier_points.append((risk_value, ret))
                    for idx, sym in enumerate(returns.columns):
                        weights_frontier.at[round(risk_value, 6), sym] = w_opt[idx]

        frontier_df = pd.DataFrame(frontier_points, columns=['Risk', 'Expected Return'])
        frontier_df = frontier_df.drop_duplicates(subset=['Risk', 'Expected Return']).sort_values('Risk').reset_index(drop=True)

        # Ativo livre de risco da moeda escolhida — usado tanto para achar a
        # carteira de máximo Sharpe de verdade (abaixo) quanto na mistura de
        # Alocação em Renda Fixa (mais adiante).
        _RF_ASSETS = {
            "Real":   ("CDI",     "CDI (Taxa DI)"),
            "Dollar": ("BIL",     "SPDR Bloomberg 1-3 Month T-Bill ETF (BIL)"),
            "Euro":   ("EXVM.DE", "iShares eb.rexx Government Germany 0-1yr (EXVM.DE)"),
        }
        _rf_ticker, _rf_label = _RF_ASSETS[currency_choice]

        def _get_rf_log_series():
            """Série de log-retornos diários do ativo livre de risco da moeda
            escolhida, com fallback direto na API do BCB para o CDI (mesma
            estratégia usada em _get_rf_rate)."""
            if _rf_ticker == "CDI":
                if br_idx is not None and "CDI" in br_idx.columns:
                    return np.log1p(br_idx["CDI"].dropna())
                try:
                    _cdi_fb = _load_cdi_bcb("2000-01-01", str(today.date()))
                    return np.log1p(_cdi_fb) if not _cdi_fb.empty else pd.Series(dtype=float)
                except Exception:
                    return pd.Series(dtype=float)
            elif _rf_ticker == "BIL":
                return returns_benchmark_all.get(
                    "U.S. Treasury Bill ETF (Cash Equivalent)", pd.Series(dtype=float)
                )
            else:
                # .squeeze() normaliza para Series: algumas versões do yfinance
                # retornam df['Close'] como DataFrame de 1 coluna (colunas
                # MultiIndex) mesmo para download de um único ticker.
                _price = safe_download(_rf_ticker, start="2000-01-01", end=today).squeeze()
                return np.log(_price / _price.shift(1)).dropna() if not _price.empty else pd.Series(dtype=float)

        # Taxa livre de risco anualizada, calculada sobre o mesmo período usado
        # para construir a fronteira (todo o histórico baixado, igual a
        # annualized_returns/cov_matrix) — não uma janela arbitrária diferente.
        # É essa taxa que define a carteira de máximo Sharpe/tangência: usar
        # rf=0 (como uma versão anterior fazia) identifica o ponto errado, já
        # que a CML só é tangente à carteira que maximiza (Retorno − rf) / Risco
        # com o rf de verdade.
        _rf_log_full = _get_rf_log_series()
        rf_annual = float(np.expm1(_rf_log_full.mean() * 252)) if not _rf_log_full.empty else 0.0

        frontier_df['Sharpe'] = (frontier_df['Expected Return'] - rf_annual) / frontier_df['Risk']
        frontier_df = frontier_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Sharpe'])

        min_vol_idx = frontier_df['Risk'].idxmin()
        max_ret_idx = frontier_df['Expected Return'].idxmax()
        max_sharpe_idx = frontier_df['Sharpe'].idxmax()

        # --- Escolha da carteira para análise --- (feito antes do gráfico da
        # fronteira para que o ponto escolhido já apareça marcado nele)
        st.header("Selecione o Portfólio para Análise")

        _tags = {
            min_vol_idx:    "[Mín. Volatilidade]  ",
            max_ret_idx:    "[Máx. Retorno]  ",
            max_sharpe_idx: "[Máx. Sharpe]  ",
        }
        frontier_labels = [
            f"{_tags.get(i, '')}Retorno: {row['Expected Return']:.1%}  |  Risco: {row['Risk']:.1%}"
            for i, row in frontier_df.iterrows()
        ]

        selected_label = st.selectbox(
            "Escolha qual portfólio analisar:",
            options=frontier_labels,
            index=int(max_sharpe_idx),
            key="portfolio_choice_opt",
        )
        idx_selected = frontier_labels.index(selected_label)
        selected_sigma = frontier_df.loc[idx_selected, "Risk"]

        selected_weights = weights_frontier.loc[round(selected_sigma, 6)]
        selected_weights = selected_weights[selected_weights > 0]

        # --- Plot Fronteira ---
        st.subheader("Fronteira Eficiente")

        add_cml = st.checkbox(
            "Adicionar Linha de Mercado de Capitais (CML)",
            value=False,
            key="add_cml_opt",
            help="Reta que liga o ativo livre de risco à carteira de máximo Sharpe (tangência).",
        )

        fig_frontier = go.Figure()
        fig_frontier.add_trace(go.Scatter(
            x=frontier_df['Risk'],
            y=frontier_df['Expected Return'],
            mode='lines+markers',
            name='Efficient Frontier'
        ))

        highlight_points = {
            "Minimum Volatility": (min_vol_idx, 'blue', 'diamond'),
            "Maximum Return": (max_ret_idx, 'green', 'square'),
            "Maximum Sharpe": (max_sharpe_idx, 'red', 'star')
        }

        for label, (idx, color, symbol) in highlight_points.items():
            fig_frontier.add_trace(go.Scatter(
                x=[frontier_df.loc[idx, 'Risk']],
                y=[frontier_df.loc[idx, 'Expected Return']],
                mode='markers+text',
                name=label,
                text=[label],
                textposition='bottom center',
                marker=dict(size=12, symbol=symbol, color=color)
            ))

        # Carteira escolhida em "Selecione o Portfólio para Análise" — marcada
        # separadamente mesmo quando coincide com um dos pontos acima, para
        # deixar claro qual é a carteira efetivamente usada na análise abaixo.
        fig_frontier.add_trace(go.Scatter(
            x=[frontier_df.loc[idx_selected, "Risk"]],
            y=[frontier_df.loc[idx_selected, "Expected Return"]],
            mode="markers+text",
            name="Selecionado",
            text=["Selecionado"],
            textposition="top center",
            marker=dict(size=16, symbol="circle-open", color="orange", line=dict(width=3)),
        ))

        # --- CML (Capital Market Line): reta que liga o ativo livre de risco
        # à carteira de máximo Sharpe — que já é a tangência de verdade, pois
        # o Sharpe da fronteira agora usa o rf real (calculado acima). ---
        if add_cml:
            if _rf_log_full.empty:
                st.warning(
                    f"Não foi possível obter dados do ativo livre de risco ({_rf_ticker}) "
                    "para desenhar a CML."
                )
            else:
                sigma_tan = float(frontier_df.loc[max_sharpe_idx, "Risk"])
                mu_tan    = float(frontier_df.loc[max_sharpe_idx, "Expected Return"])
                if sigma_tan > 0:
                    slope_cml = (mu_tan - rf_annual) / sigma_tan
                    x_line = np.linspace(0.0, float(frontier_df["Risk"].max()) * 1.1, 50)
                    y_line = rf_annual + slope_cml * x_line
                    fig_frontier.add_trace(go.Scatter(
                        x=x_line, y=y_line, mode="lines",
                        name=f"CML (rf={_rf_label}: {rf_annual:.2%})",
                        line=dict(dash="dash", color="black"),
                    ))
                    fig_frontier.add_trace(go.Scatter(
                        x=[0.0], y=[rf_annual], mode="markers+text",
                        name="Ativo Livre de Risco",
                        text=[_rf_ticker], textposition="top center",
                        marker=dict(size=10, symbol="circle", color="black"),
                    ))

        fig_frontier.update_layout(
            title=f"Efficient Frontier with Key Portfolios Highlighted ({risk_measure})",
            xaxis_title=f"Portfolio Risk ({risk_measure})",
            yaxis_title="Portfolio Expected Return",
            template="plotly_white"
        )

        st.plotly_chart(fig_frontier, use_container_width=True)

        # --- Alocação em Renda Fixa (ativo livre de risco, estilo Capital Market Line:
        # mistura o ativo livre de risco com a carteira arriscada escolhida acima,
        # proporcionalmente ao peso definido no slider). Escolhido antes da tabela de
        # composição para que ela já mostre a carteira final (risco + RF) de uma vez só. ---
        st.subheader("Alocação em Renda Fixa (Ativo Livre de Risco)")
        st.caption(f"Ativo livre de risco para {currency_choice}: **{_rf_label}**")
        peso_rf = st.slider(
            "Percentual da carteira alocado no ativo livre de risco:",
            min_value=0.0, max_value=1.0, value=0.0, step=0.01,
            key="peso_rf_opt",
        )

        if peso_rf > 0:
            selected_weights_final = selected_weights * (1 - peso_rf)
            selected_weights_final.loc[_rf_ticker] = peso_rf
        else:
            selected_weights_final = selected_weights.copy()

        st.subheader("Composição do Portfólio")

        composition_df = pd.DataFrame(selected_weights_final)
        composition_df.columns = ['Weight']
        st.dataframe(composition_df.style.format({"Weight": "{:.2%}"}))

        # --- Seleção de período ---
        st.header("Selecione o Período de Análise")

        start_analysis = st.date_input(
            "Data Inicial", 
            min_value=first_available_date.date(), 
            max_value=last_available_date.date(), 
            value=first_available_date.date()
        )

        end_analysis = st.date_input(
            "Data Final", 
            min_value=first_available_date.date(), 
            max_value=last_available_date.date(), 
            value=last_available_date.date()
        )

        if start_analysis >= end_analysis:
            st.warning("A data inicial deve ser anterior à data final.")
            st.stop()

        # --- Performance ---
        st.header("Resumo de Performance")

        returns_portfolio = returns[selected_weights.index] @ selected_weights
        returns_analysis = returns_portfolio[start_analysis:end_analysis].dropna()

        # Mistura com o ativo livre de risco (peso_rf, definido acima): converte a
        # carteira arriscada e o ativo livre de risco para retorno simples, combina
        # proporcionalmente e volta para log-retorno — mesma convenção usada no
        # restante do código (returns_analysis permanece em log-retorno). Os pesos
        # (selected_weights_final) já foram combinados acima, para a tabela de
        # Composição do Portfólio.
        if peso_rf > 0:
            _rf_log_full = _get_rf_log_series()

            if _rf_log_full.empty:
                st.warning(
                    f"Não foi possível obter dados do ativo livre de risco ({_rf_ticker}); "
                    "alocação em renda fixa não aplicada."
                )
                _common_rf_idx = pd.DatetimeIndex([])
            else:
                _rf_log_full.index = pd.to_datetime(_rf_log_full.index)
                _rf_log_analysis = _rf_log_full[start_analysis:end_analysis]
                _common_rf_idx = returns_analysis.index.intersection(_rf_log_analysis.index)
                if _common_rf_idx.empty:
                    st.warning(
                        f"Sem dados do ativo livre de risco ({_rf_ticker}) no período "
                        "selecionado; alocação em renda fixa não aplicada."
                    )

            if not _common_rf_idx.empty:
                _risky_simple = np.expm1(returns_analysis.loc[_common_rf_idx])
                _rf_simple    = np.expm1(_rf_log_analysis.loc[_common_rf_idx])
                _combo_simple = peso_rf * _rf_simple + (1 - peso_rf) * _risky_simple
                returns_analysis = np.log1p(_combo_simple)

        rf_rate = _get_rf_rate(currency_choice, start_analysis, end_analysis, returns_analysis.index)

        returns_benchmark = returns_benchmark_all[benchmark_choice][start_analysis:end_analysis]

        # Sharpe Ratio manual
        mean_return = float(returns_analysis.mean())
        std_dev = float(returns_analysis.std())
        rf_mean = rf_rate
        sharpe_ratio = ((mean_return - rf_mean) / std_dev) * np.sqrt(252)

        downside_returns_tab1 = np.minimum(returns_analysis - rf_mean, 0)
        downside_deviation_daily_tab1 = np.sqrt(np.mean(np.square(downside_returns_tab1)))
        sortino_ratio = np.nan if downside_deviation_daily_tab1 == 0 else (((mean_return - rf_mean) / downside_deviation_daily_tab1) * np.sqrt(252))

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Expected Return", f"{empyrical.annual_return(returns_analysis):.2%}")
        col2.metric("Risk (Std Dev)", f"{empyrical.annual_volatility(returns_analysis):.2%}")
        col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
        col4.metric("Sortino Ratio", f"{sortino_ratio:.2f}" if pd.notna(sortino_ratio) else "N/A")
        col5.metric("Maximum Drawdown", f"{empyrical.max_drawdown(returns_analysis):.2%}")
        col6.metric("VaR (1%)", f"{empyrical.value_at_risk(returns_analysis, cutoff=0.01):.2%}")
        
        # --- Gráfico Retorno Acumulado ---
        st.header("Comparação de Retornos Acumulados")

        # Garantir que os índices estejam alinhados para os gráficos
        common_index = returns_analysis.index.intersection(returns_benchmark.index)

        # Reindexar as séries
        returns_analysis = returns_analysis.loc[common_index]
        returns_benchmark = returns_benchmark.loc[common_index]

        cumulative_portfolio = (1 + returns_analysis).cumprod()
        cumulative_benchmark = (1 + returns_benchmark).cumprod()

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(x=cumulative_portfolio.index, y=cumulative_portfolio.values, mode='lines', name='Portfolio'))
        fig_cum.add_trace(go.Scatter(x=cumulative_benchmark.index, y=cumulative_benchmark.values, mode='lines', name=benchmark_choice))
        fig_cum.update_layout(
            title="Cumulative Returns: Portfolio vs Benchmark",
            xaxis_title="Date",
            yaxis_title="Cumulative Value",
            template="plotly_white"
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        # --- Rolling Volatility ---
        st.header("Volatilidade Móvel (30 dias)")

        rolling_vol_portfolio = returns_analysis.rolling(30).std() * np.sqrt(252)
        rolling_vol_benchmark = returns_benchmark.rolling(30).std() * np.sqrt(252)

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=rolling_vol_portfolio.index,
            y=rolling_vol_portfolio.values,
            mode='lines',
            name='Portfolio',
            line=dict(color='blue')
        ))
        fig_vol.add_trace(go.Scatter(
            x=rolling_vol_benchmark.index,
            y=rolling_vol_benchmark.values,
            mode='lines',
            name=benchmark_choice,
            line=dict(color='red')
        ))
        fig_vol.update_layout(
            title="Rolling Volatility (30 Days)",
            xaxis_title="Date",
            yaxis_title="Volatility (Annualized)",
            template="plotly_white"
        )
        st.plotly_chart(fig_vol, use_container_width=True)


        # --- Drawdown ---
        st.header("Drawdown do Portfólio")

        # Carteira
        cumulative_returns_portfolio = (1 + returns_analysis).cumprod()
        running_max_portfolio = cumulative_returns_portfolio.cummax()
        drawdown_portfolio = (cumulative_returns_portfolio / running_max_portfolio) - 1

        # Benchmark
        cumulative_returns_benchmark = (1 + returns_benchmark).cumprod()
        running_max_benchmark = cumulative_returns_benchmark.cummax()
        drawdown_benchmark = (cumulative_returns_benchmark / running_max_benchmark) - 1

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=drawdown_portfolio.index,
            y=drawdown_portfolio.values,
            mode='lines',
            name='Portfolio',
            line=dict(color='blue')
        ))
        fig_dd.add_trace(go.Scatter(
            x=drawdown_benchmark.index,
            y=drawdown_benchmark.values,
            mode='lines',
            name=benchmark_choice,
            line=dict(color='red')
        ))
        fig_dd.update_layout(
            title="Portfolio vs Benchmark Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown",
            template="plotly_white"
        )
        st.plotly_chart(fig_dd, use_container_width=True)


        # --- Correlation Matrix ---
        st.markdown("### Matriz de Correlação")
        correlation_matrix = returns.corr()
        annot_text = correlation_matrix.round(2).astype(str).values.tolist()
        fig_corr = ff.create_annotated_heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns.tolist(),
            y=correlation_matrix.index.tolist(),
            annotation_text=annot_text,
            colorscale='RdBu',
            showscale=True,
            reversescale=True,
            zmin=-1,
            zmax=1
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # --- Rentabilidade Mensal ---
        _tabela_rentabilidade_mensal(
            {"Portfólio": returns_analysis},
            benchmark_series=returns_benchmark if not returns_benchmark.empty else None,
            benchmark_label=benchmark_choice if not returns_benchmark.empty else None,
        )

        # --- Relatório PPTX ---
        m_port = _calcular_metricas(returns_analysis, rf_rate, "Carteira Otimizada")
        series_dict = {"Carteira Otimizada": returns_analysis}
        if not returns_benchmark.empty:
            series_dict[benchmark_choice] = returns_benchmark
            m_bench = _calcular_metricas(returns_benchmark, rf_rate, benchmark_choice)
            st.session_state["opt_frontier_metrics"] = {
                "port_ret":        float(frontier_df.loc[idx_selected, "Expected Return"]),
                "port_risco":      float(frontier_df.loc[idx_selected, "Risk"]),
                "bench_ret":       m_bench["retorno_anual"],
                "benchmark_label": benchmark_choice,
            }
            st.session_state["opt_metrics_data"] = {
                "ret_port":        m_port["retorno_anual"],
                "ret_bench":       m_bench["retorno_anual"],
                "vol_port":        m_port["volatilidade"],
                "vol_bench":       m_bench["volatilidade"],
                "sharpe_port":     m_port["sharpe"],
                "dd_port":         m_port["max_drawdown"],
                "dd_bench":        m_bench["max_drawdown"],
                "benchmark_label": benchmark_choice,
            }
        else:
            st.session_state["opt_frontier_metrics"] = {}
            st.session_state["opt_metrics_data"] = {}

        # Versão limpa (em português, sem título duplicado) da fronteira para o
        # relatório — o gráfico interativo em tela (fig_frontier) fica em inglês
        # e com título próprio, mas o slide do relatório já tem seu próprio
        # título ("Fronteira Eficiente"), então o gráfico não precisa de outro.
        _pt_labels_pt = {
            "Minimum Volatility": "Mínima Volatilidade",
            "Maximum Return":     "Máximo Retorno",
            "Maximum Sharpe":     "Máximo Sharpe",
        }
        _fig_frontier_report_opt = go.Figure()
        _fig_frontier_report_opt.add_trace(go.Scatter(
            x=frontier_df["Risk"], y=frontier_df["Expected Return"],
            mode="lines", name="Fronteira Eficiente",
            line=dict(color=_REPORT_PALETTE[0], width=2),
        ))
        for label, (idx, color, symbol) in highlight_points.items():
            _label_pt = _pt_labels_pt.get(label, label)
            _fig_frontier_report_opt.add_trace(go.Scatter(
                x=[frontier_df.loc[idx, "Risk"]], y=[frontier_df.loc[idx, "Expected Return"]],
                mode="markers+text", name=_label_pt,
                text=[_label_pt], textposition="bottom center",
                marker=dict(size=12, symbol=symbol, color=color),
            ))
        _fig_frontier_report_opt.add_trace(go.Scatter(
            x=[frontier_df.loc[idx_selected, "Risk"]], y=[frontier_df.loc[idx_selected, "Expected Return"]],
            mode="markers+text", name="Selecionado",
            text=["Selecionado"], textposition="top center",
            marker=dict(size=16, symbol="circle-open", color="orange", line=dict(width=3)),
        ))
        if add_cml and not _rf_log_full.empty:
            _sigma_tan = float(frontier_df.loc[max_sharpe_idx, "Risk"])
            _mu_tan    = float(frontier_df.loc[max_sharpe_idx, "Expected Return"])
            if _sigma_tan > 0:
                _x_line = np.linspace(0.0, float(frontier_df["Risk"].max()) * 1.1, 50)
                _slope_cml = (_mu_tan - rf_annual) / _sigma_tan
                _fig_frontier_report_opt.add_trace(go.Scatter(
                    x=_x_line, y=rf_annual + _slope_cml * _x_line, mode="lines",
                    name=f"CML (rf={_rf_label}: {rf_annual:.2%})",
                    line=dict(dash="dash", color="black"),
                ))
                _fig_frontier_report_opt.add_trace(go.Scatter(
                    x=[0.0], y=[rf_annual], mode="markers+text",
                    name="Ativo Livre de Risco",
                    text=[_rf_ticker], textposition="top center",
                    marker=dict(size=10, symbol="circle", color="black"),
                ))
        _fig_frontier_report_opt.update_layout(**_yc_layout(
            xaxis_title=f"Risco ({risk_measure})",
            yaxis_title="Retorno Esperado",
        ))

        st.session_state["opt_report_series"]  = series_dict
        st.session_state["opt_report_returns"] = returns
        st.session_state["opt_weights_atual"]  = selected_weights_final.to_dict()
        st.session_state["opt_weights_sug"]    = None
        st.session_state["opt_benchmark"]      = benchmark_choice if not returns_benchmark.empty else None
        st.session_state["opt_frontier_fig"]   = _fig_frontier_report_opt

        _render_report_ui(
            "opt",
            textos_padrao=_TEXTOS_PADRAO_OPT,
            fill_fronteira=_preencher_analise_fronteira_opt,
            fill_metricas=_preencher_analise_metricas_opt,
            show_fronteira=True,
        )

    except Exception as e:

         st.error("An unexpected error occurred during the optimization process.")
        #st.text(f"Technical details: {e}")
        
   
