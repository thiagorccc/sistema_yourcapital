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
from comparador import _tabela_rentabilidade_mensal, _calcular_metricas, _exibir_metricas, _compute_synthetic_returns, _get_rf_rate, _render_synth_expander, _render_constraints_expander, _build_opt_constraints


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
        logo = Image.open("Logo Oikos Horizontal Colorido.png")
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
        frontier_df['Sharpe'] = frontier_df['Expected Return'] / frontier_df['Risk']
        frontier_df = frontier_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Sharpe'])

        min_vol_idx = frontier_df['Risk'].idxmin()
        max_ret_idx = frontier_df['Expected Return'].idxmax()
        max_sharpe_idx = frontier_df['Sharpe'].idxmax()

        # --- Plot Fronteira ---
        st.subheader("Fronteira Eficiente")

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

        fig_frontier.update_layout(
            title=f"Efficient Frontier with Key Portfolios Highlighted ({risk_measure})",
            xaxis_title=f"Portfolio Risk ({risk_measure})",
            yaxis_title="Portfolio Expected Return",
            template="plotly_white"
        )

        st.plotly_chart(fig_frontier, use_container_width=True)

        # --- Escolha da carteira para análise ---
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
  
        st.subheader("Composição do Portfólio")

        composition_df = pd.DataFrame(selected_weights)
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


    except Exception as e:

         st.error("An unexpected error occurred during the optimization process.")
        #st.text(f"Technical details: {e}")
        
   
