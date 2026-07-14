

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import yfinance as yf


# -----------------------------------------------------------------------------
# Funções auxiliares
# -----------------------------------------------------------------------------

def baixar_dados_yahoo(ticker, data_inicial, data_final):
    """Baixa dados do Yahoo Finance e ajusta MultiIndex, caso o yfinance retorne colunas agrupadas."""
    dados = yf.download(
        ticker,
        start=data_inicial,
        end=data_final,
        auto_adjust=False,
        progress=False,
    )

    if dados.empty:
        return pd.DataFrame()

    if isinstance(dados.columns, pd.MultiIndex):
        dados = dados.droplevel(1, axis=1)

    dados = dados.dropna()
    return dados


def criar_candlestick(dados, titulo, nome_serie="", yaxis_titulo="Pontos", tickformat=",.0f"):
    """Cria um gráfico candlestick em Plotly a partir de um DataFrame OHLC."""
    if dados.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Não foi possível carregar os dados para: {titulo}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=dados.index,
                open=dados["Open"],
                high=dados["High"],
                low=dados["Low"],
                close=dados["Close"],
                name=nome_serie or titulo,
            )
        ]
    )

    fig.update_layout(
        title=titulo,
        xaxis_title="Data",
        yaxis_title=yaxis_titulo,
        template="plotly_white",
        height=500,
        margin=dict(l=40, r=25, t=55, b=35),
        xaxis_rangeslider_visible=False,
        font=dict(family="Raleway", size=12),
    )

    fig.update_yaxes(tickformat=tickformat)

    return fig


# -----------------------------------------------------------------------------
# Gráficos do Morning Call
# -----------------------------------------------------------------------------

def grafico_ibovespa_2026():
    """Candlestick do Ibovespa desde o início de 2026."""
    data_inicial = dt.datetime(2026, 1, 1)
    data_final = dt.datetime.today()

    ibov = baixar_dados_yahoo("^BVSP", data_inicial, data_final)
    return criar_candlestick(ibov, "Ibovespa em 2026", "Ibovespa")


def grafico_ibovespa_maximo_historico():
    """Candlestick do Ibovespa em janela mais longa para análise de máximo histórico."""
    data_inicial = dt.datetime(2020, 1, 1)
    data_final = dt.datetime.today()

    ibov = baixar_dados_yahoo("^BVSP", data_inicial, data_final)
    return criar_candlestick(ibov, "Ibovespa - Máximo Histórico", "Ibovespa")


def grafico_usdbrl_2026():
    """Candlestick do dólar (USD/BRL) desde o início de 2026."""
    data_inicial = dt.datetime(2026, 1, 1)
    data_final = dt.datetime.today()

    usdbrl = baixar_dados_yahoo("USDBRL=X", data_inicial, data_final)
    return criar_candlestick(
        usdbrl,
        "Dólar (USD/BRL) em 2026",
        "USD/BRL",
        yaxis_titulo="R$ por USD",
        tickformat=",.3f",
    )


def grafico_usdbrl_historico():
    """Candlestick do dólar (USD/BRL) desde 2020 para análise histórica."""
    data_inicial = dt.datetime(2020, 1, 1)
    data_final = dt.datetime.today()

    usdbrl = baixar_dados_yahoo("USDBRL=X", data_inicial, data_final)
    return criar_candlestick(
        usdbrl,
        "Dólar (USD/BRL) - Histórico",
        "USD/BRL",
        yaxis_titulo="R$ por USD",
        tickformat=",.3f",
    )


def grafico_sp500_2026():
    """Candlestick do S&P 500 desde o início de 2026."""
    data_inicial = dt.datetime(2026, 1, 1)
    data_final = dt.datetime.today()

    sp500 = baixar_dados_yahoo("^GSPC", data_inicial, data_final)
    return criar_candlestick(sp500, "S&P 500 em 2026", "S&P 500")


def grafico_sp500_historico():
    """Candlestick do S&P 500 desde 2020 para análise histórica."""
    data_inicial = dt.datetime(2020, 1, 1)
    data_final = dt.datetime.today()

    sp500 = baixar_dados_yahoo("^GSPC", data_inicial, data_final)
    return criar_candlestick(sp500, "S&P 500 - Histórico", "S&P 500")


def grafico_ouro_2026():
    """Candlestick do ouro (futuros) desde o início de 2026."""
    data_inicial = dt.datetime(2026, 1, 1)
    data_final = dt.datetime.today()

    ouro = baixar_dados_yahoo("GC=F", data_inicial, data_final)
    return criar_candlestick(
        ouro,
        "Ouro em 2026",
        "Ouro",
        yaxis_titulo="USD por onça",
        tickformat=",.2f",
    )


def grafico_ouro_historico():
    """Candlestick do ouro (futuros) desde 2020 para análise histórica."""
    data_inicial = dt.datetime(2020, 1, 1)
    data_final = dt.datetime.today()

    ouro = baixar_dados_yahoo("GC=F", data_inicial, data_final)
    return criar_candlestick(
        ouro,
        "Ouro - Histórico",
        "Ouro",
        yaxis_titulo="USD por onça",
        tickformat=",.2f",
    )


def grafico_petroleo_2026():
    """Candlestick do petróleo WTI (futuros) desde o início de 2026."""
    data_inicial = dt.datetime(2026, 1, 1)
    data_final = dt.datetime.today()

    petroleo = baixar_dados_yahoo("CL=F", data_inicial, data_final)
    return criar_candlestick(
        petroleo,
        "Petróleo WTI em 2026",
        "WTI",
        yaxis_titulo="USD por barril",
        tickformat=",.2f",
    )


def grafico_petroleo_historico():
    """Candlestick do petróleo WTI (futuros) desde 2020 para análise histórica."""
    data_inicial = dt.datetime(2020, 1, 1)
    data_final = dt.datetime.today()

    petroleo = baixar_dados_yahoo("CL=F", data_inicial, data_final)
    return criar_candlestick(
        petroleo,
        "Petróleo WTI - Histórico",
        "WTI",
        yaxis_titulo="USD por barril",
        tickformat=",.2f",
    )


# -----------------------------------------------------------------------------
# Painel de mercado
# -----------------------------------------------------------------------------

def grafico_painel_mercado(data_inicio_semana=None, data_fim_semana=None):
    """Dashboard vertical: Ibovespa candlestick do ano, maiores altas/baixas da semana,
    índices globais, câmbio e commodities.

    data_inicio_semana: primeiro dia útil da semana de referência (datetime.date ou datetime.datetime)
    data_fim_semana: último dia útil da semana de referência (datetime.date ou datetime.datetime)
    """
    import glob
    import os
    import re
    from plotly.subplots import make_subplots

    hoje = dt.datetime.today()
    inicio_ano = dt.datetime(hoje.year, 1, 1)

    # --- Semana de referência ---
    # Padrão: última semana completa (seg–sex) em relação a hoje
    _hoje_date = hoje.date()
    _dia = _hoje_date.weekday()                       # 0=seg, 6=dom
    _dias_desde_sexta = (_dia - 4) % 7 or 7          # dias corridos desde a última sexta
    _default_fim = _hoje_date - dt.timedelta(days=_dias_desde_sexta)
    _default_ini = _default_fim - dt.timedelta(days=4)

    # Normaliza data_fim_semana para datetime.date
    if data_fim_semana is None:
        data_fim_semana = _default_fim
    elif isinstance(data_fim_semana, dt.datetime):
        data_fim_semana = data_fim_semana.date()

    # Normaliza data_inicio_semana para datetime.date
    if data_inicio_semana is None:
        data_inicio_semana = _default_ini
    elif isinstance(data_inicio_semana, dt.datetime):
        data_inicio_semana = data_inicio_semana.date()

    # --- Ibovespa OHLC ---
    ibov = baixar_dados_yahoo("^BVSP", inicio_ano, data_fim_semana + dt.timedelta(days=1))

    # Fechamento no fim da semana de referência
    ibov_semana_df = ibov[ibov.index.date <= data_fim_semana]
    ibov_ultimo = float(ibov_semana_df["Close"].iloc[-1]) if not ibov_semana_df.empty else 0.0

    # Preço de referência: último fechamento antes do início da semana
    ibov_antes_df = ibov[ibov.index.date < data_inicio_semana]
    ibov_ref = float(ibov_antes_df["Close"].iloc[-1]) if not ibov_antes_df.empty else ibov_ultimo
    ibov_var_semana = (ibov_ultimo / ibov_ref - 1) * 100 if ibov_ref else 0.0

    # --- Download único de todos os ativos globais ---
    # Um único request garante que todos partilhem o mesmo calendário de datas.
    # O reindex+ffill preenche feriados locais com o fechamento mais recente,
    # de modo que todos os ativos ficam alinhados na mesma data de referência.
    _TICKERS = [
        "^GSPC", "^IXIC", "^STOXX", "^N225",
        "USDBRL=X", "EURBRL=X", "^TNX",
        "BZ=F", "CL=F", "TIO=F", "BTC-USD",
    ]
    _inicio_dl = data_inicio_semana - dt.timedelta(days=7)
    _fim_dl    = data_fim_semana    + dt.timedelta(days=1)

    try:
        _df_g = yf.download(_TICKERS, start=_inicio_dl, end=_fim_dl,
                            progress=False, auto_adjust=True)
        _closes = _df_g["Close"] if isinstance(_df_g.columns, pd.MultiIndex) else _df_g
        # Alinha todos na mesma grade de dias úteis e preenche lacunas de feriados
        _closes = _closes.reindex(
            pd.bdate_range(_inicio_dl, data_fim_semana)
        ).ffill()
    except Exception:
        _closes = pd.DataFrame()

    def _p(ticker):
        if _closes.empty or ticker not in _closes.columns:
            return None
        s = _closes[ticker][_closes.index.date <= data_fim_semana].dropna()
        return float(s.iloc[-1]) if not s.empty else None

    def _v(ticker):
        if _closes.empty or ticker not in _closes.columns:
            return None
        s_fim = _closes[ticker][_closes.index.date <= data_fim_semana].dropna()
        s_ini = _closes[ticker][_closes.index.date <  data_inicio_semana].dropna()
        if s_fim.empty or s_ini.empty:
            return None
        p_fim = float(s_fim.iloc[-1])
        p_ini = float(s_ini.iloc[-1])
        return (p_fim / p_ini - 1) * 100 if p_ini else None

    # --- Extração por ativo ---
    sp500,    var_sp500    = _p("^GSPC"),    _v("^GSPC")
    nasdaq,   var_nasdaq   = _p("^IXIC"),    _v("^IXIC")
    stoxx,    var_stoxx    = _p("^STOXX"),   _v("^STOXX")
    nikkei,   var_nikkei   = _p("^N225"),    _v("^N225")
    usd,      var_usd      = _p("USDBRL=X"), _v("USDBRL=X")
    eur,      var_eur      = _p("EURBRL=X"), _v("EURBRL=X")
    yield10y, var_yield10y = _p("^TNX"),     _v("^TNX")
    brent,    var_brent    = _p("BZ=F"),     _v("BZ=F")
    wti,      var_wti      = _p("CL=F"),     _v("CL=F")
    mferro,   var_mferro   = _p("TIO=F"),    _v("TIO=F")
    btc,      var_btc      = _p("BTC-USD"),  _v("BTC-USD")

    # --- Composição do Ibovespa ---
    # Lê o CSV mais recente (padrão IBOV*.csv) do diretório do app.
    # Atualiza o arquivo para refletir a carteira vigente.
    def _ler_ibov_stocks():
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_files  = sorted(glob.glob(os.path.join(script_dir, "IBOV*.csv")), reverse=True)
        if csv_files:
            try:
                df  = pd.read_csv(csv_files[0], sep=";", encoding="latin1", skiprows=1)
                col = df.columns[0]
                ok  = df[col].dropna().apply(
                    lambda x: bool(re.match(r"^[A-Z0-9]{4,6}$", str(x).strip()))
                )
                tickers = [str(x).strip() + ".SA" for x in df.loc[ok, col]]
                if tickers:
                    return tickers
            except Exception:
                pass
        # Fallback: carteira do Ibovespa de 12/05/2026
        return [
            "ALOS3.SA","ABEV3.SA","ASAI3.SA","AURE3.SA","AXIA3.SA","AXIA6.SA",
            "AZZA3.SA","B3SA3.SA","BBSE3.SA","BBDC3.SA","BBDC4.SA","BRAP4.SA",
            "BBAS3.SA","BRKM5.SA","BRAV3.SA","BPAC11.SA","CXSE3.SA","CEAB3.SA",
            "CMIG4.SA","COGN3.SA","CSMG3.SA","CPLE3.SA","CSAN3.SA","CPFE3.SA",
            "CMIN3.SA","CURY3.SA","CYRE3.SA","DIRR3.SA","EMBJ3.SA","ENGI11.SA",
            "ENEV3.SA","EGIE3.SA","EQTL3.SA","FLRY3.SA","GGBR4.SA","GOAU4.SA",
            "HAPV3.SA","HYPE3.SA","IGTI11.SA","ISAE4.SA","ITSA4.SA","ITUB4.SA",
            "KLBN11.SA","RENT3.SA","LREN3.SA","MGLU3.SA","POMO4.SA","MBRF3.SA",
            "BEEF3.SA","MOTV3.SA","MRVE3.SA","MULT3.SA","NATU3.SA","PETR3.SA",
            "PETR4.SA","RECV3.SA","PSSA3.SA","PRIO3.SA","RADL3.SA","RDOR3.SA",
            "RAIL3.SA","SBSP3.SA","SANB11.SA","CSNA3.SA","SLCE3.SA","SMFT3.SA",
            "SUZB3.SA","TAEE11.SA","VIVT3.SA","TIMS3.SA","TOTS3.SA","UGPA3.SA",
            "USIM5.SA","VALE3.SA","VAMO3.SA","VBBR3.SA","VIVA3.SA","WEGE3.SA",
            "YDUQ3.SA",
        ]

    IBOV_STOCKS = _ler_ibov_stocks()

    # --- Maiores altas/baixas da semana ---
    top_altas      = pd.Series(dtype=float)
    top_baixas     = pd.Series(dtype=float)
    ultimos_precos = pd.Series(dtype=float)

    try:
        # Baixa a partir de alguns dias antes para capturar o preço de referência (sexta anterior)
        inicio_download = data_inicio_semana - dt.timedelta(days=7)
        df_sem = yf.download(
            IBOV_STOCKS,
            start=inicio_download,
            end=data_fim_semana + dt.timedelta(days=1),
            progress=False, auto_adjust=True,
        )
        precos = df_sem["Close"] if isinstance(df_sem.columns, pd.MultiIndex) else df_sem
        precos = precos.dropna(how="all")

        # Preço de referência: último fechamento antes do início da semana
        precos_ref = precos[precos.index.date < data_inicio_semana]
        # Preços da semana de referência
        precos_sem = precos[precos.index.date <= data_fim_semana]

        if not precos_ref.empty and not precos_sem.empty:
            ref = precos_ref.iloc[-1]
            fim = precos_sem.iloc[-1]
            retornos = (fim / ref - 1) * 100
            retornos = retornos.dropna().sort_values()
            top_baixas     = retornos.head(5)
            top_altas      = retornos.tail(5).iloc[::-1]
            ultimos_precos = precos_sem.iloc[-1]
    except Exception:
        pass

    # --- Formatadores (padrão brasileiro) ---
    def fmt(v, decimals=2, prefix="", suffix=""):
        if v is None or (isinstance(v, float) and v != v):
            return "N/D"
        if abs(v) >= 1_000:
            return f"{prefix}{v:,.0f}{suffix}".replace(",", ".")
        return f"{prefix}{v:.{decimals}f}{suffix}".replace(".", ",")

    def fmt_var(v):
        if v is None or (isinstance(v, float) and v != v):
            return "—"
        return f"{v:+.2f}%".replace(".", ",")

    VERDE    = "#16A34A"
    VERMELHO = "#DC2626"
    CINZA    = "#374151"
    PRETO    = "#1E293B"

    def cor_var(v):
        return VERDE if (v is not None and v == v and v >= 0) else VERMELHO

    # --- Dados das tabelas ---
    # Coluna esquerda (linha 2): índices globais + câmbio
    nomes_esq = ["S&P 500","NASDAQ","STOXX","NIKKEI","","USD","EUR","YIELD 10Y"]
    precos_esq = [
        fmt(sp500, 0), fmt(nasdaq, 0), fmt(stoxx, 0), fmt(nikkei, 0),
        "",
        fmt(usd, 2, "R$ "), fmt(eur, 2, "R$ "),
        fmt(yield10y, 3, suffix="%"),
    ]
    vars_esq = [
        fmt_var(var_sp500), fmt_var(var_nasdaq), fmt_var(var_stoxx), fmt_var(var_nikkei),
        "",
        fmt_var(var_usd), fmt_var(var_eur), fmt_var(var_yield10y),
    ]
    cores_esq = [
        cor_var(var_sp500), cor_var(var_nasdaq), cor_var(var_stoxx), cor_var(var_nikkei),
        "#FFFFFF",
        cor_var(var_usd), cor_var(var_eur), cor_var(var_yield10y),
    ]

    # Coluna direita (linha 2): commodities + Bitcoin
    nomes_dir  = ["BRENT","WTI","M. FERRO","BITCOIN"]
    precos_dir = [
        fmt(brent, 2, "$ "), fmt(wti, 2, "$ "),
        fmt(mferro, 2, "$ "), fmt(btc, 0, "$ "),
    ]
    vars_dir  = [fmt_var(var_brent), fmt_var(var_wti), fmt_var(var_mferro), fmt_var(var_btc)]
    cores_dir = [cor_var(var_brent), cor_var(var_wti), cor_var(var_mferro), cor_var(var_btc)]

    # Altas e baixas (linha 3)
    def linha_tabela(serie):
        if serie.empty:
            return ["—"] * 5, ["—"] * 5, ["—"] * 5
        tickers   = [t.replace(".SA", "") for t in serie.index]
        preco_lst = [
            fmt(float(ultimos_precos[t]) if t in ultimos_precos.index else None, 2, "R$ ")
            for t in serie.index
        ]
        var_lst = [fmt_var(float(v)) for v in serie.values]
        return tickers, preco_lst, var_lst

    a_tick, a_prec, a_var = linha_tabela(top_altas)
    b_tick, b_prec, b_var = linha_tabela(top_baixas)

    # --- Subplots: layout vertical (3 linhas × 2 colunas) ---
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"colspan": 2, "type": "xy"}, None],
            [{"type": "table"},            {"type": "table"}],
            [{"type": "table"},            {"type": "table"}],
        ],
        row_heights=[0.52, 0.20, 0.28],
        column_widths=[0.50, 0.50],
        vertical_spacing=0.04,
        horizontal_spacing=0.04,
    )

    # Candlestick Ibovespa (linha 1, largura total)
    if not ibov.empty:
        fig.add_trace(
            go.Candlestick(
                x=ibov.index,
                open=ibov["Open"], high=ibov["High"],
                low=ibov["Low"],   close=ibov["Close"],
                name="Ibovespa",
                increasing_line_color="#2563EB",
                decreasing_line_color="#DC2626",
            ),
            row=1, col=1,
        )

    # Helper para go.Table uniforme
    def mk_table(header_labels, cell_values, cell_colors, columnwidth=None):
        tem_header = any(h for h in header_labels)
        kwargs = dict(columnwidth=columnwidth) if columnwidth else {}
        return go.Table(
            header=dict(
                values=[f"<b>{h}</b>" if h else "" for h in header_labels],
                fill_color="white", line_color="white",
                font=dict(family="Raleway", size=13, color=PRETO),
                align="center",
                height=28 if tem_header else 2,
            ),
            cells=dict(
                values=cell_values,
                fill_color="white", line_color="white",
                align=["left", "right", "right"],
                font=dict(family="Raleway", size=13, color=cell_colors),
                height=26,
            ),
            **kwargs,
        )

    ne = len(nomes_esq)
    nd = len(nomes_dir)

    # Linha 2: índices/câmbio (esq) | commodities (dir)
    fig.add_trace(
        mk_table(["", "", ""],
                 [nomes_esq, precos_esq, vars_esq],
                 [[CINZA] * ne, [CINZA] * ne, cores_esq]),
        row=2, col=1,
    )
    fig.add_trace(
        mk_table(["", "", ""],
                 [nomes_dir, precos_dir, vars_dir],
                 [[CINZA] * nd, [CINZA] * nd, cores_dir]),
        row=2, col=2,
    )

    # Linha 3: maiores altas (esq) | maiores baixas (dir)
    fig.add_trace(
        mk_table(["", "MAIORES ALTAS", ""],
                 [a_tick, a_prec, a_var],
                 [[PRETO] * 5, [PRETO] * 5, [VERDE] * 5],
                 columnwidth=[2, 4, 2]),
        row=3, col=1,
    )
    fig.add_trace(
        mk_table(["", "MAIORES BAIXAS", ""],
                 [b_tick, b_prec, b_var],
                 [[PRETO] * 5, [PRETO] * 5, [VERMELHO] * 5],
                 columnwidth=[2, 4, 2]),
        row=3, col=2,
    )

    # --- Layout ---
    cor_titulo  = VERMELHO if ibov_var_semana < 0 else VERDE
    sinal       = "+" if ibov_var_semana >= 0 else ""
    ibov_str    = f"{ibov_ultimo:,.0f}".replace(",", ".")
    var_str     = f"{sinal}{ibov_var_semana:.2f}%".replace(".", ",")
    semana_str  = f"{data_inicio_semana.strftime('%d/%m')}–{data_fim_semana.strftime('%d/%m/%Y')}"

    fig.update_layout(
        title=dict(
            text=(
                f"<b>IBOVESPA</b>  "
                f"<span style='font-size:14px; color:{CINZA}'>{semana_str}</span><br>"
                f"<span style='font-size:26px; color:{PRETO}'>{ibov_str}  "
                f"<span style='color:{cor_titulo}'>{var_str}</span></span>"
            ),
            font=dict(family="Raleway", size=34, color=PRETO),
            x=0.01, xanchor="left",
        ),
        width=800,
        height=1050,
        paper_bgcolor="white",
        plot_bgcolor="white",
        template="plotly_white",
        showlegend=False,
        margin=dict(l=20, r=20, t=115, b=10),
        font=dict(family="Raleway"),
    )

    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_yaxes(tickformat=",.0f", row=1, col=1)

    return fig


# -----------------------------------------------------------------------------
# Função lida pelo morningcall.py
# -----------------------------------------------------------------------------

def gerar_graficos(data_inicio_semana=None, data_fim_semana=None):
    """
    Retorna os gráficos disponíveis para o Morning Call.

    O nome à esquerda aparece no selectbox do morningcall.py.
    O valor à direita pode ser uma figura Plotly, Matplotlib ou um caminho .png/.jpg.
    """
    return {
        "Painel de Mercado": grafico_painel_mercado(
            data_inicio_semana=data_inicio_semana,
            data_fim_semana=data_fim_semana,
        ),
        "Ibovespa 2026": grafico_ibovespa_2026(),
        "Ibovespa - Máximo Histórico": grafico_ibovespa_maximo_historico(),
        "Dólar (USD/BRL) 2026": grafico_usdbrl_2026(),
        "Dólar (USD/BRL) - Histórico": grafico_usdbrl_historico(),
        "S&P 500 2026": grafico_sp500_2026(),
        "S&P 500 - Histórico": grafico_sp500_historico(),
        "Ouro 2026": grafico_ouro_2026(),
        "Ouro - Histórico": grafico_ouro_historico(),
        "Petróleo WTI 2026": grafico_petroleo_2026(),
        "Petróleo WTI - Histórico": grafico_petroleo_historico(),
    }