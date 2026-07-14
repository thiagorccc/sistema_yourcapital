"""
Atualiza as planilhas de índices ANBIMA usadas pelo sistema a partir dos
arquivos "-HISTORICO.xls" baixados manualmente do ANBIMA Data
(data.anbima.com.br/indices) e colocados na pasta Anbima/.

Fluxo:
    1. Baixe manualmente o histórico de cada índice desejado em
       data.anbima.com.br/indices e salve o .xls em Anbima/ (o nome do
       arquivo pode ficar como o site exporta, ex.: "IMAB-HISTORICO.xls").
    2. Rode este script:  python atualizar_indices_anbima.py
    3. Ele lê cada .xls, extrai só "Data de Referência" e "Variação Diária
       (%)" (renomeando para "Data"/"Retorno" — mesmo formato já usado pelo
       sistema em Assets.py), e salva a planilha limpa na raiz do projeto.

Os 6 arquivos já consumidos pelo sistema (IMA-B, IMA-B 5, IMA-B 5+,
IDA-IPCA, IDKA 5A, IHFA) são sobrescritos com o nome exato que Assets.py
espera. Os demais índices da pasta (ainda não usados no sistema) também são
normalizados e salvos na raiz, prontos para quando forem integrados.

IFIX é tratado à parte: o export colocado em Anbima/ (formato diferente,
tipo Refinitiv — coluna de preço em vez de retorno %) vira a base histórica
de IFIX.xlsx (Data/Close, mesmo formato que Assets.py já espera). Como o
Yahoo Finance tem histórico bem curto para o IFIX.SA, ele só é usado para
completar os dias entre a última data do export e hoje — não para
histórico profundo. Rode o script de novo sempre que quiser avançar essas
cotações, mesmo sem um novo export manual na pasta.

Script standalone — não é importado pelo Streamlit.
"""

import os
import re
import sys
import glob
import datetime as dt

import pandas as pd

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PASTA_ANBIMA = os.path.join(_BASE_DIR, "Anbima")

_IFIX_ARQUIVO = "IFIX.xlsx"
_IFIX_TICKER_YF = "IFIX.SA"
_IFIX_NOME_RE = re.compile(r"^IFIX(?:\s*\(\d+\))?\.xlsx?$", re.IGNORECASE)

# Código do arquivo (prefixo antes de "-HISTORICO") -> nome do arquivo de
# saída na raiz do projeto. Os 6 primeiros são os nomes que Assets.py já
# espera hoje — não mudar sem atualizar Assets.py junto.
CODIGO_PARA_ARQUIVO = {
    "IMAB": "IMA-B.xlsx",
    "IMAB5": "IMA-B 5.xlsx",
    "IMAB5MAIS": "IMA-B 5+.xlsx",
    "IDAIPCA": "IDA-IPCA.xlsx",
    "IDKAPRE5A": "IDKA 5A.xlsx",
    "IHFA": "IHFA.xlsx",

    # Demais índices baixados, ainda não consumidos pelo sistema —
    # normalizados e deixados prontos na raiz para uso futuro.
    "IDADI": "IDA-DI.xlsx",
    "IDAGERAL": "IDA-Geral.xlsx",
    "IDAIPCAINFRAESTRUTURA": "IDA-IPCA Infraestrutura.xlsx",
    "IDAIPCAEXINFRAESTRUTURA": "IDA-IPCA ex-Infraestrutura.xlsx",
    "IDALIQDI": "IDA-LIQ DI.xlsx",
    "IDALIQGERAL": "IDA-LIQ Geral.xlsx",
    "IDALIQIPCA": "IDA-LIQ IPCA.xlsx",
    "IDALIQIPCAINFRAESTRUTURA": "IDA-LIQ IPCA Infraestrutura.xlsx",
    "IDKAPRE3M": "IDKA 3M.xlsx",
    "IDKAPRE1A": "IDKA 1A.xlsx",
    "IDKAPRE2A": "IDKA 2A.xlsx",
    "IDKAPRE3A": "IDKA 3A.xlsx",
    "IDKAIPCA2A": "IDKA IPCA 2A.xlsx",
    "IDKAIPCA3A": "IDKA IPCA 3A.xlsx",
    "IDKAIPCA5A": "IDKA IPCA 5A.xlsx",
    "IDKAIPCA10A": "IDKA IPCA 10A.xlsx",
    "IDKAIPCA15A": "IDKA IPCA 15A.xlsx",
    "IDKAIPCA20A": "IDKA IPCA 20A.xlsx",
    "IDKAIPCA30A": "IDKA IPCA 30A.xlsx",
    "IMAB5P2": "IMA-B 5 P2.xlsx",
    "IMAGERAL": "IMA-Geral.xlsx",
    "IMAGERALEXC": "IMA-Geral ex-C.xlsx",
    "IMAS": "IMA-S.xlsx",
    "IRFM": "IRF-M.xlsx",
    "IRFM1": "IRF-M 1.xlsx",
    "IRFM1MAIS": "IRF-M 1+.xlsx",
    "IRFMP2": "IRF-M P2.xlsx",
    "IRFMP3": "IRF-M P3.xlsx",
}

_NOME_ARQUIVO_RE = re.compile(r"^([A-Z0-9]+)-HISTORICO(?:\s*\(\d+\))?\.xlsx?$", re.IGNORECASE)


def _codigo_do_arquivo(nome_arquivo):
    m = _NOME_ARQUIVO_RE.match(nome_arquivo)
    return m.group(1).upper() if m else None


def _agrupar_por_codigo(pasta):
    """Lista os .xls/.xlsx de `pasta` agrupados por código; se houver mais de
    um arquivo para o mesmo código (ex.: 'IHFA-HISTORICO (2).xls' baixado
    depois de 'IHFA-HISTORICO.xls'), fica só o modificado mais recentemente."""
    candidatos = {}
    for path in glob.glob(os.path.join(pasta, "*.xls")) + glob.glob(os.path.join(pasta, "*.xlsx")):
        nome = os.path.basename(path)
        if _IFIX_NOME_RE.match(nome):
            continue  # IFIX é tratado à parte em _processar_ifix
        codigo = _codigo_do_arquivo(nome)
        if codigo is None:
            print(f"[aviso] não reconheci o padrão de nome de '{nome}', pulando.", file=sys.stderr)
            continue
        atual = candidatos.get(codigo)
        if atual is None or os.path.getmtime(path) > os.path.getmtime(atual):
            candidatos[codigo] = path
    return candidatos


def _normalizar(path):
    df = pd.read_excel(path)
    df = df[["Data de Referência", "Variação Diária (%)"]].rename(
        columns={"Data de Referência": "Data", "Variação Diária (%)": "Retorno"}
    )
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
    df = df.dropna(subset=["Data"])
    df = df.sort_values("Data")
    df = df[~df["Data"].duplicated(keep="last")]
    return df.reset_index(drop=True)


def _localizar_ifix(pasta):
    """Acha o export de IFIX em Anbima/ (nome livre, ex.: 'IFIX (1).xlsx'),
    pegando o mais recente se houver mais de um."""
    candidatos = [
        p for p in glob.glob(os.path.join(pasta, "*.xls*"))
        if _IFIX_NOME_RE.match(os.path.basename(p))
    ]
    if not candidatos:
        return None
    return max(candidatos, key=os.path.getmtime)


def _normalizar_ifix(path):
    """Export tipo Refinitiv/LSEG: 'Date' + uma coluna de preço (ex.:
    '.IFIX (TRDPRC_1)'), mais recente primeiro, com uma linha de subtítulo
    ('Close') que vaza para dentro dos dados. Normaliza para Data/Close,
    ordem cronológica, mesmo formato que Assets.py espera para IFIX.xlsx."""
    df = pd.read_excel(path)
    col_preco = [c for c in df.columns if c != "Date"][0]
    df = df.rename(columns={"Date": "Data", col_preco: "Close"})
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Data", "Close"])
    df = df.sort_values("Data")
    df = df[~df["Data"].duplicated(keep="last")]
    return df[["Data", "Close"]].reset_index(drop=True)


def _completar_ifix_com_yfinance(df, hoje):
    """Preenche com o Yahoo Finance (IFIX.SA) os dias entre a última data do
    export da ANBIMA e hoje — o Yahoo tem histórico curto para o IFIX, mas
    serve bem para ir emendando as cotações mais recentes entre um
    download manual e outro."""
    import yfinance as yf

    ultima_data = df["Data"].max().date() if not df.empty else None
    inicio = (ultima_data + dt.timedelta(days=1)) if ultima_data else dt.date(2000, 1, 1)
    if inicio > hoje:
        return df, 0

    try:
        yf_df = yf.download(
            _IFIX_TICKER_YF, start=inicio, end=hoje + dt.timedelta(days=1),
            progress=False, auto_adjust=True,
        )
    except Exception as e:
        print(f"[IFIX] erro ao buscar Yahoo Finance: {e}", file=sys.stderr)
        return df, 0

    if yf_df.empty:
        return df, 0

    fechamentos = yf_df["Close"]
    if hasattr(fechamentos, "columns"):  # MultiIndex de colunas (1 ticker só)
        fechamentos = fechamentos[_IFIX_TICKER_YF]

    novas_linhas = pd.DataFrame({
        "Data": pd.to_datetime(fechamentos.index.date),
        "Close": fechamentos.values,
    })

    df_novo = pd.concat([df, novas_linhas], ignore_index=True)
    df_novo = df_novo.sort_values("Data")
    df_novo = df_novo[~df_novo["Data"].duplicated(keep="last")]
    return df_novo.reset_index(drop=True), len(novas_linhas)


def _processar_ifix(hoje):
    destino = os.path.join(_BASE_DIR, _IFIX_ARQUIVO)
    path_fonte = _localizar_ifix(_PASTA_ANBIMA)

    if path_fonte is not None:
        try:
            df = _normalizar_ifix(path_fonte)
        except Exception as e:
            print(f"[IFIX] erro ao processar '{os.path.basename(path_fonte)}': {e}", file=sys.stderr)
            return
        origem = os.path.basename(path_fonte)
    elif os.path.exists(destino):
        df = pd.read_excel(destino)
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
        origem = _IFIX_ARQUIVO
    else:
        print("[IFIX] nenhum export na pasta Anbima/ e nenhum IFIX.xlsx existente, pulando.")
        return

    df, n_novas_yf = _completar_ifix_com_yfinance(df, hoje)

    try:
        df.to_excel(destino, index=False)
    except PermissionError:
        print(
            f"[IFIX] não foi possível salvar '{_IFIX_ARQUIVO}' — feche o arquivo "
            f"no Excel e rode o script de novo.",
            file=sys.stderr,
        )
        return

    print(
        f"[IFIX] {origem} -> {_IFIX_ARQUIVO}: {len(df)} linhas, "
        f"{df['Data'].min():%d/%m/%Y} a {df['Data'].max():%d/%m/%Y} "
        f"({n_novas_yf} dia(s) completado(s) via Yahoo Finance)"
    )


def main():
    if not os.path.isdir(_PASTA_ANBIMA):
        print(f"Pasta '{_PASTA_ANBIMA}' não encontrada.", file=sys.stderr)
        sys.exit(1)

    hoje = dt.date.today()
    _processar_ifix(hoje)

    candidatos = _agrupar_por_codigo(_PASTA_ANBIMA)
    if not candidatos:
        print(f"Nenhum arquivo '*-HISTORICO.xls' encontrado em '{_PASTA_ANBIMA}'.")
        return

    for codigo, path in sorted(candidatos.items()):
        nome_saida = CODIGO_PARA_ARQUIVO.get(codigo, f"{codigo}.xlsx")
        try:
            df = _normalizar(path)
        except Exception as e:
            print(f"[{codigo}] erro ao processar '{os.path.basename(path)}': {e}", file=sys.stderr)
            continue

        if df.empty:
            print(f"[{codigo}] '{os.path.basename(path)}' não tem linhas válidas, pulando.")
            continue

        destino = os.path.join(_BASE_DIR, nome_saida)
        try:
            df.to_excel(destino, index=False)
        except PermissionError:
            print(
                f"[{codigo}] não foi possível salvar '{nome_saida}' — feche o arquivo "
                f"no Excel e rode o script de novo.",
                file=sys.stderr,
            )
            continue

        print(
            f"[{codigo}] {os.path.basename(path)} -> {nome_saida}: "
            f"{len(df)} linhas, {df['Data'].min():%d/%m/%Y} a {df['Data'].max():%d/%m/%Y}"
        )


if __name__ == "__main__":
    main()
