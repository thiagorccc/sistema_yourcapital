import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from Optimizer import show_optimizer
from morningcall import show_morningcall
from comparador import show_comparador
# from carteiras_oikos import show_carteiras
import streamlit.components.v1 as components

# Configurações da página
st.set_page_config(
    page_title='Your Capital',
    page_icon="icone_final.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Oculta menu lateral padrão do Streamlit
hide_default_sidebar = """
<style>
section[data-testid="stSidebarNav"] {
    display: none;
}
</style>
"""
st.markdown(hide_default_sidebar, unsafe_allow_html=True)

# Fundo bonito com imagem de fundo
page_bg_image = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url(' ');
background-size: cover;
}
[data-testid="stHeader"] {
background-color: rgba(0, 0, 0, 0);
}
[data-testid="stSidebar"] {
background-image: url('https://images.unsplash.com/photo-1695721780267-9ce4a448ef05?q=80&w=387&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');
background-size: cover;
}
</style>
"""

st.markdown(page_bg_image, unsafe_allow_html=True)

# Sidebar com menu principal
with st.sidebar:
    selected = option_menu(
        menu_title= None,
        options=["Home", "Otimizador de Portfólio", "Comparador", "Gerador de Relatório"],
        icons=["house", "bar-chart", "bar-chart", "file-earmark-ppt"],
        menu_icon="sunrise",
        default_index=0,
    )

# Página: HOME
if selected == "Home":
    logo = Image.open("yourcapital_img.jpg")
    st.image(logo, use_container_width=True)
   

elif selected == "Otimizador de Portfólio":
    show_optimizer()

elif selected == "Comparador":
    show_comparador()

# elif selected == "Carteiras Oikos":
#     show_carteiras()

elif selected == "Gerador de Relatório":
    show_morningcall()