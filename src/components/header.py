import base64

import streamlit as st


def get_base64_image(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()


def header_home():

    logo_base64 = get_base64_image("assets/logo/logo.png")
    logo_url = f"data:image/png;base64,{logo_base64}"

    st.markdown(f"""
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; margin-bottom:30px; margin-top:30px">
            <img src='{logo_url}' style='height:100px;' />
            <h1 style='text-align:center; color:#E0E3FF'>SNAP<br/>CLASS</h1>
        </div>
    """, unsafe_allow_html=True)


def header_dashboard():

    logo_base64 = get_base64_image("assets/logo/logo.png")
    logo_url = f"data:image/png;base64,{logo_base64}"

    st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:center; gap:10px">
            <img src='{logo_url}' style='height:85px;' />
            <h2 style='text-align:left; color:#5865F2'>SNAP<br/>CLASS</h2>
        </div>
    """, unsafe_allow_html=True)