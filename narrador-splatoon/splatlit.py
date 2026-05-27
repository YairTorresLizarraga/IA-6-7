import os
import uuid

import streamlit as st

from modulos.common import fps
from modulos.narracion_texto import generar_narracion
from modulos.narracion_voz import generar_tts_comentarios
from modulos.video import generar_video
from modulos.vision import analizar_video

st.set_page_config(page_title="splatlit", page_icon="🐙")

st.title("🐙 splatlit")
st.write("narrador de partidas de splatoon usando gemma4")

uploaded_file = st.file_uploader(
    "escoge un video (720p)",
    type=["mp4"],
    accept_multiple_files=False,
)

save_path = None

if uploaded_file is not None:
    file_extension = os.path.splitext(uploaded_file.name)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    save_path = os.path.join("temp", unique_filename)

    video_bytes = uploaded_file.read()

    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        with open(save_path, "wb") as f:
            f.write(video_bytes)
        st.session_state.last_uploaded = uploaded_file.name
        st.session_state.current_path = save_path

    st.success(f"video subido con exito: {save_path}")
    st.video(video_bytes)

resultados = None

if save_path is not None:
    frames_analizados = st.empty()
    frames_analizados.info("iniciando análisis de video...")
    last_i = 0
    def actualizar(i):
        global last_i
        last_i = i
        frames_analizados.info(f"frames analizados: {i}")

    resultados = analizar_video(save_path, actualizar)
    frames_analizados.success(f"finalizó análisis de video ({last_i})")

comentarios = None



if resultados is not None:
    comentarios_generados = st.empty()
    comentarios_generados.info("iniciando generación de comentarios...")

    last_i = 0
    def actualizar(i):
        global last_i
        last_i = i
        comentarios_generados.info(f"comentarios generados: {i}")

    comentarios = generar_narracion(resultados, actualizar)
    comentarios_generados.success(f"finalizó generación de comentarios ({last_i})")
    for comentario in comentarios:
        st.write(f"**{int(comentario['frame'] / fps)} {comentario['comentario']}**")
        st.write()


finalizo_narracion = False

if comentarios is not None:
    narracion_generada = st.empty()
    narracion_generada.info("iniciando generación de narración...")
    last_i = 0

    def actualizar(i):
        global last_i
        last_i = i
        narracion_generada.info(f"comentarios narrados: {i}")

    generar_tts_comentarios(comentarios, actualizar)
    finalizo_narracion = True
    narracion_generada.success("finalizó generación de narración")


if finalizo_narracion:
    renderizado_video = st.empty()
    renderizado_video.info("renderizando video final...")

    unique_filename = f"{uuid.uuid4()}.mp4"
    salida_final = os.path.join("temp", unique_filename)

    generar_video(save_path, salida_final, comentarios)
    renderizado_video.success(f"video finalizado en: {salida_final}")

    st.video(salida_final)