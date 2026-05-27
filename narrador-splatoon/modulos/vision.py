import re
from collections import deque

import cv2
import numpy as np
import pytesseract
import tensorflow as tf

from modulos.common import parametros, fps, comentar_cada_n_frames, procesar_cada_n_frames

__niveles_health = [3, 1, 2, 0]

__splatted_rectangles = [
    {'x': 569, 'y': 664, 'width': 78, 'height': 28},
    {'x': 569, 'y': 619, 'width': 78, 'height': 28},
    {'x': 569, 'y': 577, 'width': 78, 'height': 28},
    {'x': 569, 'y': 532, 'width': 78, 'height': 28},
]

__respawn_rectangle = {'x': 1094, 'y': 648, 'width': 88, 'height': 29}


def analizar_video(ruta_video: str, callback: callable):
    models = {
        'health': tf.keras.models.load_model('./models/health.keras'),
        'ultimate': tf.keras.models.load_model('./models/ultimate.keras')
    }

    historial_deteccion = []

    frame_actual = 0
    frame_ultimo_comentario = 0

    # Cooldowns para evitar solapamientos constantes
    cooldown_splat_frames = 5 * fps
    cooldown_general_frames = 4 * fps  # Esperar mínimo 4 segs entre comentarios de estado

    tiempo_inicio_ocr = None

    historial_salud_reciente = deque(maxlen=3)
    estado_salud_estable = 0
    historial_ultimate_reciente = deque(maxlen=3)
    estado_ultimate_estable = 'not_ready'

    cap = cv2.VideoCapture(ruta_video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_actual % procesar_cada_n_frames != 0:
            frame_actual += 1
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tiempo_ocr = __leer_tiempo_ocr(frame_rgb)
        if tiempo_inicio_ocr is None and tiempo_ocr is not None:
            tiempo_inicio_ocr = tiempo_ocr

        tiempo_transcurrido_ocr = None
        if tiempo_inicio_ocr is not None and tiempo_ocr is not None:
            tiempo_transcurrido_ocr = max(0, tiempo_inicio_ocr - tiempo_ocr)

        resultados = {
            'frame': frame_actual,
            'tiempo_ocr': tiempo_ocr,
            'tiempo_transcurrido_ocr': tiempo_transcurrido_ocr,
            'debe_comentar': False,
            'health_worsened': False,
            'health_recovered': False,
            'ultimate_became_ready': False,
            'ultimate_used': False,
            'splatted_count': 0,
            'splatted_detected': False,
            'player_was_killed': False,
            'tempo_comment': False,
            'game_status': __obtener_game_status(tiempo_transcurrido_ocr, frame_actual)
        }

        for categoria in parametros:
            frame_categoria = __recortar_imagen_categoria(categoria, frame_rgb)
            frame_categoria = cv2.resize(frame_categoria, (228, 228), interpolation=cv2.INTER_AREA)

            if categoria == "health":
                frame_categoria = __preprocess_health_cv(frame_categoria)

            predicciones = models[categoria].predict(np.expand_dims(frame_categoria, axis=0), verbose=0)
            class_idx = np.argmax(predicciones)
            label = parametros[categoria]['labels'][class_idx]
            resultados[categoria] = label
            resultados[categoria + '_idx'] = class_idx.item()

        splatted_count = __contar_splatted_ocr(frame_rgb)
        resultados['splatted_count'] = splatted_count
        if splatted_count > 0:
            resultados['splatted_detected'] = True

        if __detectar_respawn_ocr(frame_rgb):
            resultados['player_was_killed'] = True

        # --- LÓGICA DE DEBOUNCE (ANTI-PARPADEO) ---
        if 'health_idx' in resultados:
            historial_salud_reciente.append(__niveles_health[resultados['health_idx']])
        if 'ultimate' in resultados:
            historial_ultimate_reciente.append(resultados['ultimate'])

        if len(historial_salud_reciente) == 3 and len(set(historial_salud_reciente)) == 1:
            nuevo_estado_salud = historial_salud_reciente[0]
        else:
            nuevo_estado_salud = estado_salud_estable

        if len(historial_ultimate_reciente) == 3 and len(set(historial_ultimate_reciente)) == 1:
            nuevo_estado_ultimate = historial_ultimate_reciente[0]
        else:
            nuevo_estado_ultimate = estado_ultimate_estable

        # Registrar cambios significativos
        flag_evento_prioritario = False

        if len(historial_deteccion) >= 1:
            if nuevo_estado_salud > estado_salud_estable and nuevo_estado_salud >= 2:
                resultados['health_worsened'] = True
                flag_evento_prioritario = True
            elif nuevo_estado_salud < estado_salud_estable and estado_salud_estable >= 2:
                resultados['health_recovered'] = True
                flag_evento_prioritario = True

            if nuevo_estado_ultimate == 'ready' and estado_ultimate_estable != 'ready':
                resultados['ultimate_became_ready'] = True
                flag_evento_prioritario = True
            elif nuevo_estado_ultimate != 'ready' and estado_ultimate_estable == 'ready':
                resultados['ultimate_used'] = True
                flag_evento_prioritario = True

        estado_salud_estable = nuevo_estado_salud
        estado_ultimate_estable = nuevo_estado_ultimate

        # Lógica de Tempo
        flag_tempo = False
        if frame_actual - frame_ultimo_comentario >= comentar_cada_n_frames:
            resultados['tempo_comment'] = True
            flag_tempo = True

        # --- PUERTA DE CONTROL DE PUBLICACIÓN (COOLDOWNS) ---
        frames_desde_ultimo = frame_actual - frame_ultimo_comentario
        debe_publicar = False

        if resultados['splatted_detected'] or resultados['player_was_killed']:
            # Las muertes interrumpen casi todo
            if frames_desde_ultimo >= cooldown_splat_frames:
                debe_publicar = True
        elif flag_evento_prioritario:
            # Vida y Especiales esperan al menos 4 segundos para no encimarse
            if frames_desde_ultimo >= cooldown_general_frames:
                debe_publicar = True
        elif flag_tempo:
            # Los comentarios de relleno respetan sus 10 segundos
            debe_publicar = True

        if debe_publicar:
            frame_ultimo_comentario = frame_actual
            resultados['debe_comentar'] = True

        frame_actual += 1
        callback(frame_actual)
        historial_deteccion.append(resultados.copy())

    cap.release()

    return historial_deteccion


def __recortar_imagen_categoria(categoria: str, image: np.ndarray) -> np.ndarray:
    crops = []
    for section in parametros[categoria]['sections']:
        left = section['x']
        top = section['y']
        right = section['x'] + section['width']
        bottom = section['y'] + section['height']
        crops.append(image[top:bottom, left:right])

    max_height = max(img.shape[0] for img in crops)
    padded_crops = []

    for img in crops:
        if img.shape[0] < max_height:
            padding = np.zeros((max_height - img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype)
            img = np.concatenate([img, padding], axis=0)
        padded_crops.append(img)

    return np.concatenate(padded_crops, axis=1)


def __preprocess_health_cv(image_np):
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image_float = image_rgb.astype(np.float32) / 255.0
    grayscale = cv2.cvtColor(image_float, cv2.COLOR_RGB2GRAY)
    grayscale_3ch = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)
    return grayscale_3ch


def __recortar_zonas_splatted(image: np.ndarray) -> list[np.ndarray]:
    crops = []
    for rectangle in __splatted_rectangles:
        left = rectangle['x']
        top = rectangle['y']
        right = left + rectangle['width']
        bottom = top + rectangle['height']
        crops.append(image[top:bottom, left:right])
    return crops


def __preprocess_splatted_ocr(image: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grayscale = cv2.resize(grayscale, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    grayscale = cv2.GaussianBlur(grayscale, (3, 3), 0)
    _, threshold = cv2.threshold(grayscale, 180, 255, cv2.THRESH_BINARY)
    return threshold


def __contar_splatted_ocr(frame_rgb: np.ndarray) -> int:
    splatted_count = 0
    for zona_splatted in __recortar_zonas_splatted(frame_rgb):
        imagen_ocr = __preprocess_splatted_ocr(zona_splatted)
        texto = pytesseract.image_to_string(
            imagen_ocr,
            config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        )
        if re.search(r"\bsplatted\b", texto, flags=re.IGNORECASE):
            splatted_count += 1
    return splatted_count


def __recortar_zona_respawn(image: np.ndarray) -> np.ndarray:
    left = __respawn_rectangle['x']
    top = __respawn_rectangle['y']
    right = left + __respawn_rectangle['width']
    bottom = top + __respawn_rectangle['height']
    return image[top:bottom, left:right]


def __preprocess_respawn_ocr(image: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grayscale = cv2.resize(grayscale, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    grayscale = cv2.GaussianBlur(grayscale, (3, 3), 0)
    _, threshold = cv2.threshold(grayscale, 170, 255, cv2.THRESH_BINARY)
    return threshold


def __detectar_respawn_ocr(frame_rgb: np.ndarray) -> bool:
    zona_respawn = __recortar_zona_respawn(frame_rgb)
    imagen_ocr = __preprocess_respawn_ocr(zona_respawn)
    texto = pytesseract.image_to_string(
        imagen_ocr,
        config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    )
    return re.search(r"\brespawn\b", texto, flags=re.IGNORECASE) is not None


__timer_rectangle = {'x': 600, 'y': 31, 'width': 74, 'height': 37}


def __preprocess_timer_ocr(image: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grayscale = cv2.resize(grayscale, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    grayscale = cv2.GaussianBlur(grayscale, (3, 3), 0)
    _, threshold = cv2.threshold(grayscale, 160, 255, cv2.THRESH_BINARY)
    return threshold


def __leer_tiempo_ocr(frame_rgb: np.ndarray) -> int | None:
    left = __timer_rectangle['x']
    top = __timer_rectangle['y']
    right = left + __timer_rectangle['width']
    bottom = top + __timer_rectangle['height']
    zona_timer = frame_rgb[top:bottom, left:right]
    imagen_ocr = __preprocess_timer_ocr(zona_timer)
    texto = pytesseract.image_to_string(
        imagen_ocr,
        config="--psm 7 -c tessedit_char_whitelist=0123456789:",
    )
    match = re.search(r"(\d{1,2})\s*:?\s*(\d{2})", texto)
    if not match:
        return None
    minutos = int(match.group(1))
    segundos = int(match.group(2))
    if segundos >= 60:
        return None
    return minutos * 60 + segundos


def __obtener_game_status(tiempo_transcurrido: float | None, frame: int) -> str:
    segundos = tiempo_transcurrido if tiempo_transcurrido is not None else frame / fps
    return 'start' if segundos < 15 else 'mid game' if segundos < 120 else 'last minute' if segundos < 150 else 'last 30 seconds'
