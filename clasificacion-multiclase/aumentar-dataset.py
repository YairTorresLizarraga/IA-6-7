import os
from os import path, scandir
import cv2

carpetas = [f.path for f in scandir('dataset-procesado') if f.is_dir()]
for carpeta in carpetas:
    print(f'procesando {carpeta}')

    for archivo in os.listdir(carpeta):
        img = cv2.imread(path.join(carpeta, archivo))
        if img is None:
            continue

        nombre_base = archivo.split('.')[0]
        if "iluminado" in nombre_base or "oscuro" in nombre_base or "ruido" in nombre_base or "espejo" in nombre_base or "rotado" in nombre_base:
            continue

        # Aumentar iluminacion
        rostro_mas_iluminacion = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
        cv2.imwrite(os.path.join(carpeta, f'{nombre_base}_iluminado.jpg'), rostro_mas_iluminacion)

        # Reducir iluminacion
        rostro_menos_iluminacion = cv2.convertScaleAbs(img, alpha=0.5, beta=0)
        cv2.imwrite(os.path.join(carpeta, f'{nombre_base}_oscuro.jpg'), rostro_menos_iluminacion)

        # Agregar desenfoque gausiano
        rostro_con_ruido = cv2.GaussianBlur(img, (5, 5), 5)
        cv2.imwrite(os.path.join(carpeta, f'{nombre_base}_ruido.jpg'), rostro_con_ruido)

        # Espejo
        rostro_espejo = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(carpeta, f'{nombre_base}_espejo.jpg'), rostro_espejo)

        # Rotar imagen 90°
        rostro_rotado = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(os.path.join(carpeta, f'{nombre_base}_rotado.jpg'), rostro_rotado)