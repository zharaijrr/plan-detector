# -*- coding: utf-8 -*-
"""
Created on Fri May  1 01:20:00 2026
@author: ISAHISURISADAYIBARRA


CODIGO MAIN
"""

import cv2
import numpy as np

from .preprocesamiento import preprocess
from .deteccion import detect
from .postprocesamiento import postprocess

def main():
    print("=== PIPELINE DE DETECCIÓN DE PLANOS ===")

    # Cargar imagen
    img = cv2.imread("plan_detector/IMAGENES/IMAGEN5LIB.jpg")

    if img is None:
        print("Error: no se pudo cargar la imagen")
        return

    # Pipeline
    img_pre = preprocess(img)
    edges, lines, corners = detect(img_pre)
    img_corr, resultado, angulo = postprocess(img, img_pre, lines, corners)

    # Conteos
    num_lineas = len(lines) if lines is not None else 0
    num_esquinas = len(corners) if corners is not None else 0

    # Panel visual
    overlay = resultado.copy()
    cv2.rectangle(overlay, (20, 40), (420, 160), (60, 60, 60), -1)
    cv2.addWeighted(overlay, 0.6, resultado, 0.4, 0, resultado)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2

    # Texto en imagen
    cv2.putText(resultado, "Lineas:", (40, 80), font, scale, (0, 180, 220), thickness)
    cv2.putText(resultado, f"{num_lineas}", (280, 80), font, scale, (0, 180, 220), thickness)

    cv2.putText(resultado, "Esquinas:", (40, 125), font, scale, (50, 220, 50), thickness)
    cv2.putText(resultado, f"{num_esquinas}", (280, 125), font, scale, (50, 220, 50), thickness)

    # Ventanas
    cv2.imshow("Imagen Original", img)
    cv2.imshow("Imagen Rectificada (Transformacion Afin)", img_corr)
    cv2.imshow("Resultado Final (Lineas y Esquinas)", resultado)

    # Consola
    print(f"Lineas detectadas: {num_lineas}")
    print(f"Esquinas detectadas: {num_esquinas}")
    print(f"Angulo de correccion aplicado: {angulo:.2f}°")
    print("Proceso finalizado correctamente.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()