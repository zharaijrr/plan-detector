# -*- coding: utf-8 -*-
"""
Created on Fri May  1 01:20:00 2026
@author: ISAHISURISADAYIBARRA


CODIGO MAIN
"""
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 01:20:00 2026
@author: ISAHISURISADAYIBARRA

CODIGO MAIN
"""
import cv2
import numpy as np
import os

from .preprocesamiento import preprocess
from .deteccion import detect
from .postprocesamiento import postprocess


def main():
    print("=== PIPELINE DE DETECCIÓN DE PLANOS ===")

    # Carpeta interna del paquete
    base_path = os.path.dirname(__file__)
    images_folder = os.path.join(base_path, "IMAGENES")

    # Selección dinámica
    print("\n[CONFIGURACIÓN]")
    nombre_archivo = input(
        "Escribe el nombre de la imagen (ejemplo: IMAGEN1LIB.jpg): "
    )

    image_path = os.path.join(images_folder, nombre_archivo)

    # Verificar existencia
    if not os.path.exists(image_path):
        print(f"\nError: La imagen '{nombre_archivo}' no existe.")
        print("Verifica que esté dentro de la carpeta IMAGENES del paquete.")
        return

    # Cargar imagen
    img = cv2.imread(image_path)

    if img is None:
        print("Error: no se pudo cargar la imagen")
        return

    print("\n[PROCESANDO...]")

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
    print("-" * 40)
    print(f"Archivo analizado: {nombre_archivo}")
    print(f"Lineas detectadas: {num_lineas}")
    print(f"Esquinas detectadas: {num_esquinas}")
    print(f"Angulo de correccion aplicado: {angulo:.2f}°")
    print("-" * 40)
    print("Proceso finalizado correctamente.")
    print("Presiona cualquier tecla sobre las imágenes para cerrar.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
