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
        "Escribe el nombre de una imagen de prueba "
        "(ejemplo: IMAGEN1LIB.jpg)\n"
        "o la ruta completa de una imagen externa:\n"
    )

    # Caso 1: ruta externa válida
    if os.path.exists(nombre_archivo):
        image_path = nombre_archivo
    else:
        # Caso 2: buscar dentro del paquete
        image_path = os.path.join(images_folder, nombre_archivo)

    # Verificación final
    if not os.path.exists(image_path):
        print(f"\nError: No se encontró la imagen '{nombre_archivo}'.")
        print("Si es externa, escribe la ruta completa.")
        print("Ejemplo: C:/Users/Profesor/Desktop/plano.jpg")
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
    cv2.rectangle(overlay, (20, 40), (450, 180), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, resultado, 0.3, 0, resultado)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2

    # Texto sobre imagen
    cv2.putText(resultado, f"Analisis: {nombre_archivo}", (40, 75),
                font, 0.6, (255, 255, 255), 1)

    cv2.putText(resultado, f"Lineas detectadas: {num_lineas}", (40, 115),
                font, scale, (0, 180, 255), thickness)

    cv2.putText(resultado, f"Esquinas (Harris): {num_esquinas}", (40, 155),
                font, scale, (50, 255, 50), thickness)

    # Ventanas ajustables
    ventanas = [
        ("Imagen Original", img),
        ("Imagen Rectificada (Transformacion Afin)", img_corr),
        ("Resultado Final", resultado)
    ]

    for nombre, imagen in ventanas:
        cv2.namedWindow(nombre, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(nombre, 900, 650)
        cv2.imshow(nombre, imagen)

    # Consola
    print("\n" + "-" * 40)
    print(f"RESULTADOS PARA: {nombre_archivo}")
    print(f"> Lineas detectadas: {num_lineas}")
    print(f"> Esquinas detectadas: {num_esquinas}")
    print(f"> Angulo de correccion aplicado: {angulo:.2f}°")
    print("-" * 40)
    print("Proceso finalizado correctamente.")
    print("Presiona cualquier tecla sobre las imágenes para cerrar.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    
 