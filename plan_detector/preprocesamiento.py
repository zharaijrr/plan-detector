# -*- coding: utf-8 -*-
"""
Created on Fri May  1 00:58:48 2026

@author: ISAHISURISADAYIBARRA

CODIGO PREPROCESAMIENTO
"""
import cv2
import numpy as np

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

    kernel_limpieza = np.ones((6, 6), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_limpieza)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    final_walls = np.zeros_like(clean)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 800:
            final_walls[labels == i] = 255

    return final_walls
