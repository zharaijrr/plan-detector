# -*- coding: utf-8 -*-
"""
Created on Fri May  1 00:58:54 2026

@author: ISAHISURISADAYIBARRA


CODIGO DETECCION
"""

import numpy as np
import cv2

def manual_convolution(image, kernel):
    i_h, i_w = image.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    output = np.zeros_like(image, dtype=np.float32)
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    for y in range(i_h):
        for x in range(i_w):
            roi = padded_img[y:y+k_h, x:x+k_w]
            output[y, x] = np.sum(roi * kernel)
    return output


def detect_edges_filtered(img_pre):
    mx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    my = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = manual_convolution(img_pre, mx)
    gy = manual_convolution(img_pre, my)
    magnitud = np.sqrt(gx**2 + gy**2)
    return np.where(magnitud > 60, 255, 0).astype(np.uint8)


def detect_hough_lines_filtered(img_edges, threshold_votos=450):
    rows, cols = img_edges.shape
    diagonal = int(np.sqrt(rows**2 + cols**2))
    thetas = np.deg2rad(np.arange(0, 180))
    accumulator = np.zeros((2 * diagonal, 180), dtype=np.int32)
    y_idxs, x_idxs = np.where(img_edges > 0)

    for i in range(len(x_idxs)):
        x, y = x_idxs[i], y_idxs[i]
        for t_idx in range(180):
            rho = int(x * np.cos(thetas[t_idx]) + y * np.sin(thetas[t_idx]))
            accumulator[rho + diagonal, t_idx] += 1

    raw_lines = np.argwhere(accumulator > threshold_votos)
    filtered = []

    if len(raw_lines) > 0:
        raw_lines = sorted(raw_lines, key=lambda x: x[1])
        last_r, last_t = raw_lines[0][0]-diagonal, thetas[raw_lines[0][1]]
        filtered.append((last_r, last_t))

        for i in range(1, len(raw_lines)):
            curr_r = raw_lines[i][0] - diagonal
            curr_t = thetas[raw_lines[i][1]]
            if abs(curr_r - last_r) > 45 or abs(curr_t - last_t) > np.deg2rad(20):
                filtered.append((curr_r, curr_t))
                last_r, last_t = curr_r, curr_t

    return filtered


def detect_corners_harris(img_pre):
    dx = manual_convolution(img_pre, np.array([[0,0,0],[-1,0,1],[0,0,0]], dtype=np.float32))
    dy = manual_convolution(img_pre, np.array([[0,-1,0],[0,0,0],[0,1,0]], dtype=np.float32))

    window = np.ones((5,5), np.float32)
    Sxx = manual_convolution(dx**2, window)
    Syy = manual_convolution(dy**2, window)
    Sxy = manual_convolution(dx*dy, window)

    R = (Sxx * Syy - Sxy**2) - 0.04 * (Sxx + Syy)**2

    threshold = 0.35 * R.max()
    R_mask = np.where(R > threshold, R, 0)

    data_max = cv2.dilate(R_mask, None)
    R_final = np.where((R_mask == data_max) & (R_mask > 0), 255, 0).astype(np.uint8)

    points = np.argwhere(R_final > 0)
    validated_points = []

    for p in points:
        if img_pre[p[0], p[1]] == 255:
            validated_points.append((p[1], p[0]))

    final_unique_points = []
    min_dist = 25

    for p in validated_points:
        if all(np.linalg.norm(np.array(p) - np.array(p2)) > min_dist for p2 in final_unique_points):
            final_unique_points.append(p)

    return np.array([[p[1], p[0]] for p in final_unique_points])


def detect(img_pre):
    edges = detect_edges_filtered(img_pre)
    lines = detect_hough_lines_filtered(edges)
    corners = detect_corners_harris(img_pre)

    return edges, lines, corners