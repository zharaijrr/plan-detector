# -*- coding: utf-8 -*-
"""
Created on Fri May  1 01:00:09 2026

@author: ISAHISURISADAYIBARRA


CODIGO POSTPROCESAMIENTO
"""
import numpy as np
import cv2

def apply_affine_correction(img, lines):
    if not lines:
        return img, 0

    angles = [np.rad2deg(t) for _, t in lines]
    avg_angle = np.median([(a + 45) % 90 - 45 for a in angles])

    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255,255,255))

    return rotated, avg_angle


def find_intersections(lines, img_pre):
    intersections = []
    h, w = img_pre.shape[:2]

    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            r1, t1 = lines[i]
            r2, t2 = lines[j]

            if abs(t1 - t2) < np.deg2rad(25):
                continue

            A = np.array([[np.cos(t1), np.sin(t1)],
                          [np.cos(t2), np.sin(t2)]])
            b = np.array([r1, r2])

            try:
                xy = np.linalg.solve(A, b)
                ix, iy = int(round(xy[0])), int(round(xy[1]))

                if 0 <= ix < w and 0 <= iy < h:
                    roi = img_pre[max(0, iy-2):iy+3, max(0, ix-2):ix+3]
                    if np.any(roi == 255):
                        intersections.append((ix, iy))
            except:
                continue

    return intersections


def draw_results(img_pre, lines, corners, intersections):
    result = np.full((img_pre.shape[0], img_pre.shape[1], 3), 255, dtype=np.uint8)

    result[img_pre == 255] = [50, 50, 50]

    for r, t in lines:
        a, b = np.cos(t), np.sin(t)
        x0, y0 = a*r, b*r
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(result, (x1,y1), (x2,y2), (0,100,255), 2)

    for p in corners:
        cv2.circle(result, (p[1], p[0]), 6, (0,150,0), -1)

    for x, y in intersections:
        cv2.drawMarker(result, (x, y), (200,0,0), cv2.MARKER_CROSS, 15, 3)

    return result


def postprocess(img, img_pre, lines, corners):
    img_corr, ang = apply_affine_correction(img, lines)
    intersections = find_intersections(lines, img_pre)
    result = draw_results(img_pre, lines, corners, intersections)
    return img_corr, result, ang