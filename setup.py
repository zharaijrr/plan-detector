# -*- coding: utf-8 -*-
"""
Created on Fri May  1 16:28:41 2026

@author: CITLALI
"""

from setuptools import setup, find_packages

setup(
    name="plan_detector",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "plan_detector": ["IMAGENES/*.jpg"]
    },
    install_requires=[
        "opencv-python",
        "numpy"
    ],
    author="Equipo POZONKE",
    description="Libreria para deteccion de lineas, esquinas e intersecciones en planos arquitectonicos"
)