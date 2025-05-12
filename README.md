# AgroBot 🤖🌱

**AgroBot** es un robot autónomo diseñado para la recolección eficiente de berries en invernaderos. Utiliza visión por computadora, movilidad inteligente y un brazo robótico para identificar y recolectar frutos maduros de forma precisa y segura.

## Características

- 🎯 Detección de frambuesas mediante inteligencia artificial (YOLO11)
- 🦾 Brazo robótico de 6 grados de libertad
- 🎥 Cámara estereoscópica para detección en 3D
- 🧠 Control basado en RaspberryPi + Python
- 🌿 Enfoque en cultivos de invernadero

## Objetivo

Reducir la dependencia de mano de obra para la recolección de berries, mejorar la eficiencia del proceso agrícola y desarrollar una solución escalable y replicable para el campo mexicano.

## Estructura del proyecto

Nuestro proyecto consta de dos secciones, la primera siendo un brazo de seis grados de libertad, está construido a base de PLA y servomotores, con un Arduino UNO para control interior, la segunda sección siendo un módulo de identificación de objetos compuesto por un Raspberry PI 4, un módulo de cámara PI V2, y una conexión serial entre el PI 4 y el Arduino. 

Nuestro prototipo reconoce los frutos en sus diferentes etapas de crecimiento, desde su inicio de crecimiento hasta su punto de recolección ideal utilizando un modelo de IA YOL11 para identificación de objetos, y cuando identifica las posiciones deseadas, es capaz realizar movimientos complejos en el plano tridimensional, necesarios para recoger frutos en diferentes configuraciones de cultivos de manera óptima y garantizando que los frutos que recogidos queden en excelente estado.  
