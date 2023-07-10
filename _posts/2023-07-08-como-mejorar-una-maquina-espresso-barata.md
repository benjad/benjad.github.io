---
layout: post
title: Como mejorar una maquina de espresso barata (Parte 1)
comments: true
description: instalar PID en maquina espresso
---

Hace varios años me compre una máquina de espresso bien económica (costó unos 40 dólares) y aunque aún me puedo hacer un café sin problemas, no siempre el resultado es de los mejores. Investigando un poco encontre que un posibe  origen del problema es la regulación de temperatura que se realiza con un termostato, lo que hace que sea inestable y que a veces queme el café.

![maquina de espresso](/assets/maquina_recco.jpg)


## Temperatura de extracción
Para verificar esto se me ocurrió conectarle un sensor de temperatura directamente al hervidor y con una placa Arduino Uno medir la variación de temperatura en el tiempo. El sensor exacto que ocupe es el DS18B20 y también se necesita una resistencia de 500k ohm. El diagrama de conexión es el siguiente:

![diagrama arduino](/assets/diag1.png)

Bueno y asi se ve el sensor instalado, pegado al hervidor:
![sensor instalado](/assets/sensor_en_termo.png)


Y si graficamos la variación en el tiempo, nos da esto:
![grafico temperatura vs tiempo](/assets/temp.png)

Ahora que vemos el grafico se hace obvio (o no?) que existe una gran variación de temperatura, que llega a un mínimo de 78° y al máximo de 95°. Esto hace que la extracción sea una tómbola y que el resultado pueda ser cualquier cosa (buena o mala).

## ¿Como estabilizar la temperatura?: controlador PID
El controlador PID es una manera mas eficiente de regular una señal (en nuestro caso la temperatura) de un sistema. Una explicación mas detallada la pueden leer a  [aca](https://blog.330ohms.com/2021/06/02/que-es-un-control-pid/) o tambien ver este [video](https://www.youtube.com/watch?v=UR0hOmjaHp0).
En la segunda parte de este post veremos como programar el controlador con Arduino y asi mejorar nuestro cafecito espresso.

Cualquier consulta o comentario más abajo!





