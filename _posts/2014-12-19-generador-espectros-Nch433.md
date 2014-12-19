---
layout: post
title: Programa generador de espectros segun la Nch 433
comments: true
description: Generar espectros de diseño segun Nch 433
---

Hace un tiempo cree este programa en[ C#](http://es.wikipedia.org/wiki/C_Sharp) y queria compartirlo.
Bueno lo que hace es generar los espectros de diseño segun la norma chilena Nch 433, ocupando los tipos de suelo del [decreto 61 ](http://www.leychile.cl/Navegar?idNorma=1034101).

![My helpful screenshot](/assets/nch433cap.JPG)




## Modo de uso

El modo de uso es simple y  solo se deben seleccionar los siguientes datos dentro de los valores predeterminados:

- **Categoría**
- **Zona sísmica**
- **Tipo de suelo**
- **Factor de modificación R**

Luego se deben ingresar los periodos con mayor masa asociada, **Tx** y **Ty**.

Posteriormente, se debe  apretar el botón **Generar espectros** y luego **Generar archivos**, ahora se abrirá el explorar de carpetas, donde se debe seleccionar donde se desea guardar los archivos *.txt* con los espectros.


## Descarga
EL programa  se puede descargar desde [aquí]({{ site.url }}assets/EspectrosNch433.zip),
y si quieres ver o bajar el código fuente puedes verlo en [GitHub](https://github.com) haciendo click [aquí](https://github.com/benjad/espectrosNch433).

Cualquier consulta o comentario más abajo !
