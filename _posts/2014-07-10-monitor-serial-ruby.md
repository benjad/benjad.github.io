---
layout: post
title: Guardando datos del puerto serial 
comments: true
description: Guardar datos provenientes del puerto serial del arduino con ruby
---

Jugando con mi [Arduino](http://www.arduino.cc/) UNO y algunos sensores, me vi en la necesidad de guardar los datos proveniente del puerto serial. Y
aunque existen algunos programas disponibles en la internet pensé que sería un buen ejercicio crearlo desde 0 en  [Ruby](https://www.ruby-lang.org/es/).

##El Código



El código es bien sencillo y bastante directo. El programa genera un archivo de texto  llamado *log.txt*, con los datos provenientes del puerto de serial.
El registro de los datos se hace desde que se inicia el programa hasta su cierre.


Para que funcione correctamente se debe escribir el numero correcto del puerto del [Arduino](http://www.arduino.cc/) en *port\_str* y la frecuencia en *baud\_rate*. Para el caso del puerto se debe restar 1 al número COM de puerto, es decir, si es COM3 se debe poner 2 en *port\_str* y listo.

Cualquier consulta o comentario más abajo !
