---
layout: post
title: Guardando datos del puerto serie 
comments: true
---



Jugando con mi [Arduino](http://www.arduino.cc/) UNO y algunos sensores, me vi en la necesidad de guardar los datos proveniente del puerto serie. Y
aunque existen algunos programas disponibles en la internet pensé que sería un buen ejercicio crearlo desde 0 en  [Ruby](https://www.ruby-lang.org/es/).

##El Código

{% highlight ruby  %}
require 'serialport'

filename = 'log.txt'

port_str  = 2
baud_rate = 9600
data_bits = 8
stop_bits = 1
parity    = SerialPort::NONE
 trap("INT") { puts "Apagando."; exit}
target = File.new(filename, 'w')
sp = SerialPort.new(port_str, baud_rate, data_bits, stop_bits, parity)
while(true) do
  message = (sp.gets)
   print message
  target.write(message)
 end
{% endhighlight %}

El código es bien sencillo y bastante directo. El programa genera un archivo de texto  llamado *log.txt*, con los datos provenientes del puerto de serie.
El registro de los datos se hace desde que se inicia el programa hasta su cierre.


Para que funcione correctamente se debe escribir el numero correcto del puerto del [Arduino](http://www.arduino.cc/) en *port\_str* y la frecuencia en *baud\_rate*. Para el caso del puerto se debe restar 1 al número COM de puerto, es decir, si es COM3 se debe poner 2 en *port\_str* y listo.

Cualquier consulta o comentario más abajo !
