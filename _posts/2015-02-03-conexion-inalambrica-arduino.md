---
layout: post
title: Conexión inalámbrica de un Arduino con el APC220
comments: true
description: conectar arduino inalámbricamente con apc220
---
 

Buscando una forma inalámbrica de conectar el pc con un [Arduino](http://www.arduino.cc) Uno encontré los APC220 que parecían tener la mejor relación rango de alcance/precio.
![apc220](/assets/apc220.JPG)

## Configuración del APC220
La configuración es sencilla y se realiza con el programa [rfmagic](http://www.dfrobot.com/image/data/TEL0005/rfmagic.rar), con el que se pueden escribir o leer los datos de cada módulo. 

![rfmagic](/assets/rfmagic.JPG)

Se deben tener todos los valores iguales en los 2 módulos con la excepción del **NODE ID** , que es el id individual de cada APC220.
 
 Una vez listo esto ya se puede conectar un módulo al pc y el otro al arduino.

![conexion](/assets/arduino-apc220.JPG)

## Código de Arduino

Ahora cargaremos el [Arduino](http://www.arduino.cc) Uno  con el siguiente código que nos permitirá ver  si todo funciona bien.

{% highlight c++  %}
int val = 0;
int ledPin = 13;
void setup()
{
 Serial.begin(9600);
  pinMode( ledPin, OUTPUT );
}

void loop()
{
 val = Serial.read(); 
 if (-1 != val) {
   Serial.println(val);
  if ('A' == val || 'alo' == val) {
    Serial.println("Hola desde el Arduino!");
  }else if ('B' == val || 'b' == val) {
     digitalWrite(ledPin, HIGH);
     delay(500);
     digitalWrite(ledPin, LOW);
   }
 }
}
{% endhighlight %}

Ahora si enviamos a traves del serial el valor de **A**   nos devuelve  el saludo "Hola desde el Arduino!"  y si enviamos la letra **B** se producira un parpadeo del led del [Arduino](http://www.arduino.cc) Uno.
Cualquier consulta o comentario más abajo!
