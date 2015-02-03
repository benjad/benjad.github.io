---
layout: post
title: Conexión inalámbrica de un Arduino con el APC220
comments: true
description: conectar arduino inalámbricamente con apc220
---
 

Buscando una forma inalámbrica de conectar  un [Arduino](http://www.arduino.cc) Uno con el PC encontré los APC220 que con una distancia de 1000 mts y por un valor de USD$30 en [DealExtreme](http://www.dx.com), parecían tener la mejor relación rango de alcance/precio.

![apc220](/assets/apc220.JPG)

## Configuración del APC220
La configuración es sencilla y se realiza con el programa [rfmagic](http://www.dfrobot.com/image/data/TEL0005/rfmagic.rar), con el que se pueden escribir o leer los datos de cada módulo. 

![rfmagic](/assets/rfmagic.JPG)

Se deben conectar los APC220 al PC y verificar que todos los valores sean iguales en los 2 módulos con la excepción del **NODE ID** , que es el id individual de cada APC220.
 
 

## Código de Arduino

El siguiente paso es cargar el [Arduino](http://www.arduino.cc) Uno  con el siguiente código que nos permitirá ver  si todo funciona bien.

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
  if ('A' == val || 'a' == val) {
    Serial.println("Hola desde el Arduino!");
  }else if ('B' == val || 'b' == val) {
     digitalWrite(ledPin, HIGH);
     delay(500);
     digitalWrite(ledPin, LOW);
   }
 }
}
{% endhighlight %}

Una vez listo esto ya se puede conectar un módulo al pc y el otro al [Arduino](http://www.arduino.cc)

![conexion](/assets/arduino-apc220.jpg)

Ahora si enviamos a traves del monitor serial el valor de **A**   nos devuelve  el saludo "Hola desde el Arduino!"  y si enviamos la letra **B** se producira un parpadeo del led del [Arduino](http://www.arduino.cc) Uno.

Cualquier consulta o comentario más abajo!
