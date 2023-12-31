---
layout: post
title: Como mejorar una maquina de espresso barata (Parte 2)
comments: true
description: instalar PID en maquina espresso 2
---

Esta publicación es la continuación de otro [post](http://benjad.github.io/2023/07/08/como-mejorar-una-maquina-espresso-barata/). 

(**Disclaimer**: ya no tengo la maquina de espresso, por lo que este post es "lo que hubiera hecho" y por lo mismo no puedo garantizar que funcione.)

![maquina de espresso](/assets/maquina_recco_2.jpg)


## Reemplazar el termostato
 Como vimos en la parte, necesitamos mejorar el control de temperatura y para eso debemos reemplazar el termostato por algo de mayor precisión. Normalmente la maquina debería tener 2 termostatos, uno para la temperatura de extracción y otra para el vapor. A nosotros solo nos interesa la primera, porque para el vapor no es tan relevante, con tal supere el punto de ebullición.

<br> 
Así se ve un termostato:
![termostato](/assets/termostato.jpeg)
<br> 
Acá podemos verlo sobre el hervidor:
![sensor instalado](/assets/termostato_en_termo.png)
<br> 
###Bueno y ¿con que reemplazamos el termostato?  
Con un sensor de temperatura (que ya instalamos) y un relay para controlar en encendido/apagado del termo.
Acá se puede ver el diagrama de conexion de ambos elementos:
![diagrama arduino](/assets/diag2.png)

El relay se encargará de abrir y cerrar el circuito que enciende el termo y por lo tanto incrementa la temperatura, y con el sensor que instalamos en la [parte 1](http://benjad.github.io/2023/07/08/como-mejorar-una-maquina-espresso-barata/) controlaremos la temperatura real.

## Programar el controlador PID
Ahora que tenemos como encender/apagar el termo y además podemos monitorear la temperatura, solo tenemos que programar el control PID.. ¿o no?
Bueno después de realizar esta conexión me di cuenta que el control PID necesita que la señal de salida (en nuestro caso sería la potencia del termo) sea variable, pero para nuestro ejemplo solo podemos prender o apagar el termo a una potencia fija.
###¿Entonces que hacemos?
Para simplificar el problema implementaremos solo la P en PID, es decir solamente un control proporcional de la señal. Tambien para simular niveles de potencia se hara un encendido y apagado intermitente, creando dos niveles: 100% y 50%. Para que se entienda mejor, acá un grafico de ejemplo  con la señal tipo "cajón" que busca ser equivalente a una potencia del 50%:

![grafico simular señal](/assets/potencia_sim.JPG)

Ahora que tenemos 2 niveles de potencia en el termo, implementaremos la siguiente lógica:
> 1. si temperatura\_real es mayor a la deseada &rarr; se apagara el termo.
> 2. si (temperatura\_objetivo - temperatura\_real) =< umbral_temp &rarr; se encendera el termo a un 50 % potencia.
> 3. si (temperatura\_real - temperatura\_real) > umbral_temp &rarr; se encendera el termo a un 100 % potencia.


<br> 
Acá está el código con la lógica implementada y además la impresión de la temperatura en el monitor serial:

{% highlight arduino  %}

    // Include the required Arduino libraries:
    #include <OneWire.h>
    #include <DallasTemperature.h>

    // Define to which pin of the Arduino the 1-Wire bus is connected:
    #define ONE_WIRE_BUS 9
    // Define relay pin
    const int RELAY_PIN = 11;  
    // Define desired temp at boiler
    const int SET_TEMP = 90; 
    // Define 50 % power error threshold
    const int 50_THRESHOLD = 10; 
    // Create a new instance of the oneWire class to communicate with any OneWire device:
    OneWire oneWire(ONE_WIRE_BUS);

    // Pass the oneWire reference to DallasTemperature library:
    DallasTemperature sensors(&oneWire);

    void setup() {
    // Begin serial communication at a baud rate of 9600:
    Serial.begin(9600);
    // Start up the library:
    sensors.begin();
    }

    void loop() {
    // Send the command for all devices on the bus to perform a temperature conversion:
    sensors.requestTemperatures();

    // Fetch the temperature in degrees Celsius for device index:
    float tempC = sensors.getTempCByIndex(0); // the index 0 refers to the first device

    // Calculate error  
    float tempError = SET_TEMP - tempC
    
    // if real temp is higher than desired, turn off boiler
    if (tempError<0){
      digitalWrite(RELAY_PIN, LOW);
      delay(4000);
    }
    // if error  is greater than 50% threshold,  turn on bolier
    else if (tempError> 50_THRESHOLD){
      digitalWrite(RELAY_PIN, HIGH);
      delay(4000);
    }
    // if error is within 0 and 50% threshold, use simulated 50% power curve
    else {
      digitalWrite(RELAY_PIN, HIGH);
      delay(2000);
      digitalWrite(RELAY_PIN, LOW);
      delay(2000);
    }

    // Print the temperature in Celsius in the Serial Monitor:
    Serial.print("Temperature: ");
    Serial.print(tempC);
    Serial.print(" \xC2\xB0"); // shows degree symbol
    Serial.print("C  ");
    Serial.println(" \n");

    // Wait 0.5 second:
    delay(500);
    }
{% endhighlight %}


### Pensamientos finales
Diría que la implementación final le queda un poco grande el nombre de "control proporcional", pero aun así es una mejora al termostato que tenía previamente. Una posible mejora es crear mas niveles de potencia para mejorar la proporcionalidad y que el ajuste sea más suave. Por ultimo quiero repetir que no pude implementar esta solución (porque ya no tengo la maquina), por lo que recomiendo ir ajustando 50_THRESHOLD y ver que valor logra una temperatura final mas estable.

Cualquier consulta o comentario más abajo!


