---
layout: post
title: Conexión de arduino con acelerometro ADXL335 
comments: true
description: conectar arduino con acelerometro
---
 

El acelerómetro ADXL335 es un sensor muy útil y fácil de usar para variados proyectos. Tiene un rango de +-3g , lectura en 3 ejes (x,y,z) y un precio bastante económico.


El diagrama de conexión con el [Arduino](http://www.arduino.cc) Uno es el siguiente:

![apc220](/assets/adxl335.JPG)

Las conexiones son simples y lo único que merece explicación es el enlace del pin AREF con el   de 3.3V. Este es el voltaje máximo de referencia al hacer la conversión de análogo a digital que debe hacer el  [Arduino](http://www.arduino.cc) Uno. Al tener un rango de 10bit, cuando al acelerómetro envié 3.3V se nos devolverá el valor de 1023 y cuando sea 0V el valor de 0.


##El codigo
El código se encarga de hacer la lectura análoga de los 3 ejes en los pines A1, A2 Y A3. Con estos valores se hace la conversión a G (aceleración de gravedad) y luego envía esos datos a través del puerto serial.



{% highlight c  %}
int val = 0;

int xpin = A1;                  
int ypin = A2;                  
int zpin = A3;                
 
int t_muestreo = 500;   // Tiempo de muestreo del acelerómetro
void setup()
{
 Serial.begin(9600);
  analogReference(EXTERNAL);  // La tensión aplicada en el pin AREF (3.3V) será la que haga que el conversor analogo-digital
                              // de su máxima lectura (1023) 
    
  pinMode(xpin, INPUT);
  pinMode(ypin, INPUT);
  pinMode(zpin, INPUT);
}

void loop()
{
 int x = analogRead(xpin); // Leemos el valor de la tensión en el pin x
 
    delay(1); // Esperamos 1 ms a leer en el próximo pin
 
  int y = analogRead(ypin); // Leemos el valor de la tensión en el pin y
 
    delay(1); // Esperamos 1 ms a leer en el próximo pin
 
  int z = analogRead(zpin);
 
  // Una conversión analogo a digital va de 0 a 1023, siendo 512 
  // la mitad del rango y por lo tanto el 0
  float zero_G = 512.0;
 
  // Según el Datasheet, tenemos incrementos de 330mV por cada G de aceleración
  // por lo tanto, si pasamos de mV (330) a cuentas (1023)
  // nos queda que 1023cuentas/( 3.3V/330mV)  = 102.3, valor para convertir mV a G's
  // escala es el número de unidades que esperamos que el sensor lea cuando
  // hay un cambio de aceleración en 1G
 
  float escala = 102.3;
 
  Serial.print(((float)x - zero_G)/escala);
  Serial.print("\t");
 
  Serial.print(((float)y - zero_G)/escala);
  Serial.print("\t");
 
  Serial.print(((float)z - zero_G)/escala);
  Serial.print("\n");
 
  // delay entre cada lectura
  delay(t_muestreo);
}
{% endhighlight %}



Cualquier consulta o comentario más abajo!
