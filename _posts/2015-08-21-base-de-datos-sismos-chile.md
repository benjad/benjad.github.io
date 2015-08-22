---
layout: post
title: Base de datos de sismos en Chile, obtenida con web crawler desde sismologia.cl
comments: true
description: Base de datos de sismos en Chile obtenida con web crawler
---
 



En [sismologia.cl](http://sismologia.cl/) se pueden revisar todos los sismos ocurridos en Chile desde el 2003, y aunque la información es súper completa nunca me gusto el formato que tiene la página, donde solo se pueden ver los registros de un día a la vez. Por esto es que se ocurrio la idea de crear una base de datos con todos los sismos, utilizando un [web crawler](https://es.wikipedia.org/wiki/Araña_web) para recopilar la info.

##La base de datos
El [web crawler](https://es.wikipedia.org/wiki/Araña_web) crea la base de datos en un archivo .csv (editable en excel), donde en cada fila se ubica un sismo con su info  separado en 6 columnas (Fecha local, Fecha UTC, latitud, longitud, magnitud, referencia geografica).

El archivo final, con los sismos desde el 01/01/2003 hasta el 20/08/2015, puedes  [bajarlo aquí](/assets/sismos.csv).



## Código del web crawler
El codigo del programa esta escrito en [Ruby](https://www.ruby-lang.org/es/) y es el siguiente:

{% highlight ruby  %}
require 'rubygems'
require 'nokogiri'
require 'open-uri'
require 'csv'
require 'date'

csv = CSV.open("sismos.csv", 'w',{:col_sep => ",", :quote_char => '\'',  :row_sep =>:auto, :force_quotes => false}) # se crea el archivo .csv

BASE_URL = 'http://www.sismologia.cl/events/listados'

date_ini =Date.new(2003,1,01) # fecha inicio del registro
date_end =Date.new(2015,8,20) # fecha final del registro
days = (date_end - date_ini).to_int # dias de datos a recopilar



for i in 1..days do # comienza la recopilacion de datos
	BASE_DIR = "/#{date_ini.strftime("%Y")}/#{date_ini.strftime("%m")}/#{date_ini.strftime("%Y%m")}" 
		begin
		page = Nokogiri::HTML(open(BASE_URL+BASE_DIR + date_ini.strftime("%d") + '.html'))
		rescue Exception=>e
        puts "Error: #{e}"
        sleep 2
      	else

		rows = page.css('tbody tr')
			rows[1..-2].each do |row| 
				csv <<row.css('td').map{ |cell| (cell).text.gsub(/,/,'')} # se guarda cada fila en el archivo .csv
			end

		date_ini +=1 #avanza 1 dia

		sleep 1 # espera 1 segundo para no saturar el servidor
	  end
end 
{% endhighlight %}



Cualquier consulta o comentario más abajo!
