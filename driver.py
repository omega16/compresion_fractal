#!/usr/bin/env python

import fractal_compress as fc

import sys 



"""

24-Nov-2017

Metodo de compresion de imagenes usando fractales (PIFS)



Proyecto echo para la clase de Topologia II en ESFM-IPN 
Omega16 (nombre omitido por privacidad)
https://github.com/omega16/compresion_fractal/

Ver Readme en git para encontrar instrucciones sobre las librerias usadas

"""



# direccion de imagen , iteraciones de descompresion, si se muestra o no las imagenes al terminar 




fc.driver(sys.argv[1],sys.argv[2],sys.argv[3])



