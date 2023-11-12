### Primera implementación con U-Net

Se ha considerado las imagenes con 560x560 por temas de pooling y unpooling, no se le aplica ninguna transformacion porque hay que arreglar el problema de RandomHorizontalFlip.

El número de batch size es recomendable a 32.

El conjunto de las imagenes se han divido en 70:15:15,
en el entrenamiento se incorporará la aleatoriedad para mejor rendimiento y el modelo sea robusto, mientras que para el conjunto de validación y test no.

Se está usando AdaMax ya que el optimizador que es el mejor adaptado para problemas de tipo u-net para solventar el sobreajuste, otra alternativa; SGD o Adam.

La arquitectura se basa en la de u-net adaptando las dimensiones de las imagenes a las correspondientes en nuestro caso.

Para temas de CPU, tarda mucho tiempo en cuanto al entrenamiento y la validación, la manera de comprobar los resultados es utilizando la GPU.

