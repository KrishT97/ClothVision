


# En el __init__ la clase recibe donde esta el class_dict.csv, donde esta el encoder de onehot (joblib) y una o
# múltiples imágenes (img* como argumento, el asterisco implica 1 o más en python) y las devuelve en el mismo orden
# en dos listas, la primera tiene la imágen con sus colores de segmentacion en lugar de codificación one hot
# y la segunda contiene que clases tenía cada imagen en una lista de tuplas [(gorro pantalon chaqueta), (abrigo, vaquero, gafas)]
# hacer tan óptima como sea posible.