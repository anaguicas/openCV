# Proyecto OpenCV

## Procesamiento de imagenes

Se realizó el procesamiento de una imagen con la libreria opencv. Se inició cargando y mostrando una imagen, luego se realizó el código en el que la imagen se pasa a escala de grises, esta función se trabajo de manera secuencial, una vez terminado esto se realizó el código de manera paralela, por último se paso la imagen a escala de grises usando la función sobel de opencv.

### Función secuencial
En la función secuencial se creó una variable (gray) de tipo unsigned char para almacenar la imagen convertida en escala de grises, la imagen se fue recorriendo en cada uno de los componenetes (rojos, verde y azul) y se multiplicó con una constante diferente en cada componente para poder hacer la conversión a escala de grises, este resultado se iba almacenando en la varialble gray, una vez terminado el recorrido se mostró la nueva imagen es escala de grises.

### Función Paralela
En la función paralela se realizón un proceso similar al de la función secuencial en cuanto al recorrido de la imagen inicial y multiplicación de constantes.
