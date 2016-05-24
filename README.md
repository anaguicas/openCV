# Proyecto OpenCV

## Procesamiento de imagenes

Se realizó el procesamiento de una imagen con la libreria opencv. Se inició cargando y mostrando una imagen, luego se realizó el código en el que la imagen se pasa a escala de grises, esta función se trabajo de manera secuencial, una vez terminado esto se realizó el código de manera paralela, por último se paso la imagen a escala de grises usando la función sobel de opencv.

### Función secuencial
En la función secuencial se creó una variable (gray) de tipo unsigned char para almacenar la imagen convertida en escala de grises, la imagen se fue recorriendo en cada uno de los componenetes (rojos, verde y azul) y se multiplicó por una constante diferente en cada componente para poder hacer la conversión a escala de grises, este resultado se iba almacenando en la varialble gray, una vez terminado el recorrido se mostró la nueva imagen es escala de grises.

tamaño |	tiempo ejecución promedio
-------|--------------------------
259x194|	0.0004494
350x271|	0.0008654
600x400|	0.0019776
640x463|	0.002845
1000x814|	0.0074478

### Función Paralela
En la función paralela se realizón un proceso similar al de la función secuencial en cuanto al recorrido de la imagen inicial y multiplicación de constantes.

tamaño|	tiempo ejecución promedio
------|--------------------------
259x194|	0.0002284
350x271|	0.000399
600x400|	0.0009368
640x463|	0.001143
1000x814	0.002789

### Función Sobel OpenCV

### Función Sobel Cache Memory

tamaño|	tiempo ejecución promedio
------|----------------------------
259x194|	0.0002624
350x271|	0.0004344
600x400|	0.0009892
640x463|	0.0011962
1000x814|	0.0028976

### Función Sobel Global Memory

tamaño |	tiempo ejecución promedio
-------|---------------------------------
259x194|	0.0002722
350x271|	0.000453
600x400|	0.0010516
640x463|	0.0012874
1000x814|	0.0031062

### Función Sobel Share Memory
