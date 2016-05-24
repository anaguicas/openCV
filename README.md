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

tamaño|	OpenCV
259x194|	0.0024056
350x271|	0.0037532
600x400|	0.0076142
640x463|	0.0082628
1000x814|	0.0167672

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

### Función Secuencial Vs Función Paralela

tamaño|	Secuencial|	Paralelo|	aceleración
------|-----------|---------|---------------
259x194|	0.0004494|	0.0002284|	1.967600701
350x271|	0.0008654|	0.000399|	2.168922306
600x400|	0.0019776|	0.0009368|	2.111016225
640x463|	0.002845|	0.001143|	2.489063867
1000x814|	0.0074478|	0.002789|	2.670419505

### Función Secuencial Vs Función Sobel Cache Memory

tamaño|	Secuencial|	sobel cache|	aceleración
------|-----------|------------|----------------
259x194|	0.0004494|	0.0002624|	1.712652439
350x271|	0.0008654|	0.0004344|	1.992173112
600x400|	0.0019776|	0.0009892|	1.999191266
640x463|	0.002845|	0.0011962|	2.378364822
1000x814|	0.0074478|	0.0028976|	2.57033407

### Función Secuencial Vs Función Sobel Global

tamaño|	Secuencial|	Sobel global|	aceleración
------|-----------|-------------|--------------
259x194|	0.0004494|	0.0002722|	1.650991918
350x271|	0.0008654|	0.000453|	1.910375276
600x400|	0.0019776|	0.0010516|	1.880562952
640x463|	0.002845|	0.0012874	|2.209880379
1000x814|	0.0074478|	0.0031062|	2.397720688

### Función Secuencial Vs Función Sobel Share

tamaño|	Secuencial|	sobel share|	aceleración
------|-----------|------------|---------------
259x194|	0.0004494|	0.000259|	1.735135135
350x271|	0.0008654|	0.0004294|	2.015370284
600x400|	0.0019776|	0.0009862|	2.005272764
640x463|	0.002845|	0.0012104|	2.350462657
1000x814|	0.0074478|	0.0029114|	2.558150718

### Función Secuencial Vs Función Sobel OpenCV

tamaño|	Secuencial|	OpenCV|	aceleración
------|-----------|-------|--------------------
259x194|	0.0004494|	0.0024056|	0.1868141004
350x271|	0.0008654|	0.0037532|	0.2305765747
600x400|	0.0019776|	0.0076142|	0.2597252502
640x463|	0.002845|	0.0082628|	0.344314276
1000x814|	0.0074478|	0.0167672|	0.444188654
