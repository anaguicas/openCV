#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <cuda.h>

#define RED 2
#define GREEN 1
#define BLUE 0

#define MASK_WIDTH 3

using namespace cv;

__constant__ char M[MASK_WIDTH*MASK_WIDTH];

using namespace cv;

__device__ unsigned char clamp(int value){
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return (unsigned char)value;
}

__global__ void sobelFilter_cache(unsigned char *imageInput, int width, int height, \
        unsigned int maskWidth,unsigned char *imageOutput){

    unsigned int row = blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x+threadIdx.x;

    int Pvalue = 0;

    int N_start_point_row = row - (maskWidth/2);
    int N_start_point_col = col - (maskWidth/2);

    for(int i = 0; i < maskWidth; i++){
        for(int j = 0; j < maskWidth; j++ ){
            if((N_start_point_col + j >=0 && N_start_point_col + j < width) \
                    &&(N_start_point_row + i >=0 && N_start_point_row + i < height)){
                Pvalue += imageInput[(N_start_point_row + i)*width+(N_start_point_col + j)] * M[i*maskWidth+j];
            }
        }
    }
    imageOutput[row*width+col] = clamp(Pvalue);
}

//---------------Función paralela----------------------------
__global__ void img2gray(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if((row < height) && (col < width)){
        imageOutput[row*width+col] = imageInput[(row*width+col)*3+RED]*0.299 + imageInput[(row*width+col)*3+GREEN]*0.587 \
                                     + imageInput[(row*width+col)*3+BLUE]*0.114;
    }
}

//---------------Función secuencial------------------------
unsigned char *imagenGrises(unsigned char *gray, int height, int width, Mat image){ 
    unsigned char *imagen_salida;
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
        gray[(i*width+j)]= 0.299*image.data[(i*width+j)*3+2] + 0.587*image.data[(i*width+j)*3+1] + 0.114*image.data[(i*width+j)*3];
    }    
    }
    imagen_salida=gray;
    return imagen_salida;
}

//--------------Función sobel cache---------------------



int main(int argc, char **argv)
{
    cudaError_t error = cudaSuccess;
    clock_t start, end, startGPU, endGPU;
    double cpu_time_used, gpu_time_used;
    char* imageName = argv[1];
    unsigned char *image_aux,*gray;
    unsigned char *h_imageInput, *d_imageInput, *d_imageOutput, *h_imageOutput;
    unsigned char *h_sobelOuput,*d_sobelOutput;
    Mat image;
    image = imread(imageName, 1);

    if(argc !=2 || !image.data){
        printf("No image Data \n");
        return -1;
    }

    Size s = image.size();

    int width = s.width;
    int height = s.height;
    int size = sizeof(unsigned char)*width*height*image.channels();
    int tama = sizeof(unsigned char)*width*height;

    //------------Imagenes en escala de grises secuencial----
    Mat image_gray;

    gray = (unsigned char*)malloc(size);
    start = clock();
    image_aux=imagenGrises(gray,height,width,image);
    end=clock();
    image_gray.create(height,width,CV_8UC1);
    image_gray.data=image_aux;

    //--------------------------------------------------

    //----------------Imagen en grises paralelo---------------------
    h_imageInput = (unsigned char*)malloc(size);
    error = cudaMalloc((void**)&d_imageInput,size);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_imageInput\n");
        exit(-1);
    }

    h_imageOutput = (unsigned char *)malloc(tama);
    error = cudaMalloc((void**)&d_imageOutput,tama);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_imageOutput\n");
        exit(-1);
    }

    h_sobelOuput = (unsigned char*)malloc(tama);
    error= cudaMalloc((void**)&d_sobelOutput,tama);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_sobelOutput\n");
        exit(-1);
    }

    h_imageInput = image.data;

    startGPU = clock();
    error = cudaMemcpy(d_imageInput,h_imageInput,size, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        printf("Error copiando los datos de h_imageInput a d_imageInput \n");
        exit(-1);
    }

    int blockSize = 32;
    dim3 dimBlock(blockSize,blockSize,1);
    dim3 dimGrid(ceil(width/float(blockSize)),ceil(height/float(blockSize)),1);
    img2gray<<<dimGrid,dimBlock>>>(d_imageInput,width,height,d_imageOutput);
    cudaDeviceSynchronize();
    sobelFilter_cache<<<dimGrid,dimBlock>>>(d_imageOutput,width,height,3,d_sobelOutput);
    cudaMemcpy(h_imageOutput,d_imageOutput,tama,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sobelOuput,d_sobelOutput,tama,cudaMemcpyDeviceToHost);
    endGPU = clock();

    //---------------------------------------------------------------
    Mat gray_image;
    gray_image.create(height,width,CV_8UC1);
    gray_image.data = h_imageOutput;

    Mat gray_image_opencv;
    cvtColor(image, gray_image_opencv, CV_BGR2GRAY);


    //imwrite("./Gray_Image.jpg",gray_image);

    namedWindow("Image", WINDOW_NORMAL);
    namedWindow("Gray Image CUDA secuencial", WINDOW_NORMAL);
    namedWindow("Gray Image CUDA paralelo", WINDOW_NORMAL);
    //namedWindow("Gray Image OpenCV", WINDOW_NORMAL);

    imshow("Image",image);
    imshow("Gray Image CUDA", image_gray);
    imshow("Gray Image CUDA", gray_image);
    //imshow("Gray Image OpenCV",gray_image_opencv);

    waitKey(0);

    //free(h_imageInput);
    gpu_time_used = ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC;
    printf("Tiempo Algoritmo Paralelo: %.10f\n",gpu_time_used);
    cpu_time_used = ((double) (end - start)) /CLOCKS_PER_SEC;
    printf("Tiempo Algoritmo OpenCV: %.10f\n",cpu_time_used);
    printf("La aceleración obtenida es de %.10fX\n",cpu_time_used/gpu_time_used);

    cudaFree(d_imageInput);
    cudaFree(d_imageOutput);
    return 0;
}
