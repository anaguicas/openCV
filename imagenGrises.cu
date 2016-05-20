#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <cuda.h>

#define RED 2
#define GREEN 1
#define BLUE 0

using namespace cv;
using namespace std;

//----Función paralela para mostrar la imagen en escala de grises
__global__ void img2gray(unsigned char *imageInput, int width, int height, unsigned char *imageOutput){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if((row < height) && (col < width)){
	imageOutput[(row*width+col)]=imageInput[(row*width+col)*3+RED]*0.299+imageInput[(row*width+col)*3+GREEN]*0.587+imageInput[(row*width+col)*3+BLUE]*0.114;
    }
}

//---Función secuencial para mostrar la imagen en escala de grises
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

void sobel(Mat image, unsigned char *,unsigned char*,int mask, int width, int height){
		char mask_x[]={-1,0,1,-2,0,2,-1,0,1};
		char mask_y[]={1,2,1,0,0,0,-1,-2,-1};
		
}

int main( int argc, char** argv )
{
    unsigned char *gray, *image_aux;
    cudaError_t error = cudaSuccess;
    unsigned char *d_imageInput, *d_imageOutput, *h_imageOutput,*h_imageInput;
    //unsigned char *d_SobelOutput_X, *d_SobelOutput_Y, *d_SobelOutput, *h_SobelOutput; 
    clock_t start, end, start_gpu, end_gpu;
    double image_cpu_time,image_gpu_time;
    int width, height;// gray_width, gray_height;
    Mat src, src_gray;
    Mat grad;
    
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
  
    //infoImage image_h;

    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << std::endl;
     return -1;
    }

    Mat image;
    Mat image_gray_opencv;
    Mat image_gray;
    Mat gray_image;

    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << endl ;
        return -1;
    }

    
    Size s = image.size();
    width = s.width;
    height = s.height;
    int size = sizeof(unsigned char)*width*height*image.channels();
    int tama = sizeof(unsigned char)*width*height;
    
    gray = (unsigned char*)malloc(size);  
 
    //----llamado a la función secuencial para mostrar la imagen en grises
    start=clock();
    image_aux= imagenGrises(gray,height,width,image);
    end=clock();    
    image_gray.create(height,width,CV_8UC1);
    image_gray.data=image_aux;
    //----------------------------------------------    

    //-------------Imagen en escala de grises en GPU
    h_imageInput = (unsigned char*)malloc(size);
    h_imageOutput = (unsigned char*)malloc(tama);
    cudaMalloc((void**)&d_imageOutput,tama);
    cudaMalloc((void**)&d_imageInput,size);
    
/*    for(int i=0; i<height; i++){
	for(int j=0; j<width; j++){
	    h_imageInput[(i*width+j)]=image.data[(i*width+j)*3]+image.data[(i*width+j)*3]+image.data[(i*width+j)*3];
	}    
    }*/
    h_imageInput = image.data;    
    start_gpu=clock();
    cudaMemcpy(d_imageInput,h_imageInput,size,cudaMemcpyHostToDevice);

    int blockSize=32;
    dim3 dimBlock(blockSize,blockSize,1);
    dim3 dimGrid(ceil(width/blockSize),ceil(height/blockSize),1);
    img2gray<<<dimGrid,dimBlock>>>(d_imageInput,width,height,d_imageOutput);
    cudaDeviceSynchronize();
    cudaMemcpy(h_imageOutput,d_imageOutput,tama,cudaMemcpyDeviceToHost);
    end_gpu=clock();
    //-------------------------

    gray_image.create(height,width,CV_8UC1);
    gray_image.data = h_imageOutput;

    GaussianBlur( image, image, Size(3,3), 0, 0, BORDER_DEFAULT );

    cvtColor(image,image_gray_opencv, CV_BGR2GRAY);
    

    //----------------Algoritmo sobel en GPU

    //---------------------------------------------------

    //---Algoritmo de sobel en CPU-------------
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    
    //------- Gradient X
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    Sobel( image_gray_opencv, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );

    //--------- Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel( image_gray_opencv, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );

    //-------- Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
    //----------------------------------------------------------------


    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", image);                   // Show our image inside it.

    imshow("Gray Image Secuencial", image_gray);
    imshow("Gray Image Sobel CPU", grad );
    imshow("Gray Image Parallel", gray_image);
    waitKey(0);

    image_gpu_time=((double)(end_gpu-start_gpu))/CLOCKS_PER_SEC;
    image_cpu_time=((double)(end-start))/CLOCKS_PER_SEC;
    printf("Tiempo algoritmo secuencial: %f", image_cpu_time);
    printf("Tiempo algoritmo paralelo: %f", image_gpu_time);
    //imshow("Gray Image OpenCV",image_gray_opencv);
    //imshow("Gray Image Sobel GPU",image_sobel);  
    //imshow("Gray Image CUDA parallel",h_imageOutput);

    //free(gray_width);
    //free(gray_height);
    free(gray);
    free(d_imageInput);
    free(d_imageOutput);
    cudaFree(h_imageOutput);
    cudaFree(h_imageInput);
    return 0;
}
