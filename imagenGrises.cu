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

__device__ unsigned char clamp(int value){
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return (unsigned char)value;
}

__global__ void Union(unsigned char *Sobel_X, unsigned char *Sobel_Y, int width, int height, unsigned char *imageOutput){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if((row < height) && (col < width)){
      imageOutput[row*width+col]= clamp(sqrtf( Sobel_X[row*width+col]*Sobel_X[row*width+col]+Sobel_Y[row*width+col]*Sobel_Y[row*width+col]) );
    }
}

//------------Función sobel GPU----------------
__global__ void Sobel(unsigned char *imageInput,int *mask, int width, int height, unsigned char *imageOutput){
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    int aux=row*width+col;
    int sum=0;
    if((row < height) && (col < width)){
        if(( aux-width-2) > 0 ){
            sum += mask[0]*imageInput[aux-width-2];
        }
        if((aux-1) > 0){
            sum += mask[1]*imageInput[aux-width-1];
        }
        if(aux-width > 0){
            sum += mask[2]*imageInput[aux-width];
        }
        //------------------------------------
        if(aux-1 > 0){
            sum += mask[3]*imageInput[aux-1];
        }

        sum += mask[4]*imageInput[aux];

        if(aux+1 < width*height){
            sum += mask[5]*imageInput[aux+1];
        }
        //---------------------------------
        if(aux+width < width*height){
            sum += mask[6]*imageInput[aux+width];
        }
        if(aux+width+1 < width*height){
            sum += mask[7]*imageInput[aux+width+1];
        }
        if(aux+width+2 < width*height){
            sum += mask[8]*imageInput[aux+width+2];
        }
   
        imageOutput[row*width+col]= clamp(sum);
    }
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
    
    for(int i=0; i<height; i++){
	for(int j=0; j<width; j++){
	    h_imageInput[(i*width+j)]=image.data[(i*width+j)*3+2]+image.data[(i*width+j)*3+1]+image.data[(i*width+j)*3];
	}    
    }
//    h_imageInput=image.data;    
    start_gpu=clock();
    cudaMemcpy(d_imageInput,h_imageInput,size,cudaMemcpyHostToDevice);

    int blockSize=32;
    dim3 dimBlock(blockSize,blockSize,1);
    dim3 dimGrid(ceil(width/float(blockSize)),ceil(height/float(blockSize)),1);
    img2gray<<<dimGrid,dimBlock>>>(d_imageInput,width,height,d_imageOutput);
    cudaDeviceSynchronize();
    cudaMemcpy(h_imageOutput,d_imageOutput,tama,cudaMemcpyDeviceToHost);
    end_gpu=clock();
    //-------------------------

    Mat gray_image;
    gray_image.create(height,width,CV_8UC1);
    gray_image.data = h_imageOutput;

    GaussianBlur( image, image, Size(3,3), 0, 0, BORDER_DEFAULT );

    cvtColor(image,image_gray_opencv, CV_BGR2GRAY);
    

    //----------------Algoritmo sobel en GPU

    /*error = cudaMalloc((void**)&d_SobelOutput_X,size);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_SobelOutput_X\n");
        exit(-1);
    }
    error = cudaMalloc((void**)&d_SobelOutput_Y,size);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_SobelOutput_Y\n");
        exit(-1);
    }
    error = cudaMalloc((void**)&d_SobelOutput,size);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_SobelOutput\n");
        exit(-1);
    }
    error = cudaMalloc((void**)&h_SobelOutput,size);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_SobelOutput\n");
        exit(-1);
    }

    int * Mask_x = (int*)malloc( 3*3*sizeof(int) ); //separo memoria en host para la mascara en X
    Mask_x[0]=-1;Mask_x[1]=0;Mask_x[2]=1;
    Mask_x[3]=-2;Mask_x[4]=0;Mask_x[5]=2;
    Mask_x[6]=-1;Mask_x[7]=0;Mask_x[8]=1;

    int * Mask_y = (int*)malloc( 3*3*sizeof(int) ); //separo memoria en host para la mascara en y
    Mask_y[0]=-1;Mask_y[1]=-2;Mask_y[2]=-1;
    Mask_y[3]=0;Mask_y[4]=0;Mask_y[5]=0;
    Mask_y[6]=1;Mask_y[7]=2;Mask_y[8]=1;

    int sizeM= 3*3*sizeof(int);
    int *d_M;

    cudaMalloc((void**)&d_M,sizeM);
    cudaMemcpy(d_M,Mask_x,sizeM,cudaMemcpyHostToDevice);
    Sobel<<<dimGrid,dimBlock>>>(gray,d_M,width,height,d_SobelOutput_X);
    cudaMemcpy(d_M,Mask_y,sizeM,cudaMemcpyHostToDevice);
    Sobel<<<dimGrid,dimBlock>>>(gray,d_M,width,height,d_SobelOutput_Y);
    cudaDeviceSynchronize();
    Union<<<dimGrid,dimBlock>>>(d_SobelOutput_X,d_SobelOutput_Y,width,height,d_SobelOutput);

    h_SobelOutput=(unsigned char*)malloc(size);	
    cudaMemcpy(h_SobelOutput,d_SobelOutput,size,cudaMemcpyDeviceToHost);

    Mat image_sobel;
    image_sobel.create(height,width,CV_8UC1);
    image_sobel.data=h_SobelOutput;*/

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

    printf("Tiempo algoritmo secuencial: ", ((double)(end-start))/CLOCKS_PER_SEC);
    printf("Tiempo algoritmo paralelo: ", ((double)(end_gpu-start_gpu))/CLOCKS_PER_SEC);
    //imshow("Gray Image OpenCV",image_gray_opencv);
    //imshow("Gray Image Sobel GPU",image_sobel);
    
    //imshow("Gray Image CUDA parallel",h_imageOutput);

    waitKey(0);                                          // Wait for a keystroke in the window
    //free(gray_width);
    //free(gray_height);
    free(gray);
    free(d_imageInput);
    free(d_imageOutput);
    cudaFree(h_imageOutput);
    cudaFree(h_imageInput);
    return 0;
}
