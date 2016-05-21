#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <cuda.h>

#define RED 2
#define GREEN 1
#define BLUE 0

#define MASK_SIZE 3
#define tam_mask 3
#define TILE_SIZE 32

using namespace cv;
using namespace std;

__constant__ char mask[tam_mask * tam_mask];

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

__device__ unsigned char clamp(int valor){
		if(valor < 0){
				valor=0;
		}else if(valor > 255){
				valor = 255;
		}
		
		return valor;
}

__global__ void union_img(unsigned char *d_sobel_x, unsigned char *d_sobel_y, unsigned char *d_sobelOut, int width, int height){
		
		int row = blockIdx.y*blockDim.y+threadIdx.y;
		int col = blockIdx.x*blockDim.x+threadIdx.x;
		
		if((row<height)&&(col<width)){
				d_sobelOut[row*width+col]=clamp(sqrtf((d_sobel_x[row*width+col]*d_sobel_x[row*width+col])+(d_sobel_y[row*width+col]*d_sobel_y[row*width+col])));
		}
}

__global__ void sobelShareMemTest(unsigned char *In, unsigned char *Out, int maskWidth,int width, int height){
	
		__shared__ float N_ds[TILE_SIZE + MASK_SIZE - 1][TILE_SIZE+ MASK_SIZE - 1];
		int n = MASK_SIZE/2;
		int dest = threadIdx.y*TILE_SIZE+threadIdx.x, destY = dest / (TILE_SIZE+MASK_SIZE-1), destX = dest % (TILE_SIZE+MASK_SIZE-1),
		srcY = blockIdx.y * TILE_SIZE + destY - n, srcX = blockIdx.x * TILE_SIZE + destX - n,
		src = (srcY * width + srcX);
		
		if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
			N_ds[destY][destX] = In[src];
		else
			N_ds[destY][destX] = 0;

		dest = threadIdx.y * TILE_SIZE + threadIdx.x + TILE_SIZE * TILE_SIZE;
		destY = dest /(TILE_SIZE + MASK_SIZE - 1), destX = dest % (TILE_SIZE + MASK_SIZE - 1);
		srcY = blockIdx.y * TILE_SIZE + destY - n;
		srcX = blockIdx.x * TILE_SIZE + destX - n;
		src = (srcY * width + srcX);
	
		if (destY < TILE_SIZE + MASK_SIZE - 1) {
			if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
				N_ds[destY][destX] = In[src];
			else
				N_ds[destY][destX] = 0;
			}
		__syncthreads();
		int accum = 0;
		int y, x;
		for (y = 0; y < maskWidth; y++)
			for (x = 0; x < maskWidth; x++)
				accum += N_ds[threadIdx.y + y][threadIdx.x + x] * mask[y * maskWidth + x];
			y = blockIdx.y * TILE_SIZE + threadIdx.y;
			x = blockIdx.x * TILE_SIZE + threadIdx.x;
			if (y < height && x < width)
				Out[(y * width + x)] = clamp(accum);
			__syncthreads();			
}

//--------
void sobel(unsigned char *imageIn,unsigned char *imageOut,int mask_width, int width, int height){
}

int main( int argc, char** argv )
{
    unsigned char *gray, *image_aux;
    //cudaError_t error = cudaSuccess;
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
    

    //----------------Algoritmo sobel en GPU--------------------
		
		char mask_x[]={-1,0,1,-2,0,2,-1,0,1};
		char mask_y[]={1,2,1,0,0,0,-1,-2,-1};
		
		int img_rgb = sizeof(unsigned char)*width*height*image.channels();
		int img_gray = sizeof(unsigned char)*width*height;
		int mascara = sizeof(char)*(tam_mask * tam_mask);
		char *d_mask;
		unsigned char *d_imageIn,*d_imageOut,*d_sobelOut,*d_sobel_x,*d_sobel_y;
		unsigned char *h_imageIn;
		
		h_imageIn = (unsigned char*)malloc(img_rgb);
		h_imageIn = image.data;
		
		cudaMalloc((void**)&d_imageIn,img_rgb);
		cudaMalloc((void**)&d_imageOut,img_gray);
		cudaMalloc((void**)&d_sobelOut,img_gray);
		cudaMalloc((void**)&d_sobel_x,img_gray);
		cudaMalloc((void**)&d_sobel_y,img_gray);
		cudaMalloc((void**)&d_mask,mascara);
		
		cudaMemcpy(d_imageIn,h_imageIn,img_rgb,cudaMemcpyHostToDevice);
		cudaMemcpy(d_mask,mask_x,img_gray,cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(mask,mask_x,mascara);	
		
		sobelShareMemTest<<<dimGrid,dimBlock>>>(d_imageOutput,d_sobel_x,3,width,height);
		cudaDeviceSynchronize();
		
		cudaMemcpy(d_mask,mask_y,img_gray,cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(mask,mask_y,mascara);		
		sobelShareMemTest<<<dimGrid,dimBlock>>>(d_imageOutput,d_sobel_y,3,width,height);
		cudaDeviceSynchronize();
		
		union_img<<<dimGrid,dimBlock>>>(d_sobel_x,d_sobel_y,d_sobelOut,width,height);
		cudaMemcpy(d_imageOut,d_sobelOut,img_gray,cudaMemcpyDeviceToHost);
		
		Mat image_sobel;
		image_sobel.create(height,width,CV_8UC1);
    image_sobel.data = d_imageOut;
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
    namedWindow("Gray Image secuencial", WINDOW_AUTOSIZE);
    namedWindow("Gray Image Sobel CPU", WINDOW_AUTOSIZE);
    namedWindow("Gray Image Parallel", WINDOW_AUTOSIZE);
    namedWindow("Gray Image Sobel GPU", WINDOW_AUTOSIZE);
    
    imshow( "Display window", image);                   // Show our image inside it.
    imshow("Gray Image Secuencial", image_gray);
    imshow("Gray Image Sobel CPU", grad );
    imshow("Gray Image Parallel", gray_image);
    imshow("Gray Image Sobel GPU",image_sobel);  
    waitKey(0);

    image_gpu_time=((double)(end_gpu-start_gpu))/CLOCKS_PER_SEC;
    image_cpu_time=((double)(end-start))/CLOCKS_PER_SEC;
    printf("Tiempo algoritmo secuencial: %f \n", image_cpu_time);
    printf("Tiempo algoritmo paralelo: %f \n", image_gpu_time);

    //free(gray_width);
    //free(gray_height);
    free(gray);
    free(d_imageInput);
    free(d_imageOutput);
    cudaFree(h_imageOutput);
    cudaFree(h_imageInput);
    return 0;
}
