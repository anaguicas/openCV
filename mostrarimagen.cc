#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cuda.h>
#include <cv.h>

using namespace cv;
using namespace std;

#define RED 2
#define GREEN 1
#define BLUE 0

__gloabl__ void grayImage(unsigned char *d_dataRawImage,int width, int height, unsigned char *d_imageOutput){
	int col=blockIdx.x*blockDim.x + threadIdx.x;
	int row=blockIdx.y*blockDim.y + threadIdx.y;
	
	if((row < height) && (col < width)){
        d_imageOutput[row*width+col] = imageInput[(row*width+col)*3+RED]*0.299 + imageInput[(row*width+col)*3+GREEN]*0.587 \
                                     + imageInput[(row*width+col)*3+BLUE]*0.114;
    }
}

int main( int argc, char** argv )
{
    unsigned char *gray, *image_aux;
    unsigned char *d_dataRawImage, *d_imageOutput, *h_imageOutput;
    //uchar4* rgba;
    int width, height, gray_width, gray_height;
    Mat src, src_gray;
    Mat grad;
    
    int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;

  int c;
  
    //infoImage image_h;

    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

    Mat image;
    Mat image_gray_opencv;
    Mat image_gray;

    image = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    
    Size s = image.size();
    width = s.width;
    height = s.height;
    int tama = sizeof(unsigned char)*width*height;
    gray = (unsigned char*)malloc(tama);


    for(int i=0; i<height; i++){
	for(int j=0; j<width; j++){
		gray[(i*width+j)]= 0.299*image.data[(i*width+j)*3+2] + 0.587*image.data[(i*width+j)*3+1] + 0.114*image.data[(i*width+j)*3];
	}    
    }
    
    error = cudaMalloc((void**)&d_dataRawImage,size);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_dataRawImage\n");
        exit(-1);
    }
    
    error = cudaMalloc((void**)&d_imageOutput,sizeGray);
    if(error != cudaSuccess){
        printf("Error reservando memoria para d_imageOutput\n");
        exit(-1);
    }
    
    error = cudaMemcpy(d_dataRawImage,gray,size, cudaMemcpyHostToDevice);
    if(error != cudaSuccess){
        printf("Error copiando los datos de dataRawImage a d_dataRawImage \n");
        exit(-1);
    }
    
    int blockSize = 32;
    dim3 dimBlock(blockSize,blockSize,1);
    dim3 dimGrid(ceil(width/float(blockSize)),ceil(height/float(blockSize)),1);
    grayImage<<<dimGrid,dimBlock>>>(d_dataRawImage,width,height,d_imageOutput);
    cudaDeviceSynchronize();
    cudaMemcpy(h_imageOutput,d_imageOutput,sizeGray,cudaMemcpyDeviceToHost);
    
    image_gray.create(height,width,CV_8UC1);
    image_gray.data=gray;
    
    GaussianBlur( image, image, Size(3,3), 0, 0, BORDER_DEFAULT );

    cvtColor(image,image_gray_opencv, CV_BGR2GRAY);

    imwrite("./Gray_Image.jpg",image_gray);
    
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    
    /// Gradient X
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    Sobel( image_gray_opencv, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );

    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel( image_gray_opencv, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );

    /// Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
  
    imshow("Gray Image Sobel", grad );
    
    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", image);                   // Show our image inside it.

    imshow("Gray Image CUDA", image_gray);
    imshow("Gray Image OpenCV",image_gray_opencv);
    
    imshow("Gray Image CUDA parallel",h_imageOutput)

    waitKey(0);                                          // Wait for a keystroke in the window
    //free(gray_width);
    //free(gray_height);
    free(gray);
    return 0;
}
