#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

struct infoImage
        {
            uchar4*         rgba;  
            unsigned char*  gray;    
        };


void ImageRGBToGray(int rows, int cols){
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            
        }
    }
}

int main( int argc, char** argv )
{
    unsigned char* gray;
    uchar4* rgba;
    int width, height, gray_width, gray_height;
    
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
    width = image.cols;
    height = image.rows;
    int size = sizeof(unsigned char)*width*height*image.channels();
    //ImageRGBToGray(rows,cols);    

    image_gray.create(height,width,CV_8UC1);

    gray_width = (unsigned char*)malloc(image_gray.cols);
    gray_height = (unsigned char*)malloc(image_gray.rows);    

    cvtColor(image, gray_image_opencv, CV_BGR2GRAY);

    //rgba = reinterpret_cast<uchar4*>(gray_image_opencv.ptr<unsigned char>(0));
    //gray = image_gray.ptr<unsigned char>(0);    

    gray = (unsigned char*)malloc(size);

    gray = image.data;

    const  int tam = image_gray.rows * image_gray.cols;

    unsigned char* n_image = new unsigned char[tam];  
    
    for(int r=0; r<gray_height; r++){
        for(int c=0; c<gray_width; c++){ 
            int indice=c*gray_width+r;
            float grays = 0.299*gray[indice] + 0.587*gray[indice] + 0.114*gray[indice];
            n_image[indice] = grays;
        }
    }

    image_gray.data = n_image;

    imwrite("./Gray_Image.jpg",image_gray);

    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", image );                   // Show our image inside it.

    imshow(imageName,image);
    imshow("Gray Image CUDA", image_gray);
    imshow("Gray Image OpenCV",gray_image_opencv);

    waitKey(0);                                          // Wait for a keystroke in the window
    free(gray_width);
    free(gray_height);
    free(gray);
    return 0;
}
