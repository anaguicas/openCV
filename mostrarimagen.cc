#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define RED 2
#define GREEN 1
#define BLUE 0

int main( int argc, char** argv )
{
    unsigned char *gray, *image_aux;
    //uchar4* rgba;
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
    width = s.width;
    height = s.height;
    int tama = sizeof(unsigned char)*width*height;
    gray = (unsigned char*)malloc(tama);


    for(int i=0; i<height; i++){
	for(int j=0; j<width; j++){
		gray[(i*width+j)]= 0.299*image.data[(i*width+j)*3+2] + 0.587*image.data[(i*width+j)*3+1] + 0.114*image.data[(i*width+j)*3];
	}    
    }

    image_gray.create(height,width,CV_8UC1);
    image_gray.data=gray;

    cvtColor(image,image_gray_opencv, CV_BGR2GRAY);

    imwrite("./Gray_Image.jpg",image_gray);

    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", image);                   // Show our image inside it.

    imshow("Gray Image CUDA", image_gray);
    imshow("Gray Image OpenCV",image_gray_opencv);

    waitKey(0);                                          // Wait for a keystroke in the window
    //free(gray_width);
    //free(gray_height);
    free(gray);
    return 0;
}
