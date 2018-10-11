/*Before trying to construct hybrid images, it is suggested that you
implement myFilter() and then debug it using proj1_test_filtering.cpp */


#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

Mat myFilter(Mat, Mat, int);
Mat hybrid_image_visualize(Mat);
Mat DFT_Spectrum(Mat);



enum border { Border_Replicate, Border_Reflect, Border_Constant };

Mat myFilter(Mat im, Mat filter, int borderType = Border_Constant)
{
	/*This function is intended to behave like the built in function filter2D()

	Your function should work for color images. Simply filter each color
	channel independently.

	Your function should work for filters of any width and height
	combination, as long as the width and height are odd(e.g. 1, 7, 9).This
	restriction makes it unambigious which pixel in the filter is the center
	pixel.

	Boundary handling can be tricky.The filter can't be centered on pixels
	at the image boundary without parts of the filter being out of bounds.
	There are several options to deal with boundaries. Your code should be
	able to handle the border types defined above as the following enum types:
	* Border_Replicate:     aaaaaa|abcdefgh|hhhhhhh
	* Border_Reflect:       fedcba|abcdefgh|hgfedcb
	* Border_Constant:      iiiiii|abcdefgh|iiiiiii  with 'i=0'
	(image boundaries are denoted with '|')

	*/

	Mat outI;

	// initilize our result with zeros
	outI = Mat::zeros(im.rows, im.cols, im.type());

	int filterHeight = filter.rows / 2;
	int filterWidth = filter.cols / 2;
	int numRows = im.rows;
	int numCols = im.cols;
	int channels = im.channels();

	// Our images are CV_64FC3
	// 64bit floating point (double) type, with 3 channels
	cout << "filter = " << endl << " " << filter << endl << endl;
	//cout << "img[0] = " << endl;
	//for (int i = 0; i < im.cols * im.channels(); i++) {
	//	cout << " " << im.ptr<double>(0)[i] << " ";
	//}

	// Mat to store the values we will multiply by filter
	Mat frame;

	frame = Mat::zeros(filter.rows, filter.cols, CV_64F);

	// pointer to relevant row of image
	double* My = NULL;

	// Note that we have to separate different channels and combine them together
	// Equivalently we can handle this by locating neighbor pixels with distanced by 3 instead of 1
	int indexX, indexY;
	bool indicatorX, indicatorY;

	for (int m = 0; m < numRows; m++) {
		for (int n = 0; n < numCols * channels; n++) {
			for (int j = -filterHeight; j <= filterHeight; j++) {
				indexY = m + j;
				indicatorY = false;
				if (indexY < 0 || indexY >= numRows) {
					if (borderType == Border_Replicate) {
						if (indexY < 0) {
							indexY = 0;
						}
						else {
							indexY = numRows - 1;
						}
					}
					else if (borderType == Border_Reflect) {
						if (indexY < 0) {
							indexY = -indexY - 1;
						}
						else {
							indexY =  2 * numRows - 1  - indexY;
						}
					}
					else {
						// make indicator true to indicate we will put 0 to frame array
						indicatorY = true;
					}
				}
				if (!indicatorY)
					My = im.ptr<double>(indexY);
				for (int i = -filterWidth; i <= filterWidth; i++) {
					indexX = n + i * channels;
					indicatorX = false;
					if (indexX < 0 || indexX >= numCols * channels) {
						if (borderType == Border_Replicate) {
							if (indexX < 0) {
								indexX = indexX % channels;
							}
							else {
								indexX = (indexX) % channels + (numCols - 1) * channels;
							}
						}
						else if (borderType == Border_Reflect) {
							if (indexX < 0) {
								indexX = (indexX % channels) + ((indexX + 1) / -channels) * channels;
							}
							else {
								indexX = (indexX % channels) + (numRows - 1) * channels - ((indexX - numRows) / channels) * channels;
							}
						}
						else {
							// make indicator true to indicate we will put 0 to frame array
							indicatorX = true;
						}
					}
					if (indicatorX || indicatorY) {
						frame.at<double>(j + filterHeight, i + filterWidth) = 0;
						//frame.ptr<double>(j + filterHeight)[i + filterWidth] = 0;
					}
					else {
						frame.at<double>(j + filterHeight, i + filterWidth) = My[indexX];
						//frame.ptr<double>(j + filterHeight)[i + filterWidth] = My[indexX];
					}
				}
			}
			//cout << "frame = " << endl << " " << frame << endl << endl;
			//cout << "filter = " << endl << " " << filter << endl << endl;
			multiply(filter, frame, frame);
			//cout << "frame after mult= " << endl << " " << frame << endl << endl;
			double res = sum(frame)[0];
			//cout << res << " ";
			outI.ptr<double>(m)[n] = res;
		}
	}
	return outI;

}


Mat hybrid_image_visualize(Mat hybrid_image)
{
	//visualize a hybrid image by progressively downsampling the image and
	//concatenating all of the images together.		
	int scales = 5; //how many downsampled versions to create		
	double scale_factor = 0.5; //how much to downsample each time		
	int padding = 5; //how many pixels to pad.
	int original_height = hybrid_image.rows; // height of the image
	int num_colors = hybrid_image.channels(); //counting how many color channels the input has
	Mat output = hybrid_image;
	Mat cur_image = hybrid_image;

	for (int i = 2; i <= scales; i++)
	{
		//add padding
		hconcat(output, Mat::ones(original_height, padding, CV_8UC3), output);

		//dowsample image;
		resize(cur_image, cur_image, Size(0, 0), scale_factor, scale_factor, INTER_LINEAR);

		//pad the top and append to the output
		Mat tmp;
		vconcat(Mat::ones(original_height - cur_image.rows, cur_image.cols, CV_8UC3), cur_image, tmp);
		hconcat(output, tmp, output);
	}

	return output;
}

Mat DFT_Spectrum(Mat img)
{
	/*
	This function is intended to return the spectrum of an image in a displayable form. Displayable form
	means that once the complex DFT is calculated, the log magnitude needs to be determined from the real 
	and imaginary parts. Furthermore the center of the resultant image needs to correspond to the origin of the spectrum.
	*/

	vector<Mat> im_channels(3);
	split(img, im_channels);
	img = im_channels[0];

	/////////////////////////////////////////////////////////////////////
	//STEP 1: pad the input image to optimal size using getOptimalDFTSize()
	
	//Write your code here
	int optimalCol = getOptimalDFTSize(img.cols);
	int optimalRow = getOptimalDFTSize(img.rows);

	//add padding
	copyMakeBorder(img, img, 0, optimalRow - img.rows, 0, optimalCol - img.cols, BORDER_CONSTANT);

	///////////////////////////////////////////////////////////////////
	//STEP 2:  Determine complex DFT of the image. 
	// Use the function dft(src, dst, DFT_COMPLEX_OUTPUT) to return a complex Mat variable.
	// The first dimension represents the real part and second dimesion represents the complex part of the DFT 
	
	//Write your code here
	dft(img, img, DFT_COMPLEX_OUTPUT);
	
	

	////////////////////////////////////////////////////////////////////
	//Step 3: compute the magnitude and switch to logarithmic scale
	//=> log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	
	Mat magI;	
	//Write your code here
	vector<Mat> im_dims(2);
	split(img, im_dims); // im_dims[0] = Re(DFT(I)), im_dims[1] = Im(DFT(I))
	magnitude(im_dims[0], im_dims[1], magI); // OpenCV method to calculate the magnitude
	magI += Scalar::all(1); // Add one to switch log scale
	log(magI, magI);

	///////////////////////////////////////////////////////////////////
	// Step 4: 
	/* For visualization purposes the quadrants of the spectrum are rearranged so that the 
	   origin (zero, zero) corresponds to the image center. To achieve this swap the top left
	   quadrant with bottom right quadrant, and swap the top right quadrant with bottom left quadrant
	*/

	//crop the spectrum, if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
	
	//Write your code here
	 // rearrange the quadrants of Fourier image  so that the origin is at the image center
	int quadrantCols = magI.cols / 2;
	int quadrantRows = magI.rows / 2;

	Mat topLeftQuadrant(magI, Rect(0, 0, quadrantCols, quadrantRows));
	Mat topRightQuadrant(magI, Rect(quadrantCols, 0, quadrantCols, quadrantRows));
	Mat bottomLeftQuadrant(magI, Rect(0, quadrantRows, quadrantCols, quadrantRows));
	Mat bottomRightQuadrant(magI, Rect(quadrantCols, quadrantRows, quadrantCols, quadrantRows));

	// swap quadrants
	Mat tmp;                           
	topLeftQuadrant.copyTo(tmp);
	bottomRightQuadrant.copyTo(topLeftQuadrant);
	tmp.copyTo(bottomRightQuadrant);
	topRightQuadrant.copyTo(tmp);
	bottomLeftQuadrant.copyTo(topRightQuadrant);
	tmp.copyTo(bottomLeftQuadrant);

	// Transform the matrix with float values into a viewable image form (float between values 0 and 1).
	normalize(magI, magI, 0, 1, CV_MINMAX);
	return magI;
}

int main()
{
	//Read images
	// Changed "../data/dob.bmp" 
	//Mat image1 = imread("./data/yyemez.jpg");
	Mat image1 = imread("./data/dog.bmp");
	if (!image1.data)                              // Check for invalid image
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	// Changed "../data/cat.bmp" 
	Mat image2 = imread("./data/marilyn.bmp");
	//Mat image2 = imread("./data/deneme.png");
	if (!image2.data)                              // Check for invalid image
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	image1.convertTo(image1, CV_64FC3);
	image2.convertTo(image2, CV_64FC3);


	/*Several additional test cases are provided for you, but feel free to make
	your own(you'll need to align the images in a photo editor such as
	Photoshop).The hybrid images will differ depending on which image you
	assign as image1(which will provide the low frequencies) and which image
	you asign as image2(which will provide the high frequencies) */


	//========================================================================
	//							   PART 1 
	//========================================================================

	// IMPLEMENT THE FUNCTION myFilter(Mat,Mat,int) 
	// THIS FUNCTION TAKES THREE ARGUMENTS. FIRST ARGUMENT IS THE MAT IMAGE, 
	// SECOND ARGUMENT IS THE MAT FILTER AND THE THIRD ARGUMENT SPECIFIES THE
	// PADDING TYPE



	//========================================================================
	//							   PART2
	//========================================================================
	////  FILTERING AND HYBRID IMAGE CONSTRUCTION  ////

	int cutoff_frequency = 7; // yyemez 5
	/*This is the standard deviation, in pixels, of the
	Gaussian blur that will remove the high frequencies from one image and
	remove the low frequencies from another image (by subtracting a blurred
	version from the original version). You will want to tune this for every
	image pair to get the best results.*/

	Mat filter = getGaussianKernel(cutoff_frequency * 4 + 1, cutoff_frequency, CV_64F);
	filter = filter*filter.t();



	// YOUR CODE BELOW. 
	// Use myFilter() to create low_frequencies of image 1. The easiest
	// way to create high frequencies of image 2 is to subtract a blurred
	// version of image2 from the original version of image2. Combine the
	// low frequencies and high frequencies to create 'hybrid_image'


	Mat low_freq_img;
	low_freq_img = myFilter(image1,filter, Border_Constant);

	Mat high_freq_img;
	high_freq_img = image2 - myFilter(image2, filter, Border_Constant);

	Mat hybrid_image;
	hybrid_image = low_freq_img + high_freq_img;


	////  Visualize and save outputs  ////	
	//add a scalar to high frequency image because it is centered around zero and is mostly black	
	high_freq_img = high_freq_img + Scalar(0.5, 0.5, 0.5) * 255;
	//Convert the resulting images type to the 8 bit unsigned integer matrix with 3 channels
	high_freq_img.convertTo(high_freq_img, CV_8UC3);
	low_freq_img.convertTo(low_freq_img, CV_8UC3);
	hybrid_image.convertTo(hybrid_image, CV_8UC3);

	Mat vis = hybrid_image_visualize(hybrid_image);

	imshow("Low frequencies", low_freq_img); //waitKey(0);
	imshow("High frequencies", high_freq_img);	//waitKey(0);
	imshow("Hybrid image", vis); //waitKey(0);


	imwrite("low_frequencies.jpg", low_freq_img);
	imwrite("high_frequencies.jpg", high_freq_img);
	imwrite("hybrid_image.jpg", hybrid_image);
	imwrite("hybrid_image_scales.jpg", vis);

	//============================================================================
	//							PART 3
	//============================================================================
	//In this part determine the DFT of just one channel of image1 and image2, as well 
	// as the DFT of the low frequency image and high frequency image.

	//Complete the code for DFT_Spectrum() method

	Mat img1_DFT = DFT_Spectrum(image1);
	imshow("Image 1 DFT", img1_DFT); waitKey(0);
	imwrite("Image1_DFT.jpg", img1_DFT * 255);

	low_freq_img.convertTo(low_freq_img, CV_64FC3);
	Mat low_freq_DFT = DFT_Spectrum(low_freq_img);
	imshow("Low Frequencies DFT", low_freq_DFT); waitKey(0);
	imwrite("Low_Freq_DFT.jpg", low_freq_DFT * 255);

	Mat img2_DFT = DFT_Spectrum(image2);
	imshow("Image 2 DFT", img2_DFT); waitKey(0);
	imwrite("Image2_DFT.jpg", img2_DFT * 255);

	high_freq_img.convertTo(high_freq_img, CV_64FC3);
	Mat high_freq_DFT = DFT_Spectrum(high_freq_img);
	imshow("High Frequencies DFT", high_freq_DFT); waitKey(0);
	imwrite("High_Freq_DFT.jpg", high_freq_DFT * 255);

}