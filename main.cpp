#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv2\objdetect\objdetect.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <opencv2\imgproc\imgproc_c.h>
#include <opencv2\core\core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <map>
#include <opencv2\contrib\contrib.hpp>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <vector>
#include "Image.h"
#include <algorithm>
#include <time.h> 
#define ARRAY_ELEM 100000 

int main()
{
	 //read photos
	 std::string Firstfile;
	 std::cout << "Find face: ";
	 std::cin >> Firstfile;
	 std::cout << std::endl;
	 cv::Mat imagemation1 = cv::imread(Firstfile);
	 std::string Secondfile;
	 std::cout << "All faces: ";
	 std::cin >> Secondfile;
	 std::cout << std::endl;
	 cv::Mat imagemation2 = cv::imread(Secondfile);
	 bool Flag = true;
	 if (!imagemation1.data || !imagemation2.data)
		 Flag = false;
	 while(Flag == false )
     { 
		 
		 std::cout<< " --(!) Error reading images! Try again! " << std::endl; 
		 std::cout << "Find face: ";
	     std::cin >> Firstfile;
	     std::cout << std::endl; 
		 imagemation1 = cv::imread(Firstfile);
		 std::cout << "All faces: ";
	     std::cin >> Secondfile;
	     std::cout << std::endl;
		 imagemation2 = cv::imread(Secondfile);
		 Flag = true;
		 if ((!imagemation1.data) || (!imagemation2.data))
			Flag = false;
	 }

	 //time start
	float fTimeStart = clock()/(float)CLOCKS_PER_SEC; 
	
	Image img_main = Image(imagemation1);
	Image img_all = Image(imagemation2);
	int i = 2;
	std::vector<cv::Mat> r = img_all.Faces();
	int predict = img_all.DetectFace(img_main);
	img_main.ShowImage();
	img_all.ShowFace(predict);

	//time stop

	float fTimeStop = clock()/(float)CLOCKS_PER_SEC; 
    std::cout << fTimeStop-fTimeStart << std::endl; 
	cvWaitKey(0);

	
	system("pause");
	return 0;
}