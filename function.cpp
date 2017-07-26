#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv\highgui.h>
#include <opencv2\contrib\contrib.hpp>
#include "Image.h"
#include <vector>
Image::Image (cv::Mat &img)
{
	//if (img.size().height>500)
		//resize(img, img, cv::Size(img.size().width/2,img.size().height/2));

	imgen = img;
	std::string face_cascade_name = "C:/opencv/haarcascades/haarcascade_frontalface_alt.xml";
	cv::CascadeClassifier face_cascade;

	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading face cascade\n"); };

	cv::Mat imgGray;
	cvtColor(imgen, imgGray, cv::COLOR_BGR2GRAY);
	face_cascade.detectMultiScale(imgGray, rec_faces, 1.1, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
	//std::cout << rec_faces.size() << std::endl;
	for (size_t i = 0; i < rec_faces.size(); i++)
	{
		cv::Point p1(rec_faces[i].x, rec_faces[i].y);
		cv::Point p2(rec_faces[i].x + rec_faces[i].width, rec_faces[i].y + rec_faces[i].height);
		cv::Mat fase;
		cv::resize(imgGray(cv::Rect(p1, p2)), fase, cv::Size(100,100));
		faces.push_back(fase);
		labels.push_back(i);
		
	}
}
int Image::DetectWithExtraPoint(Image &img)
{
	std::vector<cv::KeyPoint> keypoints_main;
	std::vector<cv::KeyPoint> keypoints_object;
	int minHessian = 40;
	cv::SurfFeatureDetector detector( minHessian );
	resize(faces[0], faces[0], cv::Size(400,400*faces[0].size().height/faces[0].size().width));
    resize(img.Faces()[0], img.Faces()[0], cv::Size(400,400*img.Faces()[0].size().height/img.Faces()[0].size().width));
	detector.detect(faces[0], keypoints_object);
	detector.detect(img.Faces()[0], keypoints_main);
	cv::Mat img_keypoints_1;
	cv::Mat img_keypoints_2;
	
	cv::drawKeypoints( faces[0], keypoints_object, img_keypoints_1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );
	cv::drawKeypoints( img.Faces()[0], keypoints_main, img_keypoints_2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT );

  //-- Show detected (drawn) keypoints
    imshow("Keypoints 1", img_keypoints_1 );
    imshow("Keypoints 2", img_keypoints_2 );


	cv::SurfDescriptorExtractor extractor;
	cv::Mat desctr_main, desctr_object;
	extractor.compute(faces[0], keypoints_main, desctr_main);
	extractor.compute(img.Faces()[0], keypoints_object, desctr_object);
	cv::FlannBasedMatcher matcher;
	std::vector<cv::DMatch> matches;
	matcher.match(desctr_main, desctr_object, matches);
	double max_dist = 0;
	double min_dist = 100;
	for (int i = 0; i < desctr_main.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	std::cout << "Max dist: " << max_dist;
	std::cout << "Min dist: " << min_dist;
	std::vector<cv::DMatch> good_matches;
	for (int i = 0; i < desctr_main.rows; i++)
	{
		if (matches[i].distance < 3 * min_dist) good_matches.push_back(matches[i]);
	}
	cv::Mat img_matches;
	drawMatches(faces[0], keypoints_main, img.Faces()[0], keypoints_object, good_matches, img_matches,
		cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	std::vector<cv::Point2f> obj;
	std::vector<cv::Point2f> scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		obj.push_back(keypoints_main[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_object[good_matches[i].trainIdx].pt);
	}
	std::vector<cv::Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(faces[0].cols, 0);
	obj_corners[2] = cvPoint(faces[0].cols, faces[0].rows); obj_corners[3] = cvPoint(0, faces[0].rows);
	std::vector<cv::Point2f> scene_corners(4);

	cv::Mat H = cv::findHomography(obj, scene, CV_RANSAC);
	
	//-- ќтобразить углы целевого объекта, использу€ найденное преобразование, на сцену
	perspectiveTransform(obj_corners, scene_corners, H);

	//-- —оеденить отображенные углы
	line(img_matches, scene_corners[0] + cv::Point2f(faces[0].cols, 0), scene_corners[1] + cv::Point2f(faces[0].cols, 0), cv::Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[1] + cv::Point2f(faces[0].cols, 0), scene_corners[2] + cv::Point2f(faces[0].cols, 0), cv::Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[2] + cv::Point2f(faces[0].cols, 0), scene_corners[3] + cv::Point2f(faces[0].cols, 0), cv::Scalar(0, 255, 0), 4);
	line(img_matches, scene_corners[3] + cv::Point2f(faces[0].cols, 0), scene_corners[0] + cv::Point2f(faces[0].cols, 0), cv::Scalar(0, 255, 0), 4);

	//-- Show detected matches
	//imshow("Good Matches & Object detection", img_matches);


	//img_matches.resize(500);
	cv::Mat temp;

	resize(img_matches, temp, cv::Size(400,400*faces[0].size().height/faces[0].size().width));
    //}
	// окно дл€ отображени€ картинки
	//cvNamedWindow("original", CV_WINDOW_NORMAL);
	//cvNamedWindow("Good Matches & Object detection", CV_WINDOW_NORMAL);
	//IplImage img_itog((IplImage)img_matches);
	//newImage = cvCreateImage(cvSize((&img_itog)->width / 4, (&img_itog)->height / 4), (&img_itog)->depth, 1);
	//cvResize(newImage, newImage);
	cvNamedWindow("original", CV_WINDOW_NORMAL);
	imshow("original", temp);
	
	//cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(minHessian);*/
	return 0;
}
int Image::DetectFace(Image &img)
{
	cv::Ptr<cv::FaceRecognizer> model = cv::createFisherFaceRecognizer();

	model->train(Faces(), Labels());
	int a = model->predict(img.Faces()[0]);
	return a;
}

void Image::ShowImage()
{
	resize(imgen, imgen, cv::Size(400,400*imgen.size().height/imgen.size().width));

	imshow("original", imgen);

}
void Image::ShowFace(int &i)
{
	cv::Point p1(rec_faces[i].x, rec_faces[i].y);
	cv::Point p2(rec_faces[i].x + rec_faces[i].width, rec_faces[i].y + rec_faces[i].height);
	rectangle(imgen, p1, p2, cv::Scalar(255, 0, 0));
	resize(imgen, imgen, cv::Size(1000,1000*imgen.size().height/imgen.size().width));

	imshow("predict", imgen);
}
std::vector<cv::Mat> Image::Faces()
{
	return faces;
}
std::vector<int> Image::Labels()
{
	return labels;
}