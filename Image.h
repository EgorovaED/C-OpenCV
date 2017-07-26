#ifndef IMAGE_H
#define IMAGE_H
class Image
{
	private:
		cv::Mat imgen;
		std::vector<cv::Mat> faces;
		std::vector<int> labels;
		std::vector<cv::Rect> rec_faces;
	public:
		Image (cv::Mat &img);
		void ShowImage();
		void ShowFace(int &i);
		int DetectFace(Image &img);
		std::vector<cv::Mat> Faces();
		std::vector<int> Labels();
		int DetectWithExtraPoint(Image &img);
};
#endif