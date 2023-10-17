#ifndef _FEATUREMATCHER_H_
#define _FEATUREMATCHER_H_

#include <opencv2/opencv.hpp>

//����������ӣ�Ŀǰ����siftЧ����ã�orb����̫�٣��ֲ�������
enum class FeatureName
{
	ORB,
	SIFT,
	SURF,
	KAZE,
	AKAZE,
	BRISK,
	BINBOOST,
	VGG
};


class FeatureMatcher
{
public:
	FeatureMatcher();
	~FeatureMatcher();

public:
	//��������������
	void createFeatures(
		FeatureName featureName,
		const int numKeypoints,
		cv::Ptr<cv::Feature2D> &detector,
		cv::Ptr<cv::Feature2D> &descriptor);

	//��������ƥ����
	cv::Ptr<cv::DescriptorMatcher> createMatcher(FeatureName featureName, bool useFLANN);

	//����ƥ���ɸѡ�����˵���ƥ��
	void ratioFilter_nn(std::vector<std::vector<cv::DMatch>> &matches, float ratio);

	//���ݶԳ���ɸѡ����
	void symmetryFilter_nn(
		const std::vector<std::vector<cv::DMatch> >& matches1,
		const std::vector<std::vector<cv::DMatch> >& matches2,
		std::vector<cv::DMatch>& symMatches);

	//Ѱ��ƥ��������
	void findMatchPoints(
		const cv::Mat &img1,
		const cv::Mat &img2,
		const cv::Mat &K1,
		const cv::Mat &K2,
		FeatureName featureName,
		const int numKeypoints,
		bool useFLANN,
		std::vector<cv::Point2f> &points1,
		std::vector<cv::Point2f> &points2);

	//����ɸѡ�ı�����ֵ
	void setRatioThresh(const float ratio);

	//ransac�ľ�����ֵ
	void setRansacDist(const float dist);

private:
	//normalize the pixel coordinates
	cv::Point2f pixelNorm(const cv::Point2f &p, const cv::Mat &K);

private:
	float ratioThresh_;
	double RANSAC_dist_;


};

#endif;

