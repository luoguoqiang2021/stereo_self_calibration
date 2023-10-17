#ifndef _ESTIMATESOLVERS_H_
#define _ESTIMATESOLVERS_H_

#include <opencv2/opencv.hpp>

class EstimateSolvers
{
public:
	EstimateSolvers();

	EstimateSolvers(cv::Mat R, cv::Vec3d t) :R_init_(R), t_init_(t)
	{
		//Э�����������
		covRT_ = std::numeric_limits<double>::max();

		//ѵ������
		cost_ = std::numeric_limits<double>::max();

		//��������
		maxIterations_ = 5000;
	};

	~EstimateSolvers();

public:
	//Gram-Schmidt process
	cv::Vec3d vectorProjection(const cv::Vec3d &u, const cv::Vec3d &v);
	void findBasis(const cv::Vec3d &t, cv::Vec3d &b1, cv::Vec3d &b2);

	//estimate the R and t
	bool estimateRT(
		const std::vector<cv::Point2f> &points1,
		const std::vector<cv::Point2f> &points2,
		const double HuberThresh,
		const double toleranceError,
		cv::Mat &R, 
		cv::Vec3d &t);

	//set and get functions
	void setInitR(const cv::Mat R_init);
	void setInitT(const cv::Vec3d t_init);
	void setMaxIterations(const int iters);

	double getCovRT() const;
	double getCost() const;

private:
	cv::Mat R_init_;
	cv::Vec3d t_init_;
	double covRT_;
	double cost_;
	int maxIterations_;

};

#endif

