#ifndef _METRIC_H_
#define _METRIC_H_

#include <opencv2/opencv.hpp>

class Metric
{
public:
	Metric();
	~Metric();

	static void computeMeanAndStddev(const std::vector<double> &data, float &mean, float &stddev);

	// Computes the norm of the translation error
	static double get_translation_error(const cv::Mat &t_true, const cv::Mat &t);

	// Computes the norm of the rotation error
	static double get_rotation_error(const cv::Mat &R_true, const cv::Mat &R);

	// Y-Z-X
	static cv::Vec3d rot2euler(const cv::Mat & rotationMatrix);

	// Y-Z-X
	static cv::Mat euler2rot(const cv::Mat & euler);

};

#endif
