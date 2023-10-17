#include <opencv2/opencv.hpp>
#include "EstimateSolvers.h"
#include "FeatureMatcher.h"
#include "Metric.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
	Mat img_L = imread("../image/1_left.bmp", 0);
	Mat img_R = imread("../image/1_right.bmp", 0);

	Mat K1 = (Mat_<double>(3, 3) <<
		1.8645938255524400e+03, 0., 1.2235000000000000e+03, 0.,
		1.8650196462417871e+03, 1.0235000000000000e+03, 0., 0., 1.);

	Mat K2 = (Mat_<double>(3, 3) <<
		1.8645938255524400e+03, 0., 1.2235000000000000e+03, 0.,
		1.8650196462417871e+03, 1.0235000000000000e+03, 0., 0., 1.);

	cv::Mat distCoeffs1 = (cv::Mat_<double>(5, 1) << -1.6502732234090256e-01, 2.2387963508032568e-01,5.9499925878606780e-04, -2.1956082278288725e-03,-1.1767495838344653e-01);
	cv::Mat distCoeffs2 = (cv::Mat_<double>(5, 1) << -1.6451924083318789e-01, 3.0891086884301816e-01,1.7336983214701963e-03, 5.4523545234058050e-04,-3.0935739782656535e-01);

	Mat R_init = (Mat_<double>(3, 3) <<
		0.9999114328901709, 0.009392340442843735, 0.009429226719678319,
	-0.009342321711841973, 0.9999421290352277, -0.005334754495505971,
	-0.009478786871631773, 0.00524619114220951, 0.9999413133169075);

	Vec3d t_init = { -64.9187, 1.22334, 3.54458 };

	////////////////////////////////////////////////////////////////////////////////////////

	FeatureMatcher fm;
	fm.setRatioThresh(0.75);
	fm.setRansacDist(0.01);

	int keyNum = 1000;
	bool useFlann = true;
	FeatureName fn = FeatureName::SIFT;

	std::vector<cv::Point2f> points1;
	std::vector<cv::Point2f> points2;
	fm.findMatchPoints(img_L, img_R, K1, K2, fn, keyNum, useFlann, points1, points2);

	EstimateSolvers es;
	es.setInitR(R_init);
	es.setInitT(t_init);

	int iterations = 100;
	es.setMaxIterations(iterations);

	double toleranceError = 15;
	double hubelThresh = 0.000200;
	Mat R;
	Vec3d t;

	bool flag = es.estimateRT(points1, points2, hubelThresh, toleranceError, R, t);

	double covRT = es.getCovRT();
	double cost = es.getCost();

	////////////////////////////////////////////////////////////////////////////////////

	cout << endl;
	if (flag == true)
	{
		cout << "RT标定成功" << endl;
	}
	else
	{
		cout << "RT失败" << endl;
	}

	cout << endl;
	cout << "R_init: " << endl;
	cout << R_init << endl;
	cout << endl;
	cout << "R: " << endl;
	cout << R << endl;
	cout << endl;
	cout << "t: " << endl;
	cout << t << endl;
	cout << endl;
	cout << "RT_cov: " << covRT << endl;
	cout << "cost: " << cost << endl;
	cout << endl;


	Vec3d rot = Metric::rot2euler(R_init);
	cout << "R_init_rot_x: " << rot[0] << endl;
	cout << "R_init_rot_y: " << rot[1] << endl;
	cout << "R_init_rot_z: " << rot[2] << endl;
	cout << endl;

	rot = Metric::rot2euler(R);
	cout << "R_rot_x: " << rot[0] << endl;
	cout << "R_rot_y: " << rot[1] << endl;
	cout << "R_rot_z: " << rot[2] << endl;
	cout << endl;

	cout << "t_init_x: " << t_init[0] << endl;
	cout << "t_init_y: " << t_init[1] << endl;
	cout << "t_init_z: " << t_init[2] << endl;
	cout << endl;

	cout << "t_x: " << t[0] << endl;
	cout << "t_y: " << t[1] << endl;
	cout << "t_z: " << t[2] << endl;
	cout << endl;

	return 0;
}

