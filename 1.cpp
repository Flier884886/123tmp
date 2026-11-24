#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/geometry.hpp>
#include <vector>
using namespace cv;
using namespace std;
int main() {
	Mat img = imread("2.png");
	if (img.empty())return -1;
	Mat blue, red;
	inRange(img, Scalar(180, 80, 0), Scalar(255, 170, 60), blue);//二值化蓝
	inRange(img, Scalar(0, 0, 150), Scalar(80, 80, 255), red);
	vector<vector<Point>>cb, cr;//蓝红轮廓点
	findContours(blue, cb, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);//外轮廓，拐点
	findContours(red, cr, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	for(auto&c:cb){
		RotatedRect r = minAreaRect(c);
		Point2f p[4]; r.points(p);
		for (int i = 0; i < 4; i++) {
			line(img, p[i], p[(i + 1) % 4], Scalar(0, 255, 255), 2);//黄线
		}
	}
	for (auto& c : cr) {
		RotatedRect r = minAreaRect(c);
		Point2f p[4]; r.points(p);
		for (int i = 0; i < 4; i++) {
			line(img, p[i], p[(i + 1) % 4], Scalar(0, 255, 0), 2);//绿线
		}
	}
	imwrite("2res.png", img);
	return 0;
}
