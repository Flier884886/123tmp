#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/geometry.hpp>
#include <vector>
using namespace std;

//二值化处理
cv::Mat biner(const cv::Mat& red_score, unsigned long long* WhiteCount = nullptr) {
    const double OFFSET_RATIO = 7.0 / 11.0;
    const int OFFSET = 0;
    const double MIN_PIXEL_RATIO = 0.5; // 保证至少 50% 的像素熄灭
    const double MAX_OUTLIER_RATIO = 0.00001; // 剔除最亮的 0.001% 极端值
    const int HARD_MIN_THRESHOLD = 10;

    cv::Mat bin = cv::Mat::zeros(red_score.size(), CV_8UC1);
    double sum_val = 0.0;//总亮度和，用于计算平均亮度
    int total_pixels = red_score.rows * red_score.cols;
    unsigned long long histogram[256] = { 0 };//直方图
    for (int y = 0; y < red_score.rows; y++) {
        const uchar* row_ptr = red_score.ptr<uchar>(y);
        for (int x = 0; x < red_score.cols; x++) {
            int val = row_ptr[x];
            sum_val += val;
            histogram[val]++;
        }
    }
    if (total_pixels == 0) return bin;
    //计算鲁棒最大值（滤除最亮极端值）
    int outlier_count = std::max(1, (int)(total_pixels * MAX_OUTLIER_RATIO));
    int accumulated = 0;
    int max_val = 255;
    for (int i = 255; i >= 0; i--) {
        accumulated += histogram[i];
        if (accumulated >= outlier_count) {
            max_val = i;
            break;
        }
    }
    //寻找特定分位数的阈值下限
    int target_count = (int)(total_pixels * MIN_PIXEL_RATIO);
    accumulated = 0;
    int dynamic_min_threshold = HARD_MIN_THRESHOLD;

    for (int i = 0; i < 256; i++) {
        accumulated += histogram[i];
        if (accumulated >= target_count) {
            dynamic_min_threshold = i;
            break;
        }
    }
    //计算平均值
    double avg_val = sum_val / total_pixels;
    // 注意：计算阈值时使用的是鲁棒最大值
    double calc_threshold = avg_val + (max_val - avg_val) * OFFSET_RATIO + OFFSET;
    int final_threshold = (int)calc_threshold;
    if (final_threshold < dynamic_min_threshold) {
        final_threshold = dynamic_min_threshold;
    }
    if (final_threshold > 255) final_threshold = 255;
    if (final_threshold < HARD_MIN_THRESHOLD) final_threshold = HARD_MIN_THRESHOLD;
    if (WhiteCount != nullptr) {
        *WhiteCount = 0;
        for (int i = final_threshold; i < 256; i++)
            *WhiteCount += histogram[i];
    }
    // 二值化
    for (int y = 0; y < red_score.rows; y++) {
        const uchar* row_ptr = red_score.ptr<uchar>(y);
        uchar* bin_ptr = bin.ptr<uchar>(y);
        for (int x = 0; x < red_score.cols; x++) {
            if (row_ptr[x] > final_threshold) {
                bin_ptr[x] = 255;
            }
        }
    }
    return bin;
}

int main() {
    using namespace cv;
	Mat img = imread("3.jpg");
	if (img.empty())return -1;
	Mat gray;
	cvtColor(img.clone(), gray, COLOR_BGR2GRAY);
    gray = biner(gray);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));//核
    cv::dilate(gray, gray, kernel);//膨胀
    cv::erode(gray, gray, kernel);//腐蚀
    vector<vector<Point>> cs;
    findContours(gray, cs, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (auto& c : cs) {
        RotatedRect r = minAreaRect(c);
        if (r.size.area() < 10)continue;
        if (max(r.size.width, r.size.height) / min(r.size.width, r.size.height) < 3)continue; 
        Point2f p[4];
        r.points(p);
        for (int i = 0; i < 4; i++) {
            line(img, p[i], p[(i + 1) % 4], Scalar(255, 0, 0), 2);//蓝线
        }
    }
	imwrite("3res.png", img);
	return 0;
}
