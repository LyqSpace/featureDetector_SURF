/*
*	Copyright (C)   Lyq root#lyq.me
*	File Name     : featureDetector_SURF
*	Creation Time : 2014-10-20 16:46:00
*	Environment   : Windows8.1-64bit VS2013 OpenCV2.4.9
*	Homepage      : http://www.lyq.me
*/

#ifndef OPENCV
#define OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
using namespace cv;
#endif // !OPENCV

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>

using namespace std;

bool openCapture(VideoCapture&, const char*);
void featureDetector_SURF(const Mat&, const Mat&);

int main(void)
{
	char *filename = "shot1_720p.avi";
	VideoCapture cap;
	
	int state = openCapture(cap, filename);
	if (!state) return -2;
	
	Mat key_frame_0, key_frame_1;

	cap.set(CV_CAP_PROP_POS_AVI_RATIO, 0);
	cap.read(key_frame_0);
	cap.set(CV_CAP_PROP_POS_AVI_RATIO, 20);
	cap.read(key_frame_1);
	
	featureDetector_SURF(key_frame_0, key_frame_1);

	return 0;
}

bool openCapture(VideoCapture &cap, const char *filename)
{
	if (!cap.open(filename))
	{
		cout << "The filename error!" << endl;
		return false;
	}
	return true;
}

void featureDetector_SURF(const Mat &img0, const Mat &img1)
{
	if (img0.empty() || img1.empty())
	{
		printf("Can't read one of the images.\n");
		return;
	}
	
	SURF detector(400, 4, 2, true, false);
	vector<KeyPoint> keypoints0, keypoints1;
	detector.detect(img0, keypoints0);
	detector.detect(img1, keypoints1);
	Mat descriptors0, descriptors1;
	detector.compute(img0, keypoints0, descriptors0);
	detector.compute(img1, keypoints1, descriptors1);

	BFMatcher matcher(NORM_L2, true);
	vector<DMatch> matches;
	matcher.match(descriptors0, descriptors1, matches);

	Mat img_matches;
	drawMatches(img0, keypoints0, img1, keypoints1, matches, img_matches);

	imshow("matches", img_matches);
	waitKey(0);

	return;
}