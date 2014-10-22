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
#include <vector>

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

	cap.set(CV_CAP_PROP_POS_FRAMES, 0);
	cap.read(key_frame_0);
	cap.set(CV_CAP_PROP_POS_FRAMES, 50);
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

	Mat img0_key, img1_key;
	SURF detector(400, 5, 3, true, false);
	vector<KeyPoint> keypoints0, keypoints1;
	detector.detect(img0, keypoints0);
	detector.detect(img1, keypoints1);
	drawKeypoints(img0, keypoints0, img0_key, Scalar(255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(img1, keypoints1, img1_key, Scalar(255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("key0", img0_key);
	imshow("key1", img1_key);
	cout << keypoints0.size() << " " << keypoints1.size() << endl;
	
	Mat descriptors0, descriptors1;
	detector.compute(img0, keypoints0, descriptors0);
	detector.compute(img1, keypoints1, descriptors1);

	BFMatcher matcher(NORM_L2, true);
	vector<DMatch> matches;
	vector<DMatch> good_matches;
	double min_dist = 100;
	double max_dist = 0;
	double mid_dist;

	matcher.match(descriptors0, descriptors1, matches);

	Mat img_matches;
	for (int i = 0; i < (int)matches.size(); i++)
	{
		double dist = matches[i].distance;
		min_dist = min(min_dist, dist);
		max_dist = max(max_dist, dist);
		drawMatches(img0, keypoints0,
			img1, keypoints1,
			good_matches, img_matches,
			Scalar::all(-1), Scalar::all(-1),
			vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	}
	mid_dist = 0.15;

	printf("max dist : %.3lf\n", max_dist);
	printf("min dist : %.3lf\n", min_dist);

	for (int i = 0; i < (int)matches.size(); i++)
	{
		if (matches[i].distance <= mid_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}

	Mat img_matches;
	drawMatches(img0, keypoints0,
		img1, keypoints1,
		good_matches, img_matches,
		Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	imshow("matches", img_matches);
	waitKey(0);
	
	return;
}