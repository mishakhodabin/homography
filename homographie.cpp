#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

#include "image.h"


using namespace std;
using namespace cv;

const float inlier_threshold = 2.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio
const int number_iterations_ransac=100;

int main()
{
    Image<uchar> I1 = Image<uchar>(imread("../IMG_0045.JPG", CV_LOAD_IMAGE_GRAYSCALE));
    Image<uchar> I2 = Image<uchar>(imread("../IMG_0046.JPG", CV_LOAD_IMAGE_GRAYSCALE));
	
	namedWindow("I1", 1);
	namedWindow("I2", 1);
	imshow("I1", I1);
	imshow("I2", I2);
    waitKey();
    
    Mat homography;
    
    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;
    
    Ptr<AKAZE> akaze = AKAZE::create();
    akaze->detectAndCompute(I1, noArray(), kpts1, desc1);
    akaze->detectAndCompute(I2, noArray(), kpts2, desc2);
    Mat I1kP,I2kP;
    drawKeypoints(I1, kpts1, I1kP, Scalar::all(-1),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(I2, kpts2, I2kP, Scalar::all(-1),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("I1 key points", I1kP);
    imshow("I2 key points", I2kP);
    waitKey();

    
    BFMatcher matcher(NORM_HAMMING);
    vector< vector<DMatch> > nn_matches;
    matcher.knnMatch(desc1, desc2, nn_matches, 2);
    vector<KeyPoint> matched1, matched2;
    vector<DMatch> good_matches;
    for(size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;
        
        if(dist1 < nn_match_ratio * dist2) {
            matched1.push_back(kpts1[first.queryIdx]);
            matched2.push_back(kpts2[first.trainIdx]);
        }
    }
    
    vector<KeyPoint> inliers1, inliers2;
    for (unsigned i = 0; i < matched1.size(); i++) {
        int new_i = static_cast<int>(inliers1.size());
        inliers1.push_back(matched1[i]);
        inliers2.push_back(matched2[i]);
        good_matches.push_back(DMatch(new_i, new_i, 0));
    }
    
    vector<Point2f> ptsI1(inliers1.size());
    vector<Point2f> ptsI2(inliers1.size());
    for (int i = 0; i < inliers1.size(); i++) {
        ptsI1[i]=inliers1[i].pt;
        ptsI2[i]=inliers2[i].pt;
    }
    
    Mat M1;
    drawMatches(I1kP, inliers1, I2kP, inliers2, good_matches, M1);
    imshow("Images Liens", M1);
    
    waitKey();
    
    vector<Point2f> matchedPoints1, matchedPoints2;
    cv::KeyPoint::convert(matched1, matchedPoints1);
    cv::KeyPoint::convert(matched2, matchedPoints2);
    
    Mat H;
    H=findHomography(matchedPoints1, matchedPoints2, RANSAC, inlier_threshold);
    
    Mat K(I1.rows,
          2*I1.cols, CV_8U);
    
    warpPerspective(I2, K, H,K.size(), cv::InterpolationFlags::WARP_INVERSE_MAP);
    cout<<"size de K"<<K.cols<<endl;
    for (int i=0; i<I1.rows; i++){
        for (int j=0; j<I1.cols; j++){
            K.at<uchar>(i,j)= I1.at<uchar>(i,j);
        }
    }
    imshow("Image FIN", K);
    waitKey();

    return 0;
}
