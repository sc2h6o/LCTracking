/************************************************************************
* File:	CompressiveTracker.h
* Brief: C++ demo for paper: Kaihua Zhang, Lei Zhang, Ming-Hsuan Yang,"Real-Time Compressive Tracking," ECCV 2012.
* Version: 1.0
* Author: Yang Xian
* Email: yang_xian521@163.com
* Date:	2012/08/03
* History:
* Revised by Kaihua Zhang on 14/8/2012
* Email: zhkhua@gmail.com
* Homepage: http://www4.comp.polyu.edu.hk/~cskhzhang/
************************************************************************/
#pragma once
#ifndef __CT__
#define __CT__

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include "ORF\onlinerf.h"

using std::vector;
using namespace cv;
//---------------------------------------------------
class CompressiveTracker
{
public:
	CompressiveTracker(void);
	~CompressiveTracker(void);

private:
	int featureMinNumRect;
	int featureMaxNumRect;
	int featureNum;
	int classNum;
	int split_vrtc;
	int split_hrzt;
	vector<vector<Rect_<float>>> features;
	vector<vector<float>> featuresWeight;
	int rOuterPositive_init;
	int rOuterPositive;
	int sampleNumPos;
	int sampleNumNeg;
	int sampleNumUncertain;
	float ratioUncertain;
	vector<Rect> samplePositiveBox;
	vector<Rect> sampleNegativeBox;
	int rSearchWindowLg;
	int rSearchWindowSm;
	int numSearchLg;
	float probSearchLg;
	float scaleMin;
	float scaleMax;
	float scaleStep;
	float scaleStepSm;
	int patchWidth;
	int patchHeight;
	Mat imageIntegral;
	Mat samplePositiveFeatureValue;
	Mat sampleNegativeFeatureValue;
	Mat sampleUncertainFeatureValue;
	vector<vector<float>> muPositive;
	vector<vector<float>> sigmaPositive;
	vector<vector<float>> muNegative;
	vector<vector<float>> sigmaNegative;
	vector<float> mu;
	vector<float> sigma;
	vector<double> minFeatRange;
	vector<double>	maxFeatRange;
	float learnRate;
	float overlap;
	float ratioMarginIn;
	float ratioeMarginOut;
	float thrsdBG;
	float thrsdSelf;
	float thrsdUncertain;
	float sigmaSamplePos;
	float sigmaSampleNeg;
	vector<Rect> detectBox;
	vector<Rect> splitBox;
	vector<vector<Rect>> detectSplit;
	Mat detectFeatureValue;
	RNG rng;
	Rect innerBox, outerBox;
	Mat backProj_in, backProj_out;
	Mat pos;
	Mat neg;
	Mat confid;
	Hyperparameters hp;
	OnlineRF orf;
	Result result;

private:
	void SplitBox(Rect _objectBox, vector<Rect>& splitBox);
	void HaarFeature(Rect& _objectBox, int _numFeature);
	void HaarDistribution();
	void getUncertainSample(Sample& sample);
	void UpdateUncertain(Mat _image, Rect& splitBox);
	void sampleRect(Mat& _image, Rect& _objectBox, float _rOuter, float _rInner, int _maxSampleNum, bool checkBG, vector<Rect>& _sampleBox);
	void sampleRect(Mat& _image, Rect& _objectBox, float _srw, vector<Rect>& _sampleBox);
	void sampleNeg(Mat& _image, Rect& _objectBox, int margin_width_out, int margin_height_out,
		int margin_width_in, int margin_height_in, int _maxSampleNum, int width, int height, vector<Rect>& _sampleBox);
	void getFeatureValue(Mat& _imageIntegral, vector<Rect>& _sampleBox, float scale, Mat& _sampleFeatureValue);
	void classifierUpdate(Mat& _sampleFeatureValue, vector<float>& _mu, vector<float>& _sigma, float _learnRate);
	void radioClassifier(vector<vector<float>>& _muPos, vector<vector<float>>& _sigmaPos, vector<vector<float>>& _muNeg,
		vector<vector<float>>& _sigmaNeg, Mat& _splitFeatureValue, vector<float>& splitRadios);
	void insertORF(Mat& _sampleFeatureValue, int label);
	void evalORF(Mat& _splitFeatureValue, vector<float>& confidence);
	void evalORF(Mat& _splitFeatureValue, vector<float>& confidence, double &t);
	void backProj(Mat& _image_rgb, Rect boundIn, Rect boundOut, Mat& backProj);
	float checkBackProj(const Rect box, const Mat& backProj1, const Mat& backProj2);
	void siftSample(int splitIndex, Mat& samplePositiveFeatureValuevector, vector<Rect>& samplePositiveBox, vector<Rect>& sampleNegativeBox);
	void checkConfid(Mat& _splitFeatureValue);

public:
	void processFrame(Mat& _frame, Mat& _image_rgb, Rect& _objectBox);
	void init(Mat& _frame, Mat& _image_rgb, Rect& _objectBox);
	void testFrame(Mat& _frame, Mat& _image_rgb, Rect& _testBox);
};

#endif