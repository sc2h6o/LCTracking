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
	bool eqhst;
	int featureMinNumRect;
	int featureMaxNumRect;
	int featureNum;
	int classNum;
	int split_num;
	int split_vrtc;
	int split_hrzt;
	int rOuterPositive_init;
	int rOuterPositive;
	int sampleNumPos;
	int sampleNumNeg;
	int sampleNumNegInit;
	int sampleNumUncertain;
	float ratioUncertain;
	int rSearchWindowLg;
	int rSearchWindowSm;
	int searchRoundLg;
	int searchRoundSm;
	float alpha;
	float bias;
	float probSearchLg;
	float probSearchSm;
	float scaleMin;
	float scaleMax;
	float scaleStep;
	float scaleStepSm;
	float learnRate;
	float overlap;
	float ratioMarginIn;
	float ratioeMarginOut;
	float thrsdBG;
	float thrsdSelf;
	float thrsdUncertain;
	float ratioOrigin;
	float lastConfid;
	Hyperparameters hp;
	Hyperparameters hpOrigin;
	vector<vector<Rect_<float>>> features;
	vector<vector<float>> featuresWeight;
	vector<float> mu;
	vector<float> sigma;
	RNG rng;
	vector<Mat> initPositiveFeatureValues;
	Mat imageIntegral;
	Mat pos;
	Mat neg;
	Mat like;
	Mat confid;
	OnlineRF orf;
	OnlineRF orfOrigin;

private:
	void SplitBox(Rect _objectBox, vector<Rect>& splitBox);
	void HaarFeature(int _numFeature);
	void getUncertainSample(Sample& sample);
	void UpdateUncertain(Mat imageIntegral, Rect& splitBox);
	void sampleRect(Mat& _image, Rect& _objectBox, Rect bound, float _rOuter, float _rInner, int _maxSampleNum, bool checkBG, vector<Rect>& _sampleBox, vector<Weight>& _sampleWeight);
	void sampleRect(Mat& _image, Rect& _objectBox, float _srw, vector<Rect>& _sampleBox, vector<float>& _prior);
	void sampleNeg(Mat& _image, Rect& _objectBox, int margin_width_out, int margin_height_out,
		int margin_width_in, int margin_height_in, int _maxSampleNum, int width, int height, vector<Rect>& _sampleBox);
	void getFeatureValue(Mat& _imageIntegral, vector<Rect>& _sampleBox, float scale, Mat& _sampleFeatureValue);
	void insertORF(OnlineRF& orf, Mat& _sampleFeatureValue, int label, vector<Weight>& _sampleWeight);
	void evalORF(OnlineRF& orf, Mat& _splitFeatureValue, vector<float>& confidence, vector<float>& suspicion);
	float crossEval(vector<float>& splitConfid, vector<float>& splitSuspc);
	void siftSample(int splitIndex, Mat& sampleFeatureValue, vector<Rect>& sampleBoxes, vector<Weight>& _sampleWeight);
	void checkConfid(OnlineRF& orf, Mat& _splitFeatureValue);
	void equalHistPart(Mat& imageIn, Mat& imageOut, Rect& _objectBox, int rSearchWindowLg);
	void GammaCorrection(Mat& imageIn, Mat& imageOut, float fGamma);
	Rect meanShift(vector<float> confidence, vector<Rect>detectBox);
	void searchRegion(Mat& _frame, Mat& imageIntegral, int radius, float detectProb,
		float scaleMin, float scaleMax, float scaleStep, Rect& boxTemp, float& confidTemp);
	void detect(Mat& _frame, Mat& _image_rgb, Mat& imageIntegral, Rect& _objectBox);
public:
	void processFrame(Mat& _frame, Mat& _image_rgb, Rect& _objectBox);
	void init(Mat& _frame, Mat& _image_rgb, Rect& _objectBox);
	void testFrame(Mat& _frame, Mat& _image_rgb, Rect& _testBox);
};

#endif