#include "CompressiveTracker.h"
#include <math.h>
#include <iostream>
using namespace cv;
using namespace std;

//------------------------------------------------
CompressiveTracker::CompressiveTracker(void)
{
	featureMinNumRect = 2;
	featureMaxNumRect = 4;	// number of rectangle from 2 to 4
	featureNum = 10;	// number of all weaker classifiers, i.e,feature pool
	split_hrzt = 4;		// number of splitted segments
	split_vrtc = 4;
	classNum = split_hrzt * split_vrtc + 2;
	overlap = 0.25;
	ratioUncertain = 0.2;
	sampleNumPos = 30;
	sampleNumNeg = 200;
	sampleNumUncertain = 200;
	ratioeMarginOut = 1.5;
	ratioMarginIn = 0.1;
	thrsdBG = 4;
	thrsdSelf = 3;
	thrsdUncertain = 2.5;
	sigmaSamplePos = 0.0 / 12;
	sigmaSampleNeg = 0.0 / 12;

	rOuterPositive_init = 4;
	rOuterPositive = 4;	// radical scope of positive samples
	rSearchWindowLg = 80; // size of search window
	rSearchWindowSm = 10; // size of search window
	numSearchLg = 8;
	probSearchLg = 0.08;
	scaleMin = 0.85f;
	scaleMax = 1.16f;
	scaleStep = 0.5f;
	scaleStepSm = 0.05f;
	muPositive = vector<vector<float>>(split_hrzt*split_vrtc, vector<float>(featureNum, 0.0f));
	muNegative = vector<vector<float>>(split_hrzt*split_vrtc, vector<float>(featureNum, 0.0f));
	sigmaPositive = vector<vector<float>>(split_hrzt*split_vrtc, vector<float>(featureNum, 1.0f));
	sigmaNegative = vector<vector<float>>(split_hrzt*split_vrtc, vector<float>(featureNum, 1.0f));
	mu.resize(featureNum, 0);
	sigma.resize(featureNum, 0);
	learnRate = 0.85f;	// Learning rate parameter
	// hyperparameter for orf
	hp.counterThreshold = 70;
	hp.maxDepth = 15;
	hp.useUncertain = true;
	hp.splitRatioUncertain = 2.5;  // a node could split when samples are less than ratio*uncertains
	hp.numRandomTests = 30;
	hp.numTrees = 3;
	hp.numGrowTrees = 1;
	hp.MatureDepth = 10;
	hp.useSoftVoting = true;
	hp.verbose = 1;

	result.confidence.resize(classNum, 0);
}

CompressiveTracker::~CompressiveTracker(void)
{
}

void CompressiveTracker::SplitBox(Rect _objectBox, vector<Rect>& splitBox){
	int width_sm = _objectBox.width / ((split_hrzt-1) * (1-overlap)+1);
	int height_sm = _objectBox.height / ((split_vrtc - 1) * (1 - overlap) + 1);
	splitBox.resize(split_hrzt * split_vrtc);
	
	// left to right, up to down
	for (int i = 0; i < split_hrzt; i++){
		for (int j = 0; j < split_vrtc; j++){
			splitBox[i*split_vrtc + j] = Rect(_objectBox.x + width_sm*(1-overlap)*i, 
				_objectBox.y + height_sm*(1-overlap)*j, width_sm, height_sm);
		}
	}
}

void CompressiveTracker::HaarFeature(Rect& _objectBox, int _numFeature)
/*Description: compute Haar features
  Arguments:
  -_objectBox: [x y width height] object rectangle
  -_numFeature: total number of features.The default is 50.
*/
{
	features = vector<vector<Rect_<float>>>(_numFeature, vector<Rect_<float>>());
	featuresWeight = vector<vector<float>>(_numFeature, vector<float>());
	
	int numRect;
	Rect_<float> rectTemp;
	float weightTemp;
      
	for (int i=0; i<_numFeature; i++)
	{
		numRect = cvFloor(rng.uniform((double)featureMinNumRect, (double)featureMaxNumRect));
	    
		//int c = 1;
		for (int j=0; j<numRect; j++)
		{
			
			/*rectTemp.x = cvFloor(rng.uniform(0.0, (double)(_objectBox.width - 3)));
			rectTemp.y = cvFloor(rng.uniform(0.0, (double)(_objectBox.height - 3)));
			rectTemp.width = cvCeil(rng.uniform(0.0, (double)(_objectBox.width - rectTemp.x - 2)));
			rectTemp.height = cvCeil(rng.uniform(0.0, (double)(_objectBox.height - rectTemp.y - 2)));*/
			rectTemp.x = rng.uniform(0.0, 0.95);
			rectTemp.y = rng.uniform(0.0, 0.95);
			rectTemp.width = rng.uniform(0.04, 1.0 - rectTemp.x);
			rectTemp.height = rng.uniform(0.04, 1.0 - rectTemp.y);
			features[i].push_back(rectTemp);

			weightTemp = (float)pow(-1.0, cvFloor(rng.uniform(0.0, 2.0))) / sqrt(float(numRect));
            //weightTemp = (float)pow(-1.0, c);
			

			featuresWeight[i].push_back(weightTemp);
           
		}
	}
}

void CompressiveTracker::HaarDistribution(){
	int numFeature = features.size();
	for (int i = 0; i < numFeature; i++){
		mu[i] = sigma[i] = 0;
		for (int j = 0; j < features[i].size(); j++){
			float mu_box = features[i][j].area() * 128; // the mu of a single pixel is 128
			float sigma_box = features[i][j].area() * (256 * 256 / 12);	// the sigma of a single pixel is (256-0)^2/12
			
			// linear combination of Gaussian
			mu[i] += featuresWeight[i][j] * mu_box;
			sigma[i] += featuresWeight[i][j] * featuresWeight[i][j] * sigma_box;
		}
	}
}

void CompressiveTracker::UpdateUncertain(Mat _image, Rect& splitBox){
	Scalar muTemp;
	Scalar sigmaTemp;

	vector<Rect> uncertainDetect(sampleNumUncertain);
	int searchWidth = _image.cols - splitBox.width;
	int searchHeight = _image.rows - splitBox.height;
	for (int i = 0; i < sampleNumUncertain; i++){
		uncertainDetect[i].x = rng.uniform(0, searchWidth);
		uncertainDetect[i].y = rng.uniform(0, searchHeight);
		uncertainDetect[i].width = splitBox.width;
		uncertainDetect[i].height = splitBox.height;
	}
	getFeatureValue(imageIntegral, uncertainDetect, 1, sampleUncertainFeatureValue);
	for (int i = 0; i<featureNum; i++)
	{
		meanStdDev(sampleUncertainFeatureValue.row(i), muTemp, sigmaTemp);

		sigma[i] = (float)sqrt(learnRate*sigma[i] * sigma[i] + (1.0f - learnRate)*sigmaTemp.val[0] * sigmaTemp.val[0]
			+ learnRate*(1.0f - learnRate)*(mu[i] - muTemp.val[0])*(mu[i] - muTemp.val[0]));	// equation 6 in paper

		mu[i] = mu[i] * learnRate + (1.0f - learnRate)*muTemp.val[0];	// equation 6 in paper
	}
}

void CompressiveTracker::getUncertainSample(Sample& sample){
	sample.x.resize(featureNum);
	for (int i = 0; i < featureNum; i++){
		float featureRandom = rng.gaussian(sigma[i]);
		featureRandom += mu[i];
		sample.x[i] = featureRandom;
		// cout << sample.x[i] << " ";
	}
	// cout << endl;
	sample.y = split_hrzt*split_vrtc + 1; // the class of uncertain
	sample.w = 1.0;
}

void CompressiveTracker::sampleRect(Mat& _image, Rect& _objectBox, float _rOuter, float _rInner, int _maxSampleNum, bool checkBG,vector<Rect>& _sampleBox)
/* Description: compute the coordinate of positive and negative sample image templates
   Arguments:
   -_image:        processing frame
   -_objectBox:    recent object position
   -_rOuter:       Outer sampling radius
   -_rInner:       inner sampling radius
   -_maxSampleNum: maximal number of sampled images
   -_sampleBox:    Storing the rectangle coordinates of the sampled images.
   */
{
	int rowsz = _image.rows - _objectBox.height - 1;
	int colsz = _image.cols - _objectBox.width - 1;
	float inradsq = _rInner*_rInner;
	float outradsq = _rOuter*_rOuter;


	int dist;

	int minrow = max(0, (int)_objectBox.y - (int)_rOuter);
	int maxrow = min((int)rowsz - 1, (int)_objectBox.y + (int)_rOuter);
	int mincol = max(0, (int)_objectBox.x - (int)_rOuter);
	int maxcol = min((int)colsz - 1, (int)_objectBox.x + (int)_rOuter);



	int i = 0;

	float prob = ((float)(_maxSampleNum)) / (maxrow - minrow + 1) / (maxcol - mincol + 1);

	int r;
	int c;

	_sampleBox.clear();//important
	Rect rec(0, 0, 0, 0);

	for (r = minrow; r <= (int)maxrow; r++){
		for (c = mincol; c <= (int)maxcol; c++){
			dist = (_objectBox.y - r)*(_objectBox.y - r) + (_objectBox.x - c)*(_objectBox.x - c);

			if (rng.uniform(0., 1.) < prob && dist < outradsq && dist >= inradsq){

				rec.x = c;
				rec.y = r;
				rec.width = _objectBox.width;
				rec.height = _objectBox.height;
				
				_sampleBox.push_back(rec);
				i++;
			}
		}
	}
	_sampleBox.resize(i);
		
}

void CompressiveTracker::sampleRect(Mat& _image, Rect& _objectBox, float _srw, vector<Rect>& _sampleBox)
/* Description: Compute the coordinate of samples when detecting the object.*/
{
	int rowsz = _image.rows - _objectBox.height - 1;
	int colsz = _image.cols - _objectBox.width - 1;
	float inradsq = _srw*_srw;	
	

	int dist;

	int minrow = max(0,(int)_objectBox.y-(int)_srw);
	int maxrow = min((int)rowsz-1,(int)_objectBox.y+(int)_srw);
	int mincol = max(0,(int)_objectBox.x-(int)_srw);
	int maxcol = min((int)colsz-1,(int)_objectBox.x+(int)_srw);

	int i = 0;

	int r;
	int c;

	Rect rec(0,0,0,0);
    _sampleBox.clear();//important

	for( r=minrow; r<=(int)maxrow; r++ )
		for( c=mincol; c<=(int)maxcol; c++ ){
			dist = (_objectBox.y-r)*(_objectBox.y-r) + (_objectBox.x-c)*(_objectBox.x-c);

			if( dist < inradsq ){

				rec.x = c;
				rec.y = r;
				rec.width = _objectBox.width;
				rec.height= _objectBox.height;

				_sampleBox.push_back(rec);				

				i++;
			}
		}
	
		_sampleBox.resize(i);

}

void CompressiveTracker::sampleNeg(Mat& _image, Rect& _objectBox, int margin_width_out, int margin_height_out,
	int margin_width_in, int margin_height_in, int _maxSampleNum, int width, int height, vector<Rect>& _sampleBox)
{
	_sampleBox.clear();
	Rect window, badRegion;
	window.x = max(0, _objectBox.x - margin_width_out);
	window.y = max(0, _objectBox.y - margin_height_out);
	window.width = min(_image.cols, _objectBox.x + _objectBox.width + margin_width_out) - window.x;
	window.height = min(_image.rows, _objectBox.y + _objectBox.height + margin_height_out) - window.y;
	badRegion.x = max(window.x, _objectBox.x - width + margin_width_in);
	badRegion.y = max(window.y, _objectBox.y - height + margin_height_in);
	badRegion.width = min(window.x + window.width - width, _objectBox.x + _objectBox.width - margin_width_in) - badRegion.x;
	badRegion.height = min(window.y + window.height - height, _objectBox.y + _objectBox.height - margin_height_in) - badRegion.y;
	if (badRegion.area() <= 0) badRegion = Rect();
	rectangle(neg, window, 122, -1);
	rectangle(neg, badRegion, 200, -1);
	int all_sample = (window.width - width) *(window.height - height) - badRegion.width * badRegion.height;
	float prob = min(1.0f, _maxSampleNum*1.0f / all_sample);
	int cnt = 0;
	for (int i = window.x; i < window.x + window.width - width; i++){
		for (int j = window.y; j < window.y + window.height - height; j++){
			if (!badRegion.contains(Point(i, j))){
				circle(neg, Point(i, j), 1, 244);
				cnt++;
			}
			if (!badRegion.contains(Point(i,j)) && rng.uniform(0.0, 1.0) < prob)
			{
				_sampleBox.push_back(Rect(i, j, width, height));
				//rectangle(neg, Rect(i, j, width, height), 255, -1);
			}
		}
	}
}


// Compute the features of samples
void CompressiveTracker::getFeatureValue(Mat& _imageIntegral, vector<Rect>& _sampleBox, float scale, Mat& _sampleFeatureValue)
{
	int sampleBoxSize = _sampleBox.size();
	_sampleFeatureValue.release();
	_sampleFeatureValue.create(featureNum, sampleBoxSize, CV_32F);
	float tempValue;
	float tempArea;
	int xMin;
	int xMax;
	int yMin;
	int yMax;

	for (int i=0; i<featureNum; i++)
	{
		for (int j=0; j<sampleBoxSize; j++)
		{
			tempValue = 0.0f;
			for (size_t k=0; k<features[i].size(); k++)
			{
				/*xMin = _sampleBox[j].x + scale * (features[i][k].x);
				xMax = _sampleBox[j].x + scale * (features[i][k].x + features[i][k].width);
				yMin = _sampleBox[j].y + scale * (features[i][k].y);
				yMax = _sampleBox[j].y + scale * (features[i][k].y + features[i][k].height);*/
				xMin = _sampleBox[j].x + _sampleBox[j].width * (features[i][k].x);
				xMax = _sampleBox[j].x + _sampleBox[j].width * (features[i][k].x + features[i][k].width);
				yMin = _sampleBox[j].y + _sampleBox[j].height * (features[i][k].y);
				yMax = _sampleBox[j].y + _sampleBox[j].height * (features[i][k].y + features[i][k].height);
				tempArea = (xMax - xMin)*(yMax - yMin);

				tempValue += (1.0 / tempArea) * featuresWeight[i][k] *
					(_imageIntegral.at<float>(yMin, xMin) +
					_imageIntegral.at<float>(yMax, xMax) -
					_imageIntegral.at<float>(yMin, xMax) -
					_imageIntegral.at<float>(yMax, xMin));
			}
			_sampleFeatureValue.at<float>(i,j) = tempValue;
		}
	}
}

// Update the mean and variance of the gaussian classifier
void CompressiveTracker::classifierUpdate(Mat& _sampleFeatureValue, vector<float>& _mu, vector<float>& _sigma, float _learnRate)
{
	Scalar muTemp;
	Scalar sigmaTemp;
    
	for (int i=0; i<featureNum; i++)
	{
		meanStdDev(_sampleFeatureValue.row(i), muTemp, sigmaTemp);
	   
		_sigma[i] = (float)sqrt( _learnRate*_sigma[i]*_sigma[i]	+ (1.0f-_learnRate)*sigmaTemp.val[0]*sigmaTemp.val[0] 
		+ _learnRate*(1.0f-_learnRate)*(_mu[i]-muTemp.val[0])*(_mu[i]-muTemp.val[0]));	// equation 6 in paper

		_mu[i] = _mu[i]*_learnRate + (1.0f-_learnRate)*muTemp.val[0];	// equation 6 in paper
	}
}

// Compute the ratio classifier 
void CompressiveTracker::radioClassifier(vector<vector<float>>& _muPos, vector<vector<float>>& _sigmaPos,
	vector<vector<float>>& _muNeg, vector<vector<float>>& _sigmaNeg, Mat& _splitFeatureValue, vector<float>& splitRadios)
{
	float sumRadio;
	float pPos;
	float pNeg;
	int splitBoxNum = _splitFeatureValue.cols;

	splitRadios.resize(splitBoxNum);
	for (int j = 0; j<splitBoxNum; j++)
	{
		sumRadio = 0.0f;
		for (int i=0; i<featureNum; i++)
		{
			pPos = exp((_splitFeatureValue.at<float>(i, j) - _muPos[j][i])*(_splitFeatureValue.at<float>(i, j) - _muPos[j][i]) / -(2.0f*_sigmaPos[j][i] * _sigmaPos[j][i] + 1e-30)) / (_sigmaPos[j][i] + 1e-30);
			pNeg = exp((_splitFeatureValue.at<float>(i, j) - _muNeg[j][i])*(_splitFeatureValue.at<float>(i, j) - _muNeg[j][i]) / -(2.0f*_sigmaNeg[j][i] * _sigmaNeg[j][i] + 1e-30)) / (_sigmaNeg[j][i] + 1e-30);
			sumRadio += log(pPos+1e-30) - log(pNeg+1e-30);	// equation 4
		}
		splitRadios[j] = sumRadio;
	}
}

void CompressiveTracker::insertORF(Mat& _sampleFeatureValue, int label)
{
	int cnt = 0;
	for (int i = 0; i < _sampleFeatureValue.cols;  i++){
		Sample sample;
		sample.x.resize(_sampleFeatureValue.rows);
		for (int j = 0; j < _sampleFeatureValue.rows; j++){
			sample.x[j] = _sampleFeatureValue.at<float>(j, i);
		}
		sample.y = label;
		sample.w = 1.0;
		orf.insert(sample);
		if (rng.uniform(0.0, 1.0) < ratioUncertain){
			Sample sampleUncertain;
			getUncertainSample(sampleUncertain);
			orf.insert(sampleUncertain);
			cnt++;
		}
	}
}

void CompressiveTracker::backProj(Mat& _image_rgb, Rect boundIn, Rect boundOut, Mat& backProj)
{
	int smin = 0;
	int _vmin = 0;
	int _vmax = 255;
	int hist_sizes[] = { 36, 36, 36 };
	int hist_channels[] = { 0, 1, 2 };
	float h_ranges[] = { 0, 256 };
	float s_ranges[] = { 0, 256 };
	float v_ranges[] = { 0, 256 };
	const float *hist_ranges[] = { h_ranges, s_ranges, v_ranges };

	Mat hsv, hue, mask, mask_bound, hist;
	cvtColor(_image_rgb, hsv, CV_RGB2HSV);
	inRange(hsv, Scalar(0, smin, MIN(_vmin, _vmax)),
		Scalar(180, 256, MAX(_vmin, _vmax)), mask);
	mask_bound = Mat(mask.size(), CV_8U, Scalar::all(0));
	rectangle(mask_bound, boundOut, 255, -1); // thickness = -1 means fill with color
	rectangle(mask_bound, boundIn, 0, -1);
	mask &= mask_bound;

	calcHist(&hsv, 1, hist_channels, mask, hist, 2, hist_sizes, hist_ranges);
	normalize(hist, hist, 0, 255, CV_MINMAX);

	calcBackProject(&hsv, 1, hist_channels, hist, backProj, hist_ranges);
}

float CompressiveTracker::checkBackProj(const Rect box,const Mat& backProj1, const Mat& backProj2){
	float prob1=0, prob2=0;
	for (int i = box.x; i < box.x + box.width; i++){
		for (int j = box.y; j < box.y + box.height; j++){
			prob1 += backProj1.at<uchar>(j, i);
			prob2 += backProj2.at<uchar>(j, i);
		}
	}
	if (prob2 == 0) return FLT_MAX;
	return prob1 / prob2;
}

void CompressiveTracker::evalORF(Mat& _splitFeatureValue, vector<float>& confidence)
{
	confidence.resize(_splitFeatureValue.cols + 1, 0); // plus 1 for the background class
	for (int i = 0; i < _splitFeatureValue.cols; i++){
		Sample sample;
		sample.x.resize(_splitFeatureValue.rows);
		for (int j = 0; j < _splitFeatureValue.rows; j++){
			sample.x[j] = _splitFeatureValue.at<float>(j, i);
		}
		sample.w = 1.0;

		orf.eval(sample, result);

		//confidence[i] = 0.5 * (result.confidence[i] - 1 * result.confidence[split_hrzt*split_vrtc] + 1);
		confidence[i] = result.confidence[i];
	}
}

void CompressiveTracker::evalORF(Mat& _splitFeatureValue, vector<float>& confidence, double &t)
{
	confidence.resize(_splitFeatureValue.cols + 1, 0); // plus 1 for the background class
	for (int i = 0; i < _splitFeatureValue.cols; i++){

		Sample sample;
		sample.x.resize(_splitFeatureValue.rows);
		for (int j = 0; j < _splitFeatureValue.rows; j++){
			sample.x[j] = _splitFeatureValue.at<float>(j, i);
		}
		sample.w = 1.0;

		double tc = getTickCount();
		orf.eval(sample,result);
		t += (cvGetTickCount() - tc) / getTickFrequency();
		//confidence[i] = 0.5 * (result.confidence[i] - 1 * result.confidence[split_hrzt*split_vrtc] + 1);
		confidence[i] = result.confidence[i];

	}
}

bool OnlineNode::echo = false;
void CompressiveTracker::checkConfid(Mat& _splitFeatureValue)
{
	OnlineNode::echo = true;
	int splitNum = _splitFeatureValue.cols;
	for (int i = 0; i < _splitFeatureValue.cols; i++){
		Sample sample;
		sample.x.resize(_splitFeatureValue.rows);
		for (int j = 0; j < _splitFeatureValue.rows; j++){
			sample.x[j] = _splitFeatureValue.at<float>(j, i);
		}
		sample.w = 1.0;

		if (OnlineNode::echo)
			cout << "path of patch " << i << ": " << endl;
		orf.eval(sample, result);

		float width = (float)confid.rows / (splitNum + 2);
		float height = confid.cols / splitNum;
		
		for (int j = 0; j < _splitFeatureValue.cols + 2; j++){
			rectangle(confid, Rect(j*width, i*height, width, height), 
				result.confidence[j] * 255, -1);
		}
	}
	OnlineNode::echo = false;
}

void CompressiveTracker::siftSample(int splitIndex, Mat& samplePositiveFeatureValuevector, vector<Rect>& samplePositiveBox, vector<Rect>& sampleNegativeBox){
	vector<Rect> positive_new;

	for (int i = 0; i < samplePositiveFeatureValue.cols; i++){
		Sample sample;
		sample.x.resize(samplePositiveFeatureValue.rows);
		for (int j = 0; j < samplePositiveFeatureValue.rows; j++){
			sample.x[j] = samplePositiveFeatureValue.at<float>(j, i);
		}
		sample.w = 1.0;

		orf.eval(sample, result);
		float confid_fg = -FLT_MAX;
		float confid_bg = result.confidence[split_hrzt*split_vrtc];
		float confid_other = -FLT_MAX;
		float confid_self = result.confidence[splitIndex];
		for (int j = 0; j < split_hrzt*split_vrtc + 1; j++){
			if (j != split_hrzt*split_vrtc) 
				confid_fg = max(confid_fg, (float)result.confidence[j]);
			if (j != splitIndex) 
				confid_other = max(confid_other, (float)result.confidence[j]);
		}
		// pick out bcg patches
		if (confid_bg > thrsdBG * confid_fg && 
			ratioUncertain*confid_bg > thrsdUncertain*result.confidence[split_hrzt*split_vrtc + 1])
		{
			sampleNegativeBox.push_back(samplePositiveBox[i]);
			rectangle(neg, samplePositiveBox[i], confid_bg * 255, -1);
		}
		else{
			// pick out fully coonfident patches
			if (confid_self < thrsdSelf * confid_other)
				positive_new.push_back(samplePositiveBox[i]);
		}
	}
	samplePositiveBox = positive_new;
}

void CompressiveTracker::init(Mat& _frame, Mat& _image_rgb, Rect& _objectBox)
{
	pos = Mat(_frame.size(), CV_8U, Scalar::all(0));
	neg = Mat(_frame.size(), CV_8U, Scalar::all(0));
	confid = Mat(400, 400, CV_8U, Scalar(0));
	/*
	scaleBox(_objectBox, innerBox, 0.7);
	scaleBox(_objectBox, outerBox, 1.5);
	backProj(_image_rgb, Rect(), innerBox, backProj_in);
	backProj(_image_rgb, _objectBox, outerBox, backProj_out);*/
	
	// init the Online Random Forest
	orf.init(hp, classNum);
	
	// compute feature template
	SplitBox(_objectBox, splitBox);
	patchWidth = 1 * splitBox[0].width;
	patchHeight = 1 * splitBox[0].height;
	HaarFeature(Rect(0,0,patchWidth,patchHeight), featureNum);

	// compute sample templates
	integral(_frame, imageIntegral, CV_32F);
	UpdateUncertain(_frame, splitBox[0]);

	sampleNeg(_frame, _objectBox,
		max(rSearchWindowLg, (int)(ratioeMarginOut * splitBox[0].width)),
		max(rSearchWindowLg, (int)(ratioeMarginOut * splitBox[0].height)),
		(int)(ratioMarginIn * splitBox[0].width), (int)(ratioMarginIn * splitBox[0].height),
		sampleNumNeg, splitBox[0].width, splitBox[0].height, sampleNegativeBox);
	getFeatureValue(imageIntegral, sampleNegativeBox, 1, sampleNegativeFeatureValue);
	insertORF(sampleNegativeFeatureValue, split_hrzt*split_vrtc);
	
	for (int i = 0; i < split_hrzt*split_vrtc; i++){
		sampleRect(_frame, splitBox[i], rOuterPositive_init, 0, sampleNumPos, false, samplePositiveBox);
		//samplePositiveBox.clear();
		//samplePositiveBox.resize(sampleNumPos, splitBox[i]);
		getFeatureValue(imageIntegral, samplePositiveBox, 1, samplePositiveFeatureValue);
		/*cout << "sample feature value "<<i<<": ";
		cout << samplePositiveFeatureValue.at<float>(0, 0) << " ";
		cout << samplePositiveFeatureValue.at<float>(1, 0) << " ";
		cout << samplePositiveFeatureValue.at<float>(2, 0) << " ";
		cout << endl;*/
		insertORF(samplePositiveFeatureValue, i);
	}
	orf.update();

	SplitBox(_objectBox, splitBox);
	getFeatureValue(imageIntegral, splitBox, 1, detectFeatureValue);
	checkConfid(detectFeatureValue);

	imshow("neg", neg);
	imshow("confid", confid);
	waitKey(1);
}


double OnlineTree::t = 0;
double OnlineNode::t = 0;
void CompressiveTracker::processFrame(Mat& _frame, Mat& _image_rgb, Rect& _objectBox)
{
	double t1(0), t2(0), t3(0), t4(0), t5(0), t6(0);
	double tc = getTickCount();
	double freq = getTickFrequency();

	//scaleBox(_objectBox, innerBox, 0.7);
	//scaleBox(_objectBox, outerBox, 1.5);
	//backProj(_image_rgb, Rect(), innerBox, backProj_in);
	///backProj(_image_rgb, _objectBox, outerBox, backProj_out);
	//imshow("prob_in",backProj_in);
	//imshow("prob_out", backProj_out);
	//imshow("pos", pos);
	pos = Mat(_frame.size(), CV_8U, Scalar::all(0));
	neg = Mat(_frame.size(), CV_8U, Scalar::all(0));
	Mat like(_frame.size(), CV_8U, Scalar::all(0));
	/*********detect*********/
	OnlineTree::t = 0;
	integral(_frame, imageIntegral, CV_32F);
	UpdateUncertain(_frame, splitBox[0]);

	sampleRect(_frame, _objectBox, rSearchWindowLg, detectBox);
	vector<float> confidence(detectBox.size());
	vector<Rect> boxResize;
	vector<float> confidResize;
	// test different position
	for (int i = 0; i < detectBox.size(); i++){
		confidence[i] = -1;
		if (rng.uniform(0.0, 1.0) > probSearchLg) continue;
		// test different size
		boxResize.clear();
		confidResize.clear();
		for (float scale = 1.0; scale <= 1.1; scale += scaleStep){
			Rect boxTemp;
			float confidTemp = 0;
			vector<float> splitConfid;
			scaleBox(detectBox[i], boxTemp, scale);
			if (!boxInImage(_frame, boxTemp)) continue;
			tc = getTickCount();
			SplitBox(boxTemp, splitBox);
			getFeatureValue(imageIntegral, splitBox, 1, detectFeatureValue);
			t1 += (getTickCount() - tc) / freq;

			evalORF(detectFeatureValue, splitConfid, t2);
			for (int j = 0; j < splitConfid.size(); j++){
				confidTemp += splitConfid[j] / splitConfid.size();
			}
			boxResize.push_back(boxTemp);
			confidResize.push_back(confidTemp);
		}
		int sizeMaxIndex = argmax(confidResize);
		detectBox[i] = boxResize[sizeMaxIndex];
		confidence[i] = confidResize[sizeMaxIndex];
		like.at<uchar>(0.5*(detectBox[i].tl() + detectBox[i].br())) = confidence[i] * 255;
	}
	circle(_image_rgb, 0.5*_objectBox.tl() + 0.5*_objectBox.br(), rSearchWindowLg, Scalar(0, 255, 255));
	_objectBox = detectBox[argmax(confidence)];
	circle(_image_rgb, 0.5*_objectBox.tl() + 0.5*_objectBox.br(), rSearchWindowSm, Scalar(0, 255, 0));

	// test in a restricted area
	sampleRect(_frame, _objectBox, rSearchWindowSm, detectBox);
	confidence.resize(detectBox.size());
	for (int i = 0; i < detectBox.size(); i++){
		confidence[i] = -1;
		// test different size
		boxResize.clear();
		confidResize.clear();
		for (float scale = scaleMin; scale < scaleMax; scale += scaleStepSm){
			Rect boxTemp;
			float confidTemp = 0;
			vector<float> splitConfid;
			scaleBox(detectBox[i], boxTemp, scale);
			if (!boxInImage(_frame, boxTemp)) continue;
			tc = getTickCount();
			SplitBox(boxTemp, splitBox);
			getFeatureValue(imageIntegral, splitBox, 1, detectFeatureValue);
			t1 += (getTickCount() - tc) / freq;

			evalORF(detectFeatureValue, splitConfid, t2);
			for (int j = 0; j < splitConfid.size(); j++){
				confidTemp += splitConfid[j] / splitConfid.size();
			}
			boxResize.push_back(boxTemp);
			confidResize.push_back(confidTemp);
		}
		int sizeMaxIndex = argmax(confidResize);
		detectBox[i] = boxResize[sizeMaxIndex];
		confidence[i] = confidResize[sizeMaxIndex];
		//like.at<uchar>(0.5*(detectBox[i].tl() + detectBox[i].br())) = confidence[i] * 255;
	}
	_objectBox = detectBox[argmax(confidence)];

	//a more detailed detect with different size
	//boxResize.clear();
	//confidResize.clear();
	//// test different size
	//for (float scale = scaleMin; scale < scaleMax; scale += scaleStepSm){
	//	Rect boxTemp;
	//	float confidTemp = 0;
	//	vector<float> splitConfid;
	//	scaleBox(_objectBox, boxTemp, scale);
	//	if (!boxInImage(_frame, boxTemp)) continue;
	//	tc = getTickCount();
	//	SplitBox(boxTemp, splitBox);
	//	getFeatureValue(imageIntegral, splitBox, 1, detectFeatureValue);
	//	t1 += (getTickCount() - tc) / freq;

	//	evalORF(detectFeatureValue, splitConfid, t2);
	//	for (int j = 0; j < splitConfid.size(); j++){
	//		confidTemp += splitConfid[j] / splitConfid.size();
	//	}
	//	boxResize.push_back(boxTemp);
	//	confidResize.push_back(confidTemp);
	//}
	//int sizeMaxIndex = argmax(confidResize);
	//_objectBox = boxResize[sizeMaxIndex];

	/*********update*********/
	orf.refresh();
	vector<float> splitConfid;
	SplitBox(_objectBox, splitBox);
	getFeatureValue(imageIntegral, splitBox, 1, detectFeatureValue);
	evalORF(detectFeatureValue, splitConfid);
	checkConfid(detectFeatureValue);

	sampleNeg(_frame, _objectBox,
		max(rSearchWindowLg, (int)(ratioeMarginOut * splitBox[0].width)),
		max(rSearchWindowLg, (int)(ratioeMarginOut * splitBox[0].height)),
		(int)(ratioMarginIn * splitBox[0].width), (int)(ratioMarginIn * splitBox[0].height),
		sampleNumNeg, splitBox[0].width, splitBox[0].height, sampleNegativeBox);
	getFeatureValue(imageIntegral, sampleNegativeBox, 1, sampleNegativeFeatureValue);
	insertORF(sampleNegativeFeatureValue, split_hrzt*split_vrtc);
	//orf.update();
	sampleNegativeBox.clear();
	for (int i = 0; i < split_hrzt*split_vrtc; i++){

		tc = getTickCount();
		sampleRect(_frame, splitBox[i], rOuterPositive, 0.0, sampleNumPos, false, samplePositiveBox);
		//sampleRect(_frame, splitBox[i], rSearchWindow*1.5, rOuterPositive + 3.0, 100, false, sampleNegativeBox);
		//samplePositiveBox.clear();
		//samplePositiveBox.resize(sampleNumPos, splitBox[i]);
		getFeatureValue(imageIntegral, samplePositiveBox, 1, samplePositiveFeatureValue);
		siftSample(i, samplePositiveFeatureValue, samplePositiveBox, sampleNegativeBox);
		getFeatureValue(imageIntegral, samplePositiveBox, 1, samplePositiveFeatureValue);
		//cout << "samples of patch "<< i << ": "<< samplePositiveBox.size() << " " << sampleNegativeBox.size() << endl;
		t3 += (getTickCount() - tc) / freq;

		tc = getTickCount();
		insertORF(samplePositiveFeatureValue, i);
		t4 += (getTickCount() - tc) / freq;

		//classifierUpdate(samplePositiveFeatureValue, muPositive[i], sigmaPositive[i], learnRate);
		//classifierUpdate(sampleNegativeFeatureValue, muNegative[i], sigmaNegative[i], learnRate);
		rectangle(_frame, splitBox[i], splitConfid[i] * 255);
	}
	getFeatureValue(imageIntegral, sampleNegativeBox, 1, sampleNegativeFeatureValue);
	tc = getTickCount();
	insertORF(sampleNegativeFeatureValue, split_hrzt*split_vrtc);
	t4 += (getTickCount() - tc) / freq;

	tc = getTickCount();
	OnlineNode::t = 0;
	orf.update();
	t6 = OnlineNode::t;
	t5 += (getTickCount() - tc) / freq;

	cout << "get features uses " << t1 << "seconds" << endl;
	cout << "classification uses " << t2  << "seconds" << endl;
	cout << "sample uses " << t3 << "seconds" << endl;
	cout << "insert uses " << t4 << "seconds" << endl;
	cout << "update uses " << t5 << "seconds" << endl;
	cout << "node::eval uses " << t6 << "seconds" << endl;

	rectangle(neg, _objectBox, 32);
	imshow("neg", neg);
	imshow("confid",confid);
	imshow("like", like);
	cout << endl;
}


void CompressiveTracker::testFrame(Mat& _frame, Mat& _image_rgb, Rect& _testBox)
{
	/*********test*********/
	integral(_frame, imageIntegral, CV_32F);
	vector<float> splitConfid;
	SplitBox(_testBox, splitBox);
	getFeatureValue(imageIntegral, splitBox, 1, detectFeatureValue);
	evalORF(detectFeatureValue, splitConfid);
	checkConfid(detectFeatureValue);
	for (int i = 0; i < split_hrzt*split_vrtc; i++) {
		rectangle(_frame, splitBox[i], splitConfid[i] * 255);
	}
	imshow("confid", confid);
	cout << endl;
}