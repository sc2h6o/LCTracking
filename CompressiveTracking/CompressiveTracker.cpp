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
	split_num = 12;
	split_hrzt = 4;		// number of splitted segments
	split_vrtc = 4;
	classNum = split_hrzt * split_vrtc + 2;
	overlap = 0.3;
	ratioUncertain = 0.0;
	sampleNumPos = 30;
	sampleNumNeg = 300;
	sampleNumNegInit = 200;
	sampleNumUncertain = 200;
	ratioeMarginOut = 1.5;
	ratioMarginIn = 0.75;
	thrsdBG = 3.5;
	thrsdSelf = 999;
	thrsdUncertain = 2.5;
	ratioOrigin = 0.6;
	eqhst = false;
	alpha = 3;
	bias = 0.5;

	rOuterPositive_init = 3;
	rOuterPositive = 3;	// radical scope of positive samples
	rSearchWindowLg = 35; // size of search window
	rSearchWindowSm = 5; // size of search window
	searchRoundLg = 1;
	searchRoundSm = 1;
	probSearchLg = 0.1;
	probSearchSm = 1.0;
	scaleMin = 1.00f;
	scaleMax = 1.01f;
	scaleStep = 0.5f;
	scaleStepSm = 0.02f;
	mu.resize(featureNum, 0);
	sigma.resize(featureNum, 0);
	learnRate = 0.85f;	// Learning rate parameter
	lastConfid = 1.0;
	// hyperparameter for orf
	hp.counterThreshold = 70;
	hp.maxDepth = 30;
	hp.useUncertain = true;
	hp.splitRatioUncertain = 2.5;  // a node could split when samples are less than ratio*uncertains
	hp.numRandomTests = 24;
	hp.numTrees = 4;
	hp.numGrowTrees = 1;
	hp.MatureDepth = 18;
	hp.useSoftVoting = true;
	hp.verbose = 1;
	hp.lamda = 1.0;
	hpOrigin = hp;
	hpOrigin.numTrees = 10;
	hpOrigin.counterThreshold = 40;
}

CompressiveTracker::~CompressiveTracker(void)
{
}

void CompressiveTracker::SplitBox(Rect _objectBox, vector<Rect>& splitBox){
	int width_sm = _objectBox.width / ((split_hrzt-1) * (1-overlap)+1);
	int height_sm = _objectBox.height / ((split_vrtc - 1) * (1 - overlap) + 1);
	splitBox.resize(split_hrzt * split_vrtc);
	
	for (int i = 0; i < split_hrzt; i++){
		for (int j = 0; j < split_vrtc; j++){
			splitBox[i*split_vrtc + j] = Rect(_objectBox.x + width_sm*(1 - overlap)*i,
				_objectBox.y + height_sm*(1 - overlap)*j, width_sm, height_sm);
		}
	}
	// splitBox[split_hrzt*split_vrtc - 1] = _objectBox;
}

void CompressiveTracker::HaarFeature(int _numFeature)
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

void CompressiveTracker::UpdateUncertain(Mat imageIntegral, Rect& splitBox){
	Scalar muTemp;
	Scalar sigmaTemp;

	vector<Rect> uncertainDetect(sampleNumUncertain);
	int searchWidth = imageIntegral.cols - splitBox.width - 1;
	int searchHeight = imageIntegral.rows - splitBox.height - 1;
	for (int i = 0; i < sampleNumUncertain; i++){
		uncertainDetect[i].x = rng.uniform(0, searchWidth);
		uncertainDetect[i].y = rng.uniform(0, searchHeight);
		uncertainDetect[i].width = splitBox.width;
		uncertainDetect[i].height = splitBox.height;
	}

	Mat sampleUncertainFeatureValue;
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

void CompressiveTracker::sampleRect(Mat& _image, Rect& _objectBox, Rect bound, float _rOuter, float _rInner, int _maxSampleNum, bool checkBG, vector<Rect>& _sampleBox, vector<Weight>& _sampleWeight)
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

	/*minrow = max(minrow, bound.y);
	maxrow = min(maxrow, bound.y+bound.height);
	mincol = max(mincol, bound.x);
	maxcol = min(maxcol, bound.x + bound.width);*/

	int i = 0;

	float prob = ((float)(_maxSampleNum)) / (maxrow - minrow + 1) / (maxcol - mincol + 1);

	int r;
	int c;

	_sampleBox.clear();//important
	_sampleWeight.clear();
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
				_sampleWeight.push_back(1.0);
				i++;
			}
		}
	}
	_sampleBox.resize(i);
		
}

void CompressiveTracker::sampleRect(Mat& _image, Rect& _objectBox, float _srw, vector<Rect>& _sampleBox, vector<float>& _prior)
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
	_prior.clear();

	for (r = minrow; r <= (int)maxrow; r++){
		for (c = mincol; c <= (int)maxcol; c++){
			dist = (_objectBox.y - r)*(_objectBox.y - r) + (_objectBox.x - c)*(_objectBox.x - c);

			if (dist < inradsq){

				rec.x = c;
				rec.y = r;
				rec.width = _objectBox.width;
				rec.height = _objectBox.height;

				_sampleBox.push_back(rec);
				_prior.push_back(exp(-dist * 1.0 / 8100));
				i++;
			}
		}
	}

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
	float reproArea;
	int xMin;
	int xMax;
	int yMin;
	int yMax;

	for (int i = 0; i<featureNum; i++)
	{
		for (int j = 0; j<sampleBoxSize; j++)
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
				reproArea = (tempArea == 0) ? 0 : (1.0 / tempArea);
				tempValue += reproArea * featuresWeight[i][k] *
					(_imageIntegral.at<float>(yMin, xMin) +
					_imageIntegral.at<float>(yMax, xMax) -
					_imageIntegral.at<float>(yMin, xMax) -
					_imageIntegral.at<float>(yMax, xMin));
			}
			_sampleFeatureValue.at<float>(i,j) = tempValue;
		}
	}
	//for (int j = 0; j < _sampleFeatureValue.cols; j++){
	//	Mat col;
	//	col = _sampleFeatureValue(Rect(j, 0, 1, _sampleFeatureValue.rows));
	//	//normalize(col, col, 1, NORM_L2);
	//}
}

void CompressiveTracker::insertORF(OnlineRF& orf, Mat& _sampleFeatureValue, int label, vector<Weight>& _sampleWeight)
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

void CompressiveTracker::evalORF(OnlineRF& orf, Mat& _splitFeatureValue, vector<float>& confidence, vector<float>& suspicion)
{
	confidence.resize(_splitFeatureValue.cols, 0); // plus 1 for the background class
	suspicion.resize(_splitFeatureValue.cols, 0); // plus 1 for the background class
	for (int i = 0; i < _splitFeatureValue.cols; i++){
		Sample sample;
		sample.x.resize(_splitFeatureValue.rows);
		for (int j = 0; j < _splitFeatureValue.rows; j++){
			sample.x[j] = _splitFeatureValue.at<float>(j, i);
		}
		sample.w = 1.0;
		Result result;
		orf.eval(sample, result);

		//confidence[i] = 0.5 * (result.confidence[i] - 1 * result.confidence[split_hrzt*split_vrtc] + 1);
		confidence[i] = result.confidence[i];
		suspicion[i] = result.confidence[split_hrzt*split_vrtc];
		//confidence[i] = result.confidence[i] / (result.confidence[i] + result.confidence[split_hrzt*split_vrtc]);
	}
}

float CompressiveTracker::crossEval(vector<float>& splitConfid, vector<float>& splitSuspc){
	float c = 1.0, d = 1.0;
	for (int i = 0; i < splitConfid.size(); i++){
		c *= 1 - splitConfid[i];
		d *= 1 - splitSuspc[i];
	}
	return (1 - c) * d;
}

bool OnlineNode::echo = false;
void CompressiveTracker::checkConfid(OnlineRF& orf, Mat& _splitFeatureValue)
{
	OnlineNode::echo = false;
	int splitNum = _splitFeatureValue.cols;
	for (int i = 0; i < _splitFeatureValue.cols; i++){
		Sample sample;
		sample.x.resize(_splitFeatureValue.rows);
		for (int j = 0; j < _splitFeatureValue.rows; j++){
			sample.x[j] = _splitFeatureValue.at<float>(j, i);
		}
		sample.w = 1.0;

		Result result;
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

void CompressiveTracker::siftSample(int splitIndex, Mat& sampleFeatureValue, vector<Rect>& sampleBoxes, vector<Weight>& sampleWeight){
	vector<Rect> sample_new;
	vector<Weight> weight_new;
	for (int i = 0; i < sampleFeatureValue.cols; i++){
		Sample sample;
		sample.x.resize(sampleFeatureValue.rows);
		for (int j = 0; j < sampleFeatureValue.rows; j++){
			sample.x[j] = sampleFeatureValue.at<float>(j, i);
		}
		sample.w = 1.0;

		Result result;
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
		if (splitIndex < split_hrzt*split_vrtc &&
			confid_bg > thrsdBG * confid_fg )
			//ratioUncertain*confid_bg > thrsdUncertain*result.confidence[split_hrzt*split_vrtc + 1])
		{
			// pick out false positive samples
			// sampleNegativeBox.push_back(samplePositiveBox[i]);
			rectangle(neg, sampleBoxes[i], confid_bg * 255, -1);
		}
		else if(confid_self > thrsdSelf * confid_other){
			// pick out fully coonfident patches
			rectangle(neg, sampleBoxes[i], 10, 1);
		}
		else{
			sample_new.push_back(sampleBoxes[i]);
			weight_new.push_back(sampleWeight[i]);
		}
	}
	sampleBoxes = sample_new;
	sampleWeight = weight_new;
}

Rect CompressiveTracker::meanShift(vector<float> confidence, vector<Rect>detectBox){
	if (detectBox.empty()) return Rect(0, 0, 0, 0);
	Point tl(0, 0);
	Point diag = (detectBox[0].br() - detectBox[0].tl());
	float confidSum = 0;
	float alpha = 2;
	for (int i = 0; i < detectBox.size(); i++){
		tl += exp(alpha * confidence[i]) * detectBox[i].tl();
		confidSum += exp(alpha * confidence[i]);
	}
	if (confidSum == 0) return detectBox[0];
	tl = (1.0 / confidSum) * tl;
	return Rect(tl, tl + diag);
}

void CompressiveTracker::equalHistPart(Mat& imageIn, Mat& imageOut, Rect& _objectBox, int rSearchWindowLg){
	int x_min = max(0, _objectBox.x - rSearchWindowLg);
	int y_min = max(0, _objectBox.y - rSearchWindowLg);
	int x_max = min(imageIn.cols, _objectBox.x + _objectBox.width + rSearchWindowLg);
	int y_max = min(imageIn.rows, _objectBox.y + _objectBox.height + rSearchWindowLg);
	Rect region(x_min, y_min, x_max - x_min, y_max - y_min);
	equalizeHist(imageIn(region), imageOut(region));
}

void CompressiveTracker::GammaCorrection(Mat& imageIn, Mat& imageOut, float fGamma)
{
	imageOut = imageIn.clone();

	// build look up table
	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
	{
		lut[i] = pow((float)(i / 255.0), fGamma) * 255.0;
	}

	for (int i = 0; i < imageIn.rows; i++){
		for (int j = 0; j < imageIn.cols; j++){
			imageOut.at<uchar>(i, j) = lut[imageIn.at<uchar>(i, j)];
		}
	}
			


}

void CompressiveTracker::searchRegion(Mat& _frame, Mat& imageIntegral, int radius, float detectProb,
	float scaleMin, float scaleMax, float scaleStep, Rect& boxTemp, float& confidTemp)
{
	vector <Rect> detectBox;
	vector<float> confidence;
	vector<float> prior;
	vector<Rect> boxResize;
	vector<float> confidResize;
	sampleRect(_frame, boxTemp, radius, detectBox, prior);
	confidence.resize(detectBox.size());
	// test different position
	for (int i = 0; i < detectBox.size(); i++){
		confidence[i] = -1;
		// test different size
		boxResize.clear();
		confidResize.clear();
		for (float scale = scaleMin; scale <= scaleMax; scale += scaleStep){
			if (rng.uniform(0.0, 1.0) > detectProb) continue;
			Rect boxTemp;
			float confidTemp = 0;
			vector<Rect> splitBox;
			vector<float> splitConfid;
			vector<float> splitSuspc;
			Mat detectFeatureValue;
			scaleBox(detectBox[i], boxTemp, scale);
			if (!boxInImage(_frame, boxTemp)) continue;
			SplitBox(boxTemp, splitBox);
			getFeatureValue(imageIntegral, splitBox, 1, detectFeatureValue);

			evalORF(orf, detectFeatureValue, splitConfid, splitSuspc);
			confidTemp = softmax(splitConfid, alpha, bias);
			//confidTemp = crossEval(splitConfid, splitSuspc);
			evalORF(orfOrigin, detectFeatureValue, splitConfid, splitSuspc);
			confidTemp = (1 - ratioOrigin)*confidTemp + ratioOrigin*softmax(splitConfid, alpha, bias);
			//confidTemp = (1 - ratioOrigin)*confidTemp + ratioOrigin*crossEval(splitConfid, splitSuspc);

			boxResize.push_back(boxTemp);
			confidResize.push_back(confidTemp);
		}
		if (confidResize.empty()) continue;
		int sizeMaxIndex = argmax(confidResize);
		detectBox[i] = boxResize[sizeMaxIndex];
		confidence[i] = prior[i] * confidResize[sizeMaxIndex];
		like.at<uchar>(0.5*(detectBox[i].tl() + detectBox[i].br())) = confidence[i] * 255;
	}
	int maxIndex = argmax(confidence);
	if (confidence[maxIndex] > confidTemp){
		boxTemp = detectBox[maxIndex];
		confidTemp = confidence[maxIndex];
	}
}

void CompressiveTracker::detect(Mat& _frame, Mat& _image_rgb, Mat& imageIntegral, Rect& _objectBox)
{
	vector<Rect> candidates;
	vector<float> confid;
	circle(_image_rgb, 0.5*_objectBox.tl() + 0.5*_objectBox.br(), rSearchWindowLg, Scalar(0, 255, 255));
	for (int i = 0; i < searchRoundLg; i++){
		Rect boxOld, boxNew = _objectBox;
		float confidTemp = 0;
		searchRegion(_frame, imageIntegral,rSearchWindowLg, probSearchLg, 1.0, 1.0, 1, boxNew, confidTemp);
		circle(_image_rgb, 0.5*boxNew.tl() + 0.5*boxNew.br(), rSearchWindowSm, Scalar(0, 255, 0));
		searchRegion(_frame, imageIntegral, rSearchWindowSm, probSearchSm, scaleMin, scaleMax, scaleStepSm, boxNew, confidTemp);
		/*for (int j = 0; j < searchRoundSm && boxOld!=boxNew; j++){
			boxOld = boxNew;
			searchRegion(_frame, imageIntegral, rSearchWindowSm, probSearchSm, 1.0, 1.0, scaleStepSm, boxNew, confidTemp);
			searchRegion(_frame, imageIntegral, 1, probSearchSm, scaleMin, scaleMax, scaleStepSm, boxNew, confidTemp);
		}*/
		candidates.push_back(boxNew);
		confid.push_back(confidTemp);
	}
	_objectBox = candidates[argmax(confid)];
	lastConfid = confid[argmax(confid)];

}

void CompressiveTracker::init(Mat& _frame, Mat& _image_rgb, Rect& _objectBox)
{
	pos = Mat(_frame.size(), CV_8U, Scalar::all(0));
	neg = Mat(_frame.size(), CV_8U, Scalar::all(0));
	confid = Mat(400, 400, CV_8U, Scalar(0));
	
	// randomly choose parts
	// split_num = min(split_num, _objectBox.width * _objectBox.height / 100 );
	split_hrzt = (int)round(sqrt(_objectBox.width * split_num / _objectBox.height));
	split_vrtc = (int)round(sqrt(_objectBox.height * split_num / _objectBox.width));

	// init the Online Random Forest
	orf.init(hp, classNum);
	orfOrigin.init(hpOrigin, classNum);
	
	// compute feature template
	HaarFeature(featureNum);

	// compute sample templates
	vector<Rect> splitBox;
	SplitBox(_objectBox, splitBox);
	if (eqhst)
		equalHistPart(_frame, _frame, _objectBox, rSearchWindowLg);
	//GammaCorrection(_frame, _frame, 1.5);
	integral(_frame, imageIntegral, CV_32F);
	UpdateUncertain(imageIntegral, splitBox[0]);

	vector<Rect> samplePositiveBox;
	vector<Rect> sampleNegativeBox;
	vector<Weight> sampleWeight;
	Mat samplePositiveFeatureValue;
	Mat sampleNegativeFeatureValue;
	initPositiveFeatureValues.resize(split_hrzt*split_vrtc);
	sampleNeg(_frame, _objectBox,
		max((int)(ratioeMarginOut * rSearchWindowLg), (int)(ratioeMarginOut * splitBox[0].width)),
		max((int)(ratioeMarginOut * rSearchWindowLg), (int)(ratioeMarginOut * splitBox[0].height)),
		(int)(ratioMarginIn * splitBox[0].width), (int)(ratioMarginIn * splitBox[0].height),
		sampleNumNegInit, splitBox[0].width, splitBox[0].height, sampleNegativeBox);
	getFeatureValue(imageIntegral, sampleNegativeBox, 1, sampleNegativeFeatureValue);
	sampleWeight.resize(sampleNegativeBox.size(), 1.0);
	insertORF(orf, sampleNegativeFeatureValue, split_hrzt*split_vrtc, sampleWeight);
	insertORF(orfOrigin, sampleNegativeFeatureValue, split_hrzt*split_vrtc, sampleWeight);
	
	for (int i = 0; i < split_hrzt*split_vrtc; i++){
		sampleRect(_frame, splitBox[i], _objectBox, rOuterPositive_init, 0, sampleNumPos, false, samplePositiveBox, sampleWeight);
		getFeatureValue(imageIntegral, samplePositiveBox, 1, samplePositiveFeatureValue);
		initPositiveFeatureValues[i] = samplePositiveFeatureValue.clone();
		insertORF(orf, samplePositiveFeatureValue, i, sampleWeight);
		insertORF(orfOrigin, samplePositiveFeatureValue, i, sampleWeight);
	}
	orf.update();
	orfOrigin.update();

	Mat detectFeatureValue;
	SplitBox(_objectBox, splitBox);
	getFeatureValue(imageIntegral, splitBox, 1, detectFeatureValue);
	checkConfid(orf, detectFeatureValue);
	//imshow("neg", neg);
	//imshow("confid", confid);
	waitKey(1);
}


double OnlineTree::t = 0;
double OnlineNode::t = 0;
void CompressiveTracker::processFrame(Mat& _frame, Mat& _image_rgb, Rect& _objectBox)
{
	double t1(0), t2(0), t3(0), t4(0), t5(0), t6(0);
	double tc = getTickCount();
	double tc6 = getTickCount();
	double freq = getTickFrequency();
	
	pos = Mat(_frame.size(), CV_8U, Scalar::all(0));
	neg = Mat(_frame.size(), CV_8U, Scalar::all(0));
	like = Mat(_frame.size(), CV_8U, Scalar::all(0));
	vector<Rect> splitBox;
	SplitBox(_objectBox, splitBox);
	/*********detect*********/
	OnlineTree::t = 0;
	if (eqhst) 
		equalHistPart(_frame, _frame, _objectBox, rSearchWindowLg);
	//GammaCorrection(_frame, _frame, 1.5);
	integral(_frame, imageIntegral, CV_32F);
	UpdateUncertain(imageIntegral, splitBox[0]);

	tc = getTickCount();
	detect(_frame, _image_rgb, imageIntegral, _objectBox);
	t2 += (getTickCount() - tc) / freq;
	/*********update*********/
	orf.refresh();
	vector<float> splitConfid;
	vector<float> splitSuspc;
	Mat detectFeatureValue;
	SplitBox(_objectBox, splitBox);
	getFeatureValue(imageIntegral, splitBox, 1, detectFeatureValue);
	evalORF(orf, detectFeatureValue, splitConfid, splitSuspc);
	checkConfid(orf, detectFeatureValue);


	vector<Rect> samplePositiveBox;
	vector<Rect> sampleNegativeBox;
	vector<Weight> sampleWeight;
	Mat samplePositiveFeatureValue;
	Mat sampleNegativeFeatureValue;
	sampleNeg(_frame, _objectBox,
		max((int)(ratioeMarginOut * rSearchWindowLg), (int)(ratioeMarginOut * splitBox[0].width)),
		max((int)(ratioeMarginOut * rSearchWindowLg), (int)(ratioeMarginOut * splitBox[0].height)),
		(int)(ratioMarginIn * splitBox[0].width), (int)(ratioMarginIn * splitBox[0].height),
		sampleNumNeg, splitBox[0].width, splitBox[0].height, sampleNegativeBox);
	sampleWeight.resize(sampleNegativeBox.size(), 1.0);
	getFeatureValue(imageIntegral, sampleNegativeBox, 1, sampleNegativeFeatureValue);
	siftSample(split_hrzt*split_vrtc, sampleNegativeFeatureValue, sampleNegativeBox, sampleWeight);
	getFeatureValue(imageIntegral, sampleNegativeBox, 1, sampleNegativeFeatureValue);
	//orf.update();
	//orfOrigin.init(hpOrigin, classNum);
	for (int i = 0; i < split_hrzt*split_vrtc; i++){
		tc = getTickCount();
		sampleRect(_frame, splitBox[i], _objectBox, rOuterPositive, 0.0, sampleNumPos, false, samplePositiveBox, sampleWeight);
		getFeatureValue(imageIntegral, samplePositiveBox, 1, samplePositiveFeatureValue);
		siftSample(i, samplePositiveFeatureValue, samplePositiveBox, sampleWeight);
		getFeatureValue(imageIntegral, samplePositiveBox, 1, samplePositiveFeatureValue);
		//cout << "samples of patch "<< i << ": "<< samplePositiveBox.size() << " " << sampleNegativeBox.size() << endl;
		t3 += (getTickCount() - tc) / freq;

		tc = getTickCount();
		insertORF(orf, samplePositiveFeatureValue, i, sampleWeight);
		//if (lastConfid >= 0.5)
		//	insertORF(orfOrigin, initPositiveFeatureValues[i], i, sampleWeight);
		t4 += (getTickCount() - tc) / freq;

		rectangle(_frame, splitBox[i], splitConfid[i] * 255);
	}
	tc = getTickCount();
	insertORF(orf, sampleNegativeFeatureValue, split_hrzt*split_vrtc, sampleWeight);
	//insertORF(orfOrigin, sampleNegativeFeatureValue, split_hrzt*split_vrtc, sampleWeight);
	t4 += (getTickCount() - tc) / freq;

	tc = getTickCount();
	orf.update();
	t5 += (getTickCount() - tc) / freq;
	t6 = (getTickCount() - tc6) / freq;
	cout << "get features uses " << t1 << "seconds" << endl;
	cout << "classification uses " << t2  << "seconds" << endl;
	cout << "sample uses " << t3 << "seconds" << endl;
	cout << "insert uses " << t4 << "seconds" << endl;
	cout << "update uses " << t5 << "seconds" << endl;
	cout << "in all uses " << t6 << "seconds" << endl;
	//if (lastConfid>0.5)
	//	cout << "orfOrigin updated" << endl;
	//rectangle(neg, _objectBox, 32);
	//imshow("neg", neg);
	imshow("confid",confid);
	//imshow("like", like);
	cout << endl;
}


void CompressiveTracker::testFrame(Mat& _frame, Mat& _image_rgb, Rect& _testBox)
{
	/*********test*********/
	integral(_frame, imageIntegral, CV_32F);
	vector<Rect> splitBox;
	vector<float> splitConfid;
	vector<float> splitSuspc;
	Mat detectFeatureValue;
	SplitBox(_testBox, splitBox);
	getFeatureValue(imageIntegral, splitBox, 1, detectFeatureValue);
	evalORF(orf, detectFeatureValue, splitConfid, splitSuspc);
	checkConfid(orf, detectFeatureValue);
	for (int i = 0; i < split_hrzt*split_vrtc; i++) {
		rectangle(_frame, splitBox[i], splitConfid[i] * 255);
	}
	imshow("confid", confid);
	cout << endl;
}