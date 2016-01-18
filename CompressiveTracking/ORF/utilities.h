#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#ifndef WIN32
#include <time.h>
#include <sstream>
#endif

#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

// Random Numbers Generators
unsigned int getDevRandom();

//! Returns a random number in [0, 1]
inline double randDouble() {
    /*static bool didSeeding = false;

    if (!didSeeding) {
#ifdef WIN32
        srand(0);
#else
        unsigned int seedNum;
        struct timeval TV;
        unsigned int curTime;

        gettimeofday(&TV, NULL);
        curTime = (unsigned int) TV.tv_usec;
        seedNum = (unsigned int) time(NULL) + curTime + getpid() + getDevRandom();

        srand(seedNum);
#endif
        didSeeding = true;
    }*/
    return rand() / (RAND_MAX + 1.0);
}

//! Returns a random number in [min, max]
inline double randomFromRange(const double &minRange, const double &maxRange) {
    return minRange + (maxRange - minRange) * randDouble();
}

//! Random permutations
void randPerm(const int &inNum, vector<int> &outVect);
void randPerm(const int &inNum, const int inPart, vector<int> &outVect);

inline void fillWithRandomNumbers(const int &length, vector<double> &inVect) {
    inVect.clear();
    for (int i = 0; i < length; i++) {
        inVect.push_back(2.0 * (randDouble() - 0.5));
    }
}

inline int argmax(const vector<double> &inVect) {
    double maxValue = inVect[0];
    int maxIndex = 0;
	for (int i = 0; i < inVect.size(); i++) {
		if (inVect[i] > maxValue){
			maxValue = inVect[i];
			maxIndex = i;
		}
	}

    return maxIndex;
}

inline int argmax(const vector<float> &inVect) {
	float maxValue = inVect[0];
	int maxIndex = 0;
	for (int i = 0; i < inVect.size(); i++) {
		if (inVect[i] > maxValue){
			maxValue = inVect[i];
			maxIndex = i;
		}
	}

	return maxIndex;
}

inline double avrg(const vector<float> &inVect) {
	if (inVect.size() == 0) return 0;
	float result = 0;
	for (int i = 0; i < inVect.size(); i++) {
		result += inVect[i];
	}
	return result / inVect.size();
}

inline double softmax(const vector<float> &inVect, float alpha, float bias) {
	if (inVect.size() == 0) return 0;
	float numerator = DBL_MIN;
	float dominator = DBL_MIN;
	for (int i = 0; i < inVect.size(); i++) {
		numerator += exp(alpha * abs(inVect[i] - bias)) * inVect[i];
		dominator += exp(alpha * abs(inVect[i] - bias));
	}
	return numerator / dominator;
}

inline int getRank(const vector<float> &inVect, int index){
	int rank = 1;
	for (int i = 0; i < inVect.size(); i++) {
		if (i == index) continue;
		if (inVect[i]>inVect[index])
			rank++;
	}
	return rank;
}

inline double sum(const vector<double> &inVect) {
    double val = 0.0;
    vector<double>::const_iterator itr(inVect.begin()), end(inVect.end());
    while (itr != end) {
        val += *itr;
        ++itr;
    }

    return val;
}

inline void scale(vector<double> &inVect, double scale) {
	for (int i = 0; i < inVect.size(); i++){
		inVect[i] *= scale;
	}
}

inline void add(const vector<double> &inVect, vector<double> &outVect) {
	for (int i = 0; i < inVect.size(); i++){
		outVect[i] += inVect[i];
	}
}

inline void minusVec(const vector<double>& v1, const vector<double> &v2, vector<double>& vecOut){
	vecOut.resize(v1.size());
	for (int i = 0; i < v1.size(); i++){
		vecOut[i] = v1[i] - v2[i];
	}
}

inline void timesVec(const vector<double>& v1, const vector<double> &v2, vector<double>& vecOut){
	vecOut.resize(v1.size());
	for (int i = 0; i < v1.size(); i++){
		vecOut[i] = v1[i] * v2[i];
	}
}

inline void devideVec(const vector<double>& v1, const vector<double> &v2, vector<double>& vecOut){
	vecOut.resize(v1.size());
	for (int i = 0; i < v1.size(); i++){
		assert( v2[i] != 0);
		vecOut[i] = v1[i] / v2[i];
	}
}


inline double dot(const vector<double>& v1, const vector<double>& v2){
	assert(v1.size() == v2.size());
	double proj = 0;
	for (int i = 0; i < v1.size(); i++) {
		proj += v1[i] * v2[i];
	}
	return proj;
}

inline void normalize(vector<double>& vec){
	double div, sqrSum = 0;
	for (int i = 0; i < vec.size(); i++){
		sqrSum += vec[i] * vec[i];
	}
	div = (sqrSum==0)? 0: (1/sqrSum);
	for (int i = 0; i < vec.size(); i++){
		vec[i] *= div;
	}
}


//! Poisson sampling
inline int poisson(double A) {
	int k = 0;
	int maxK = 10;
	while (1) {
		double U_k = randDouble();
		A *= U_k;
		if (k > maxK || A < exp(-1.0)) {
			break;
		}
		k++;
	}
	return k;
}

// OpenCV

inline void scaleBox(Rect& inputBox, Rect& outputBox, float scale){
	Point2f center, diag;
	diag = Point2f(0.5 * inputBox.width, 0.5 * inputBox.height);
	center = Point2f(inputBox.x, inputBox.y) + diag;
	diag *= scale;
	outputBox = Rect(center - diag, center + diag);
}

inline void alignBox(Rect& inputBox, Rect& alignTarget,Rect& outputBox){
	Point2f center, diag;
	Point2d diagFull;
	diag = Point2f(0.5 * inputBox.width, 0.5 * inputBox.height);
	diagFull = Point2d(inputBox.width, inputBox.height);
	center = 0.5 * Point2f(alignTarget.tl()) + 0.5 * Point2f(alignTarget.br());
	outputBox = Rect(Point2d(center - diag), Point2d(center - diag) + diagFull);
}

inline void scaleAndMoveBox(Rect& inputBox, Rect& outputBox, float scale){
	Point center, diag;
	diag = Point(0.5 * inputBox.width, 0.5 * inputBox.height);
	center = Point(inputBox.x, inputBox.y) + diag;
	diag *= scale;
	outputBox = Rect(center - diag, center + diag);
	outputBox -= (1 - scale)*center;
}

inline bool boxInImage(Mat& image, Rect& box){
	Rect bound(0, 0, image.cols, image.rows);
	return(bound.contains(box.tl()) && bound.contains(box.br()));
}


inline void minMaxRows(Mat& mat1, Mat& mat2, vector<double>& minVec, vector<double>& maxVec){
	assert(mat1.rows == mat2.rows);
	minVec.resize(mat1.rows);
	maxVec.resize(mat1.rows);
	for (int i = 0; i < mat1.rows; i++){
		float _min = DBL_MAX;
		float _max = -DBL_MAX;
		for (int j = 0; j < mat1.cols; j++){
			float temp = mat1.at<float>(i, j);
			_min = min(_min, temp);
			_max = max(_max, temp);
		}
		for (int j = 0; j < mat2.cols; j++){
			float temp = mat2.at<float>(i, j);
			_min = min(_min, temp);
			_max = max(_max, temp);
		}
		minVec[i] = _min;
		maxVec[i] = _max;
	}
}



class SeqCapture : public VideoCapture
{
private:
	bool readImage(Mat& image){
		if (index > endIndex)
			return false;
		string frameName;
		stringstream ss;
		ss << index;
		ss >> frameName;
		if (length > 0){
			int numZero = length - frameName.length();
			frameName = string(numZero, '0') + frameName;
		}
		frameName = dir + '/' + frameName + format;
		image = imread(frameName);
		index++;
		return image.data;
	}

public:
	bool isOpen;
	bool useSeq;
	int index;
	int endIndex;
	int length;
	string dir;
	string format;
	Mat frame;

	SeqCapture() :isOpen(false), useSeq(false), index(0), endIndex(INT_MAX), length(0), format(".jpg"){};

	virtual bool open(const string& fileName, bool useSequence){
		useSeq = useSequence;
		cout << useSeq;
		if (useSeq){
			dir = fileName;
			isOpen = readImage(frame);
			index--;
			return isOpen;
		}
		else{
			return VideoCapture::open(fileName);
		}
	}

	virtual bool open(int device){
		return VideoCapture::open(device);
	}

	virtual bool isOpened() const{
		if (useSeq)
			return isOpen;
		else
			return VideoCapture::isOpened();
	}

	virtual bool read(Mat& image){
		if (useSeq)
			return readImage(image);
		else
			return VideoCapture::read(image);
	}

	virtual SeqCapture& operator >> (Mat& image){
		if (useSeq)
			readImage(image);
		else
			VideoCapture::operator>>(image);
		return *this;
	}

	void setNameFormat(int nameLength, const string& subfix){
		length = nameLength;
		format = subfix;
	}

	void setRange(int start, int end){
		index = start;
		endIndex = end;
	}
};


class BoxWriter{
private:
	ofstream file;
public:
	void open(const string& filename){
		file.open(filename);
	};
	void write(Rect& box){
		file << box.x << "," << box.y << "," << box.width << "," << box.height << endl;
	};
	void close(){
		file.close();
	};
};


#endif /* UTILITIES_H_ */
