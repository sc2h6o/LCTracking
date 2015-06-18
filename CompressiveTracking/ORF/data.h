#ifndef DATA_H_
#define DATA_H_
//#define USE_DATASET

#include <iostream>
#include <vector>
// #include <gmm.h>
#include <string>

using namespace std;
// using namespace gmm;

// TYPEDEFS
typedef int Label;
typedef double Weight;
typedef vector<double> SparseVector;

// DATA CLASSES
class Sample {
public:
    SparseVector x;
    Label y;
    Weight w;

    void disp() {
        cout << "Sample: y = " << y << ", w = " << w << ", x = ";
        // cout << x << endl;
    }
};

#ifdef USE_DATASET
class DataSet {
public:
    vector<Sample> m_samples;
    int m_numSamples;
    int m_numFeatures;
    int m_numClasses;

    vector<double> m_minFeatRange;
    vector<double> m_maxFeatRange;

    void findFeatRange();

    void loadLIBSVM(string filename);
};
#endif

class Result {
public:
    vector<double> confidence;
    int prediction;

	virtual inline void zero() {
		memset(confidence.data(), 0, confidence.size()*sizeof(double));
	}
};

#endif /* DATA_H_ */
