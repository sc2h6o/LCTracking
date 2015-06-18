#ifndef ONLINETREE_H_
#define ONLINETREE_H_

#include <cstdlib>
#include <iostream>

#include "classifier.h"
#include "data.h"
#include "hyperparameters.h"
#include "onlinenode.h"

using namespace std;

class OnlineTree: public Classifier {
public:
    OnlineTree(const Hyperparameters &hp, int numClasses) :
		m_counter(0.0), m_hp(&hp), testScore(0),testCnt(0),height(0){
        m_rootNode = new OnlineNode(hp, numClasses, 0);
    }

    ~OnlineTree() {
        delete m_rootNode;
    }

	virtual void insert(Sample &sample) {
		m_rootNode->insert(sample);
	}

    virtual void update() {
		m_rootNode->update(height);
		height--; // to keep same with root
	}

	static double t;
	virtual void eval(Sample &sample, Result& result) {
		m_rootNode->eval(sample, result);
	}

	void test(Sample& sample){
		Result result;
		eval(sample, result);
		testScore += result.confidence[sample.y];
		testCnt++;
	}

	double score(){
		return (testScore / testCnt);
	}

	void foregetScore(){
		testCnt *= 1;
		testScore *= 1;
	}

	bool isMature(){
		return (height > m_hp->MatureDepth);
	}

#ifdef USE_DATASET
    virtual void train(DataSet &dataset) {
        vector<int> randIndex;
        int sampRatio = dataset.m_numSamples / 10;
        for (int n = 0; n < m_hp->numEpochs; n++) {
            randPerm(dataset.m_numSamples, randIndex);
            for (int i = 0; i < dataset.m_numSamples; i++) {
                update(dataset.m_samples[randIndex[i]]);
                if (m_hp->verbose >= 3 && (i % sampRatio) == 0) {
                    cout << "--- Online Random Tree training --- Epoch: " << n + 1 << " --- ";
                    cout << (10 * i) / sampRatio << "%" << endl;
                }
            }
        }
    }

    virtual vector<Result> test(DataSet &dataset) {
        vector<Result> results;
        for (int i = 0; i < dataset.m_numSamples; i++) {
            results.push_back(eval(dataset.m_samples[i]));
        }

        double error = compError(results, dataset);
        if (m_hp->verbose >= 3) {
            cout << "--- Online Random Tree test error: " << error << endl;
        }

        return results;
    }

    virtual vector<Result> trainAndTest(DataSet &dataset_tr, DataSet &dataset_ts) {
        vector<Result> results;
        vector<int> randIndex;
        int sampRatio = dataset_tr.m_numSamples / 10;
        vector<double> testError;
        for (int n = 0; n < m_hp->numEpochs; n++) {
            randPerm(dataset_tr.m_numSamples, randIndex);
            for (int i = 0; i < dataset_tr.m_numSamples; i++) {
                update(dataset_tr.m_samples[randIndex[i]]);
                if (m_hp->verbose >= 3 && (i % sampRatio) == 0) {
                    cout << "--- Online Random Tree training --- Epoch: " << n + 1 << " --- ";
                    cout << (10 * i) / sampRatio << "%" << endl;
                }
            }

            results = test(dataset_ts);
            testError.push_back(compError(results, dataset_ts));
        }

        if (m_hp->verbose >= 3) {
            cout << endl << "--- Online Random Tree test error over epochs: ";
            dispErrors(testError);
        }

        return results;
    }
#endif

private:
    double m_counter;
	double testScore;
	double testCnt;
	int height;
    const Hyperparameters *m_hp;

	OnlineNode* m_rootNode;
};

#endif /* ONLINETREE_H_ */
