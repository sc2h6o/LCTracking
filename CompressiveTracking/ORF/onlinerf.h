#ifndef ONLINERF_H_
#define ONLINERF_H_

#include "classifier.h"
#include "data.h"
#include "hyperparameters.h"
#include "onlinetree.h"
#include "utilities.h"

class OnlineRF: public Classifier {
public:
	OnlineRF(){};

    ~OnlineRF() {
        for (int i = 0; i < m_hp.numTrees; i++) {
            delete m_trees[i];
        }
	}

	void init(const Hyperparameters &hp, int numClasses)
	{
		m_numClasses = numClasses;
		m_counter = 0.0;
		m_hp = hp;
		// at first init all trees, then discard some to establish grow trees
		OnlineTree *tree;
		for (int i = 0; i < hp.numTrees; i++) {
			tree = new OnlineTree(hp, numClasses);
			m_trees.push_back(tree);
		}
		m_oobe.resize(hp.numTrees, 0);
		treeResult.confidence.resize(numClasses, 0);
	}

	virtual void insert(Sample &sample) {
		m_counter += sample.w;

		int numTries;
		for (int i = 0; i < m_hp.numTrees; i++) {
			numTries = poisson(m_hp.lamda);
			if (numTries) {
				for (int n = 0; n < 2; n++) {
					m_trees[i]->insert(sample);
				}
			}
			else {
				m_trees[i]->addTest(sample);
			}
		}
	}

    virtual void update() {
		for (int i = 0; i < m_hp.numTrees; i++) {
			m_trees[i]->update();
		}
	}

	virtual void eval(Sample &sample, Result& result) {
		if (!result.confidence.size())
			result.confidence.resize(m_numClasses);
		result.zero();
		for (int i = 0; i < m_hp.numTrees; i++) {
			m_trees[i]->eval(sample, treeResult);
			if (m_hp.useSoftVoting) {
				add(treeResult.confidence, result.confidence);
			} else {
				result.confidence[treeResult.prediction]++;
			}
		}
		scale(result.confidence, 1.0 / m_hp.numTrees);
		result.prediction = argmax(result.confidence);

	}

	// check whethor to replace trees
	// use before start to insert
	void refresh(){
		// the grow trees are at the tail of the vector
		bool flag = false;
		double minScore, minIndex;
		for (int i = m_hp.numTrees - m_hp.numGrowTrees; i < m_hp.numTrees; i++){
			if (!m_trees[i]->isMature()) continue;
			flag = true;

			//replace the worst tree
			minScore = DBL_MAX;
			minIndex = -1;
			for (int j = 0; j < m_hp.numTrees - m_hp.numGrowTrees; j++){
				if (m_trees[j]->score() < minScore){
					minScore = m_trees[j]->score();
					minIndex = j;
				}
				else{

				}
			}
			delete m_trees[minIndex];
			m_trees[minIndex] = m_trees[i];
			m_trees[i] = new OnlineTree(m_hp, m_numClasses);
			m_trees[i]->foregetScore();

			// if some replacement happened, forget some scores
			if (flag){
				cout << "*****************************************" << endl << endl;
				cout << "Forest Refreshed for " << minIndex << endl << endl;
				cout << "*****************************************" << endl;
				/*for (int i = 0; i < m_hp.numTrees - m_hp.numGrowTrees; i++)
				m_trees[i]->foregetScore();*/
			}
		}
	}

#ifdef USE_DATSET

    virtual void train(DataSet &dataset) {
        vector<int> randIndex;
        int sampRatio = dataset.m_numSamples / 10;
        for (int n = 0; n < m_hp->numEpochs; n++) {
            randPerm(dataset.m_numSamples, randIndex);
            for (int i = 0; i < dataset.m_numSamples; i++) {
                update(dataset.m_samples[randIndex[i]]);
                if (m_hp->verbose >= 1 && (i % sampRatio) == 0) {
                    cout << "--- Online Random Forest training --- Epoch: " << n + 1 << " --- ";
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
        if (m_hp->verbose >= 1) {
            cout << "--- Online Random Forest test error: " << error << endl;
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
                if (m_hp->verbose >= 1 && (i % sampRatio) == 0) {
                    cout << "--- Online Random Forest training --- Epoch: " << n + 1 << " --- ";
                    cout << (10 * i) / sampRatio << "%" << endl;
                }
            }

            results = test(dataset_ts);
            testError.push_back(compError(results, dataset_ts));
        }

        if (m_hp->verbose >= 1) {
            cout << endl << "--- Online Random Forest test error over epochs: ";
            dispErrors(testError);
        }

        return results;
    }

#endif

protected:
    int m_numClasses;
    double m_counter;
    Hyperparameters m_hp;

	Result treeResult;

    vector<OnlineTree*> m_trees;
	vector<double> m_oobe;
};

#endif /* ONLINERF_H_ */
