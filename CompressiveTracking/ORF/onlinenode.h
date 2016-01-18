#ifndef ONLINENODE_H_
#define ONLINENODE_H_

#include <vector>

#include "data.h"
#include "hyperparameters.h"
#include "randomtest.h"
#include "utilities.h"

using namespace std;

class OnlineNode {
public:

    static bool echo;

    OnlineNode() {
        m_isLeaf = true;
    }

    OnlineNode(const Hyperparameters &hp, int numClasses, const int &depth) :
        m_numClasses(numClasses), m_depth(depth), m_height(0), m_isLeaf(true),
        m_needUpdate(true), m_counter(0.0), m_label(-1), m_parentCounter(0.0), m_hp(&hp), m_onlineTests()
    {
        for (int i = 0; i < numClasses; i++) {
            m_labelStats.push_back(0.0);
        }
        updateProb.resize(m_numClasses - 1, 1.0);
    }

    OnlineNode(const Hyperparameters &hp, int numClasses, const int &depth,
        const vector<double> &parentStats, const vector<Sample> &parentSamples) :
        m_numClasses(numClasses), m_depth(depth), m_height(0), m_isLeaf(true),
        m_needUpdate(true), m_counter(0.0), m_label(-1), m_parentCounter(0.0), m_hp(&hp), m_onlineTests()
    {
        m_labelStats = parentStats;
        m_label = argmax(m_labelStats);
        m_counter = sum(m_labelStats);
        m_parentCounter = sum(m_labelStats);
        updateProb.resize(m_numClasses - 1, 1.0);
        if (m_depth < m_hp->maxDepth) {
            m_samples = parentSamples;
            for (int i = 0; i < updateProb.size(); i++){
                double p = m_labelStats[i] / m_counter;
                if (p == 0)
                    updateProb[i] = 1;
                else if (i == m_numClasses - 2)
                    updateProb[i] = pow(p, m_counter);
                else
                    updateProb[i] = pow(1 - p, m_counter);
            }
        }

    }

    ~OnlineNode() {
        if (!m_isLeaf) {
            delete m_leftChildNode;
            delete m_rightChildNode;
        }
    }


    void insert(Sample &sample){
        m_needUpdate = true;
        m_counter += sample.w;
        m_labelStats[sample.y] += sample.w;
        

        if (m_isLeaf) {
            if (m_depth < m_hp->maxDepth)
                m_samples.push_back(sample);

            // Update the label
            m_label = argmax(m_labelStats);

        }

        else {
            if (m_bestTest.eval(sample)) {
                m_rightChildNode->insert(sample);
            }
            else {
                m_leftChildNode->insert(sample);
            }
        }
    }

    void updatePrepare(vector<double> &updateP) {
        if (!m_needUpdate) return; // height not changed

        if (m_isLeaf) {
            for (int i = 0; i < updateProb.size(); i++){
                double p = m_labelStats[i] / m_counter;
                if (p == 0)
                    updateProb[i] = 1;
                else if (i == m_numClasses - 2)
                    updateProb[i] = pow(p, m_counter);
                else
                    updateProb[i] = pow(1 - p, m_counter);
            }
            timesVec(updateProb, updateP, updateP);
        }
        else {
            m_rightChildNode->updatePrepare(updateP);
            m_leftChildNode->updatePrepare(updateP);
        }
    }

    static double t;
    void update(int& height, vector<double> &updateP) {
        if (!m_needUpdate) return; // height not changed

        if (m_isLeaf) {
            // Decide for split
            if (shouldISplit()) {
                m_isLeaf = false;

                // Create online tests
                m_onlineTests.resize(m_hp->numRandomTests);
                for (int i = 0; i < m_hp->numRandomTests; i++) {
                    m_onlineTests[i].create(m_numClasses, m_samples);
                    for (int j = 0; j < m_samples.size(); j++){
                        m_onlineTests[i].update(m_samples[j]);
                    }
                }

                // Find the best online test
                int maxIndex = 0;
                double maxScore = -1e10, score;
                vector<double> c(m_numClasses - 1);
                for (int i = 0; i < c.size(); i++){
                    if (updateProb[i] == 0)
                        c[i] = 0;
                    else
                        c[i] = updateP[i] / updateProb[i];
                }
                for (int i = 0; i < m_hp->numRandomTests; i++) {
                    score = m_onlineTests[i].score(c);
                    if (score > maxScore) {
                        maxScore = score;
                        maxIndex = i;
                    }
                }
                m_bestTest = m_onlineTests[maxIndex];
                m_onlineTests.clear();

                // Split the samples
                vector<Sample> sampleLeft, sampleRight;
                vector<bool> isRight(m_samples.size(), false);
                int cntRight = 0, cntLeft = 0;
                for (int i = 0; i < m_samples.size(); i++){
                    if (m_bestTest.eval(m_samples[i])){
                        cntRight++;
                        isRight[i] = true;
                        //sampleRight.push_back(m_samples[i]);
                    }
                    else{
                        cntLeft++;
                        //sampleLeft.push_back(m_samples[i]);
                    }
                }
                sampleRight.resize(cntRight);
                sampleLeft.resize(cntLeft);
                cntRight = cntLeft = 0;
                for (int i = 0; i < m_samples.size(); i++){
                    if (isRight[i]){
                        sampleRight[cntRight] = m_samples[i];
                        cntRight++;
                    }
                    else{
                        sampleLeft[cntLeft] = m_samples[i];
                        cntLeft++;
                    }

                }

                // Split
                vector<double> trueStats, falseStats;
                m_bestTest.getStats(trueStats, falseStats);
                m_rightChildNode = new OnlineNode(*m_hp, m_numClasses, m_depth + 1,
                    trueStats, sampleRight);
                m_leftChildNode = new OnlineNode(*m_hp, m_numClasses, m_depth + 1,
                    falseStats, sampleLeft);
                m_samples.clear();
                timesVec(m_rightChildNode->updateProb, c, c);
                timesVec(m_leftChildNode->updateProb, c, updateP);
                m_rightChildNode->update(m_height, updateP);
                m_leftChildNode->update(m_height, updateP);
            }
        }
        else {
            m_rightChildNode->update(m_height, updateP);
            m_leftChildNode->update(m_height, updateP);
        }

        height = max(height, m_height + 1);
        m_needUpdate = false;
        m_samples.clear();
    }

    void eval(Sample &sample, Result& result) {
        if (!result.confidence.size()) 
            result.confidence.resize(m_numClasses);
        if (m_isLeaf) {
            if (m_counter) {
                result.confidence = m_labelStats;
                scale(result.confidence, 1.0 / (m_counter));
                result.prediction = m_label;
            } 
            else {
                for (int i = 0; i < m_numClasses; i++) {
                    result.confidence[i] = 1.0 / m_numClasses;
                }
                result.prediction = 0;
            }
        } 
        else {
            if (m_bestTest.eval(sample)) {
                m_rightChildNode->eval(sample, result);
            } 
            else {
                m_leftChildNode->eval(sample, result);
            }
        }
    }

    bool shouldISplit() {
        bool isPure = false;
        if (m_hp->useUncertain){
            int pureClass = 0;
            for (int i = 0; i < m_numClasses - 1; i++) {
                if (m_labelStats[i] == m_counter - m_labelStats[m_numClasses - 1]) {
                    isPure = true;
                    pureClass = i;
                    break;
                }
            }
            if (isPure &&
                m_counter != m_labelStats[m_numClasses - 1] &&
                (m_labelStats[pureClass] <= m_hp->splitRatioUncertain * m_labelStats[m_numClasses-1]))
            {
                // cout << "split for certain " << m_labelStats[pureClass] << " " << m_labelStats[m_numClasses-1] << endl;
                isPure = false;
            }
        }
        else {
            for (int i = 0; i < m_numClasses; i++) {
                if (m_labelStats[i] == m_counter) {
                    isPure = true;
                    break;
                }
            }
        }
        

        if (isPure) {
            return false;
        }

        if (m_depth >= m_hp->maxDepth) { // do not split if max depth is reached
            return false;
        }

        if (m_counter < m_hp->counterThreshold) { // do not split if not enough samples seen
            return false;
        }
        // cout << m_depth << endl;
        return true;
    }

};

//private:
    int m_numClasses;
    int m_depth;
    int m_height;   // the depth of the subtree, start from 0
    bool m_isLeaf;
    bool m_needUpdate;
    double m_counter;
    int m_label;
    double m_parentCounter;
    const Hyperparameters *m_hp;


    vector<double> updateProb;
    vector<double> m_labelStats;
    vector<Sample> m_samples;


    OnlineNode* m_leftChildNode;
    OnlineNode* m_rightChildNode;

    vector<HyperplaneFeature> m_onlineTests;
    HyperplaneFeature m_bestTest;

#endif /* ONLINENODE_H_ */
