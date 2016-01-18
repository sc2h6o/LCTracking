#ifndef RANDOMTEST_H_
#define RANDOMTEST_H_

#include "data.h"
#include "utilities.h"

class RandomTest {
public:
    RandomTest() {

    }

    RandomTest(int numClasses) :
        m_numClasses(numClasses), m_negClass(numClasses-2), m_trueCount(0.0), m_falseCount(0.0) {
        for (int i = 0; i < numClasses; i++) {
            m_trueStats.push_back(0.0);
            m_falseStats.push_back(0.0);
        }
        m_threshold = randomFromRange(-1, 1);
    }

    RandomTest(int numClasses, double featMin, double featMax) :
		m_numClasses(numClasses),  m_negClass(numClasses - 2), m_trueCount(0.0), m_falseCount(0.0) {
        for (int i = 0; i < numClasses; i++) {
            m_trueStats.push_back(0.0);
            m_falseStats.push_back(0.0);
        }
        m_threshold = randomFromRange(featMin, featMax);
    }

    void updateStats(const Sample &sample, const bool decision) {
        if (decision) {
            m_trueCount += sample.w;
            m_trueStats[sample.y] += sample.w;
        } else {
            m_falseCount += sample.w;
            m_falseStats[sample.y] += sample.w;
        }
    }

	double score(const vector<double> &c) {

		double totalCount = m_trueCount + m_falseCount;
		double p = 0.0;
		vector<double> pBag;
		pBag = c;
		// Prior Score
		for (int i = 0; i < m_numClasses; i++) {
			p = (m_trueStats[i] + m_falseStats[i]) / totalCount;
			if (p && i == m_negClass) {
				pBag[i] *= pow(p, (int)(p * totalCount));
			}
			else if(p && i < m_negClass) {
				pBag[i] *= pow(1 - p, (int)(p * totalCount));
			}
		}
		double priorScore = 1;
		for (int i = 0; i < pBag.size(); i++){
			if (i == m_negClass)
				priorScore *= pBag[i];
			else
				priorScore *= 1 - pBag[i];
		}

		// Posterior Entropy
		pBag = c;
		if (m_trueCount) {
			for (int i = 0; i < m_numClasses; i++) {
				p = m_trueStats[i] / m_trueCount;
				if (p && i == m_negClass) {
					pBag[i] *= pow(p, (int)(p * m_trueCount));
				}
				else if (p) {
					pBag[i] *= pow(1 - p, (int)(p * m_trueCount));
				}
			}
		}
		if (m_falseCount) {
			for (int i = 0; i < m_numClasses; i++) {
				p = m_falseStats[i] / m_falseCount;
				if (p && i == m_negClass) {
					pBag[i] *= pow(p, (int)(p * m_falseCount));
				}
				else if (p) {
					pBag[i] *= pow(1 - p, (int)(p * m_falseCount));
				}
			}
		}
		double posterScore = 1;
		for (int i = 0; i < pBag.size(); i++){
			if (i == m_negClass)
				posterScore *= pBag[i];
			else
				posterScore *= 1 - pBag[i];
		}

		// Information Gain
		return (posterScore - priorScore);
    }

    /*pair<vector<double> , vector<double> > getStats() {
        return pair<vector<double> , vector<double> > (m_trueStats, m_falseStats);
    }*/

	void getStats(vector<double>& trueStats, vector<double>& falseStats) {
		trueStats.resize(m_numClasses);
		falseStats.resize(m_numClasses);
		trueStats = m_trueStats;
		falseStats = m_falseStats;
	}

protected:
	int m_negClass;
    int m_numClasses;
    double m_threshold;
    double m_trueCount;
    double m_falseCount;
    vector<double> m_trueStats;
    vector<double> m_falseStats;
};

class HyperplaneFeature: public RandomTest {
public:
    HyperplaneFeature() {

    }

	void create(int numClasses, const vector<Sample>& samples) {
		m_numClasses = numClasses;
		m_trueCount = m_falseCount = 0.0;
		m_trueStats.resize(numClasses, 0);
		m_falseStats.resize(numClasses, 0);

		// calculate the random hyperplane with two points
		int s1, s2;
		s1 = (int)floor(samples.size() * randDouble());
		if (s1 == samples.size()) s1--;
		s2 = (int)floor((samples.size()-1) * randDouble());
		if (s2 == samples.size()) s2--;
		if (s2 == s1) s2++; // make sure s2 != s1
	
		m_proj.resize(samples[s1].x.size());
		minusVec(samples[s1].x, samples[s2].x, m_proj);
		normalize(m_proj);
		float sectionRatio = 0.2 + 0.6*randDouble();
		m_threshold = sectionRatio * dot(samples[s1].x, m_proj) + (1 - sectionRatio) * dot(samples[s2].x, m_proj);

	}

    void update(Sample &sample) {
        updateStats(sample, eval(sample));
    }

    bool eval(Sample &sample) {
		double proj = dot(sample.x, m_proj);
        return (proj > m_threshold) ? true : false;
    }

private:
    vector<double> m_proj;	// the normal vector of the plane
};

#endif /* RANDOMTEST_H_ */
