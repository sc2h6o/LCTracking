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

    double score() {
   //     double totalCount = m_trueCount + m_falseCount;

   //     // Split Entropy
   //     double p, splitEntropy = 0.0;
   //     if (m_trueCount) {
   //         p = m_trueCount / totalCount;
   //         splitEntropy -= p * log2(p);
   //     }
   //     if (m_falseCount) {
			//p = m_falseCount / totalCount;
   //         splitEntropy -= p * log2(p);
   //     }

   //     // Prior Entropy
   //     double priorEntropy = 0.0;
   //     for (int i = 0; i < m_numClasses; i++) {
   //         p = (m_trueStats[i] + m_falseStats[i]) / totalCount;
   //         if (p) {
   //             priorEntropy -= p * log2(p);
   //         }
   //     }

   //     // Posterior Entropy
   //     double trueScore = 0.0, falseScore = 0.0;
   //     if (m_trueCount) {
   //         for (int i = 0; i < m_numClasses; i++) {
   //             p = m_trueStats[i] / m_trueCount;
   //             if (p) {
   //                 trueScore -= p * log2(p);
   //             }
   //         }
   //     }
   //     if (m_falseCount) {
   //         for (int i = 0; i < m_numClasses; i++) {
   //             p = m_falseStats[i] / m_falseCount;
   //             if (p) {
   //                 falseScore -= p * log2(p);
   //             }
   //         }
   //     }
   //     double posteriorEntropy = (m_trueCount * trueScore + m_falseCount * falseScore) / totalCount;

   //     // Information Gain
   //     return (priorEntropy - posteriorEntropy);

		double totalCount = m_trueCount + m_falseCount;
		double p= 0.0;
		double ksi = 0;
		// Prior Entropy
		double priorEntropy = 0.0;
		double priorConfid = 1.0;
		for (int i = 0; i < m_numClasses; i++) {
			p = (m_trueStats[i] + m_falseStats[i]) / totalCount;
			if (p && i == m_negClass) {
				priorEntropy -= p * totalCount * log2(p);
				//priorEntropy += p * totalCount * log2(1 + ksi - p);
			}
			else if(p) {
				priorEntropy -= p * totalCount * log2(p);
				priorConfid *= pow(1 - p, p * totalCount);
			}
			
		}

		// Posterior Entropy
		double trueScore = 0.0, falseScore = 0.0;
		double trueConfid = 1.0, falseConfid = 1.0;
		if (m_trueCount) {
			for (int i = 0; i < m_numClasses; i++) {
				p = m_trueStats[i] / m_trueCount;
				if (p && i == m_negClass) {
					trueScore -= p * m_trueCount * log2(p);
					//trueScore += p * m_trueCount * log2(1 + ksi - p);
				}
				else if (p) {
					trueScore -= p * m_trueCount * log2(p);
					trueConfid *= pow(1 - p, p * m_trueCount);
					//trueScore += p * m_trueCount * log2(1 + ksi - p);
				}
			}
		}
		if (m_falseCount) {
			for (int i = 0; i < m_numClasses; i++) {
				p = m_falseStats[i] / m_falseCount;
				if (p && i == m_negClass) {
					falseScore -= p * m_falseCount * log2(p);
					//falseScqore += p * m_falseCount * log2(1 + ksi - p);
				}
				else if (p) {
					falseScore -= p * m_falseCount * log2(p);
					falseConfid *= pow(1 - p, p * m_falseCount);
					//falseScore += p * m_falseCount * log2(1 + ksi - p);
				}
			}
		}

		// Information Gain
		return (priorEntropy - trueScore - falseScore - ksi * trueConfid * falseConfid);
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
