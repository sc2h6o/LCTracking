#ifndef HYPERPARAMETERS_H_
#define HYPERPARAMETERS_H_

#include <string>
using namespace std;

class Hyperparameters
{
 public:
	Hyperparameters(){};
    // Hyperparameters(const string& confFile);

    // Online node
    int numRandomTests;
    int counterThreshold;
    int maxDepth;
	bool useUncertain;
	float splitRatioUncertain;

    // Online tree

    // Online forest
	int numTrees;
	int numGrowTrees;	// the number of new trees at one time for evovling
	int MatureDepth;	// the depth criteria to judge whether a gowing tree could substitute
    int useSoftVoting;
    int numEpochs;
	float lamda;

    // Data
    //string trainData;
    //string testData;

    // Output
    int verbose;
};

#endif /* HYPERPARAMETERS_H_ */
