#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	Alphabet wordAlpha; // should be initialized outside
	
	SparseParams sparseparams;
public:
	Alphabet labelAlpha; // should be initialized outside
	SoftMaxLoss loss;
	

	const static int levels = 3;




public:
	bool initial(HyperParams& opts, AlignedMemoryPool* mem = NULL){

		// some model parameters should be initialized outside
		if (wordAlpha.size() <= 0 || labelAlpha.size() <= 0){
			return false;
		}
		opts.labelSize = labelAlpha.size();
		sparseparams.initial(&wordAlpha, opts.labelSize);
		return true;
	}


	void exportModelParams(ModelUpdate& ada){
		
		sparseparams.exportAdaParams(ada);
		
	}


	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&(sparseparams.W), "sparseparams.W");

	}

	// will add it later
	void saveModel(){

	}

	void loadModel(const string& inFile){

	}

};

#endif /* SRC_ModelParams_H_ */