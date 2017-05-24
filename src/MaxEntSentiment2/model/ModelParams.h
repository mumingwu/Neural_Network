#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams{

public:
	Alphabet wordAlpha; // should be initialized outside
	
	SparseParams sparseparams;

	UniParams olayer_linear;
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
		sparseparams.initial(&wordAlpha, opts._hiddenSize);
		olayer_linear.initial(opts.labelSize, opts._hiddenSize, true, mem);
		return true;
	}


	void exportModelParams(ModelUpdate& ada){
		
		sparseparams.exportAdaParams(ada);
		olayer_linear.exportAdaParams(ada);
		
	}


	void exportCheckGradParams(CheckGrad& checkgrad){
		checkgrad.add(&(sparseparams.W), "sparseparams.W");
		checkgrad.add(&(olayer_linear.W), "olayer_linear.W");
		checkgrad.add(&(olayer_linear.b), "olayer_linear.b");

	}

	// will add it later
	void saveModel(){

	}

	void loadModel(const string& inFile){

	}

};

#endif /* SRC_ModelParams_H_ */