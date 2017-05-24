#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct ComputionGraph : Graph{
public:
	const static int max_sentence_length = 2048;

public:
	// node instances
	SparseNode _sparsenode;

public:
	ComputionGraph() : Graph(){
	}

	~ComputionGraph(){
		clear();
	}

public:
	//allocate enough nodes 
	
	inline void clear(){
		Graph::clear();
		_sparsenode.clearValue();


	}

public:
	inline void initial(ModelParams& model, HyperParams& opts, AlignedMemoryPool* mem = NULL){	
		_sparsenode.init(opts.labelSize, opts._dropProb, mem);
		_sparsenode.setParam(&model.sparseparams);
	}
	

public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const Feature& feature, bool bTrain = false){
		//first step: clear value
		clearValue(bTrain); // compute is a must step for train, predict and cost computation

		// second step: build graph
		//forward
		_sparsenode.forward(this, feature.words);

	}

};

#endif /* SRC_ComputionGraph_H_ */