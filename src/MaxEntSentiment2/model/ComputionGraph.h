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
	UniNode _uninode;

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
		_uninode.clearValue();


	}

public:
	inline void initial(ModelParams& model, HyperParams& opts, AlignedMemoryPool* mem = NULL){	
		_sparsenode.init(opts._hiddenSize, opts._dropProb, mem);
		_sparsenode.setParam(&model.sparseparams);
		_uninode.init(opts.labelSize, opts._dropProb, mem);
		_uninode.setParam(&model.olayer_linear);

	}
	

public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const Feature& feature, bool bTrain = false){
		//first step: clear value
		clearValue(bTrain); // compute is a must step for train, predict and cost computation

		// second step: build graph
		//forward
		_sparsenode.forward(this, feature.words);
		_uninode.forward(this,&_sparsenode);

	}

};

#endif /* SRC_ComputionGraph_H_ */