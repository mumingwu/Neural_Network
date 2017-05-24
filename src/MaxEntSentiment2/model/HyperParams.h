#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3L.h"
#include "Example.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams{

	dtype _regParameter; // for optimization
	dtype _adaAlpha;  // for optimization
	dtype _adaEps; // for optimization
	dtype _dropProb;



	//auto generated
	int wordDim;
	int inputSize;
	int labelSize;
	int _rnnHiddenSize;
	int _hiddenSize;
	int _wordcontext;


public:
	HyperParams(){
		bAssigned = false;
	}

public:
	void setRequared(Options& opt){
		_regParameter = opt.regParameter;
		_adaAlpha = opt.adaAlpha;
		_adaEps = opt.adaEps;
		_dropProb = opt.dropProb;
		_rnnHiddenSize = opt.rnnHiddenSize;
		_hiddenSize = opt.hiddenSize;
		_wordcontext = opt.wordcontext;
		wordDim = opt.wordEmbSize;

		bAssigned = true;
	}

	void clear(){
		bAssigned = false;
	}

	bool bValid(){
		return bAssigned;
	}


public:

	void print(){

	}

private:
	bool bAssigned;
};


#endif /* SRC_HyperParams_H_ */