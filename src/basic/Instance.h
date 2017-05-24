#ifndef _INSTANCE_H_
#define _INSTANCE_H_

#include <iostream>
using namespace std;

class Instance
{
public:
	void clear()
	{
		words.clear();
		label.clear();
	}

	void evaluate(const string& predict_label, Metric& eval) const
	{
		if (predict_label == label)
			eval.correct_label_count++;
		eval.overall_label_count++;
	}

	void copyValuesFrom(const Instance& anInstance)
	{
		allocate(anInstance.size());
		label = anInstance.label;
		words = anInstance.words;
	}


	int size() const {
		return words.size();
	}

	void allocate(int length)
	{
		clear();
		words.resize(length);
	}

	
public:
	vector<string> words;
	string label;
};

#endif /*_INSTANCE_H_*/
