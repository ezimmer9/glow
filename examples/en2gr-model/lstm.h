/*
 * lstm.h
 *
 *  Created on: Nov 12, 2019
 *      Author: ezimmer9
 */

#ifndef EXAMPLES_EN2GR_MODEL_LSTM_H_
#define EXAMPLES_EN2GR_MODEL_LSTM_H_

#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Optimizer/GraphOptimizer/GraphOptimizer.h"
#include "glow/Graph/Graph.h"
#include "glow/Quantization/Quantization.h"
#include "glow/Quantization/Serialization.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Timer.h"


#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "model.h"

using namespace glow;

#ifdef __cplusplus
extern "C"
{
#endif

struct Model;
class Gate{

private:
	std::string _name;
	unsigned _inputSize;
	unsigned _hiddenSize;

public:
	Placeholder *Wi , *Wh , *Bi , *Bh ;
	Model *_model;

	Gate(){
		_name = "gate";
		_inputSize = 0;
		_hiddenSize = 0;
	}

	void setPlaceholders();
	void allocate();
	void setParams(std::string name, unsigned inputSize, unsigned hiddenSize);

};

class LSTM{

private:
	std::string _name;
	unsigned inputSize;
	unsigned hiddenSize;

public:
	Gate forget;
	Gate input;
	Gate output;
	Gate cell;
	Model *_model;

	LSTM(Model *model){
		_name = "lstm.instance";
		_model = model;
		forget._model = model;
		input._model = model;
		output._model = model;
		cell._model = model;
	}

	void setParams(std::string name_, unsigned inputSize_, unsigned hiddenSize_);
	void updateGates();
	void createInferPytorchLSTM(PlaceholderBindings &bindings, llvm::StringRef namePrefix,
			const llvm::ArrayRef<NodeValue> inputs, unsigned batchSize,
			unsigned hiddenSize, unsigned outputSize,
			std::vector<NodeValue> &outputs);

	void createDecoderInferPytorchLSTM(PlaceholderBindings &bindings, llvm::StringRef namePrefix,
			const llvm::ArrayRef<NodeValue> inputs, unsigned batchSize,
			unsigned hiddenSize, unsigned outputSize,
			std::vector<std::vector<NodeValue>> &outputs,
			std::vector<Node *> hidden , uint word_inx);
};

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* EXAMPLES_EN2GR_MODEL_LSTM_H_ */
