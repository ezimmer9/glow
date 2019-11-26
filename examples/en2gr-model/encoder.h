/*
 * encoder.h
 *
 *  Created on: Nov 24, 2019
 *      Author: ezimmer9
 */

#ifndef EXAMPLES_EN2GR_MODEL_ENCODER_H_
#define EXAMPLES_EN2GR_MODEL_ENCODER_H_

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

struct Model;
class LSTM;

#ifdef __cplusplus
extern "C"
{
#endif

class Encoder {

private:
	uint m_batch_size , m_beam_size;
	void init();

public:
	Model *m_model;
	Encoder(uint batch , uint beam , Model *model){
		m_model = model; m_batch_size = batch; m_beam_size = beam;
		init();
	}
	std::shared_ptr<LSTM> encoderLayer0 , encoderLayerOpp ,encoderLayer1 ,encoderLayer2 ,encoderLayer3;
	uint get_batch_size(){
		return m_batch_size;
	}
	uint get_beam_size(){
		return m_beam_size;
	}
	Node* loadEncoder(Placeholder *embedding_tok_ , Placeholder *input_);
	void loadEncoderWieghts();
	void loadEncoderBiases();
	void loadEncoderReverse();
};

#ifdef __cplusplus
} // extern "C"
#endif


#endif /* EXAMPLES_EN2GR_MODEL_ENCODER_H_ */
