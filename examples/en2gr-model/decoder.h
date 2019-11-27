/*
 * decoder.h
 *
 *  Created on: Nov 26, 2019
 *      Author: ezimmer9
 */

#ifndef EXAMPLES_EN2GR_MODEL_DECODER_H_
#define EXAMPLES_EN2GR_MODEL_DECODER_H_

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

class Model;
class LSTM;
class Attention;

#ifdef __cplusplus
extern "C"
{
#endif

class Decoder {
private:
	uint m_batch_size , m_beam_size;

public:
	Model *m_model;
	Decoder(uint batch , uint beam , Model *model){
		m_model = model; m_batch_size = batch; m_beam_size = beam;
	}
	void init();
	uint get_batch_size(){
		return m_batch_size;
	}
	uint get_beam_size(){
		return m_beam_size;
	}
	std::shared_ptr<LSTM> decoderLayer0 , decoderLayer1 , decoderLayer2 , decoderLayer3;
	std::shared_ptr<Attention> m_attention;
	AttentionParams attention;
	Placeholder *classifier_w ,*classifier_b;
	Node* loadGreadyDecoder();
	BeamDecoderOutput loadBeamDecoder();
	void loadDecoderWieghts();
	void loadDecoderBiases();
};


#ifdef __cplusplus
} // extern "C"
#endif

#endif /* EXAMPLES_EN2GR_MODEL_DECODER_H_ */
