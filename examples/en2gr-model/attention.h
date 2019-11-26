/*
 * attention.h
 *
 *  Created on: Nov 14, 2019
 *      Author: ezimmer9
 */

#ifndef EXAMPLES_EN2GR_MODEL_ATTENTION_H_
#define EXAMPLES_EN2GR_MODEL_ATTENTION_H_

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
class Attention{

private:
	uint m_batch_size , m_beam_size;
	void init();

public:
	Attention(uint batch , uint beam , Model *model){
		m_batch_size = batch;
		m_beam_size = beam;
		_model = model;
		init();
	}

	Model *_model;
	Placeholder *Wa , *Bwa , *Ua , *Bua , *Vt , *NormBias , *NormScalar , *SM;
	AttentionParams attention_out;
	AttentionParams loadAttention(Node *AttentionQuery);
	uint get_batch_size(){
		return m_batch_size;
	}
	uint get_beam_size(){
		return m_beam_size;
	}

};


#ifdef __cplusplus
} // extern "C"
#endif



#endif /* EXAMPLES_EN2GR_MODEL_ATTENTION_H_ */
