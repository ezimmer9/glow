/*
 * model.h
 *
 *  Created on: May 5, 2019
 *      Author: ezimmer9
 */
#ifndef EXAMPLES_EN2GR_MODEL_MODEL_H_
#define EXAMPLES_EN2GR_MODEL_MODEL_H_

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

#include "en2gr_common.h"
#include "lstm.h"
#include "encoder.h"
#include "decoder.h"
#include "attention.h"


using namespace glow;

#ifdef __cplusplus
extern "C"
{
#endif

const unsigned ENCODER_LSTMS_NUM = 4;
const unsigned DECODER_LSTMS_NUM = 4;
const unsigned MAX_LENGTH = 4;
const unsigned EMBEDDING_SIZE = 1024;
const unsigned HIDDEN_SIZE = 1024;
const uint LSTM_LEVELS = 4;
const std::string files_gates[LSTM_LEVELS] = {"input" , "forget" , "cell" , "output" };
const int64_t PAD = 0;
const int64_t UNK = 1;
const int64_t BOS = 2;
const int64_t EOS = 3;


extern llvm::cl::opt<std::string> ExecutionBackend;
extern llvm::cl::opt<unsigned> batchSizeOpt;
extern llvm::cl::OptionCategory debugCat;
extern llvm::cl::OptionCategory quantizationCat;
extern llvm::cl::OptionCategory en2grCat;
extern llvm::cl::opt<std::string> dumpGraphDAGFileOpt;
extern llvm::cl::opt<bool> debugMode;
extern llvm::cl::opt<int> beamSizeOpt;
extern llvm::cl::opt<uint> maxLengthOpt;
extern llvm::cl::opt<std::string> inputOpt;
extern llvm::cl::opt<std::string> file;

class Encoder;
class Decoder;

class Model {

public:
	unsigned batchSize_;
	uint beam_size;
	uint max_length;
	ExecutionContext context;
	ExecutionEngine EE_{ExecutionBackend};
	Function *F_;
	Vocabulary tok_;
	Placeholder *input_;
	Placeholder *output_;
	Placeholder *scores_;
	Placeholder *attention_mask_;
	LoweredInfoMap loweredMap_;
	std::unique_ptr<TraceContext> traceContext;
	AttentionParams attention;
	bool alloc_mask = false;
	Node *encoderHiddenOutput_;

	Model(unsigned batchSize, int beamSize, int maxLength) :
		batchSize_(batchSize), beam_size(beamSize), max_length(maxLength) {
		F_ = EE_.getModule().createFunction("main");
		common_ = std::make_shared<En2grCommon>();
		m_encoder = std::make_shared<Encoder>(batchSize , beamSize , this);
		m_decoder = std::make_shared<Decoder>(batchSize , beamSize , this);
	}
	Placeholder *get_embedding_tok(){
		return embedding_tok_;
	}

	void loadTokens();
	void loadEncoder();
	void loadGreadyDecoder();
	void loadBeamDecoder();
	void translate(const std::vector<std::string> &batch);
	void dumpGraphDAG(const char *filename) { F_->dumpDAG(filename); }
	void compile();

	std::shared_ptr<En2grCommon> common_;
	std::shared_ptr<Encoder> m_encoder;
	std::shared_ptr<Decoder> m_decoder;

private:
	Placeholder *embedding_tok_;
	Placeholder *loadEmbedding(llvm::StringRef langPrefix, size_t langSize);
};

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* EXAMPLES_EN2GR_MODEL_MODEL_H_ */
