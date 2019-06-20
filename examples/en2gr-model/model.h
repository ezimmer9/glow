/*
 * model.h
 *
 *  Created on: May 5, 2019
 *      Author: ezimmer9
 */
#ifndef EXAMPLES_EN2GR_MODEL_MODEL_H_
#define EXAMPLES_EN2GR_MODEL_MODEL_H_

#include "glow/ExecutionEngine/ExecutionEngine.h"
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
using namespace glow;

const unsigned ENCODER_LSTMS_NUM = 4;
const unsigned DECODER_LSTMS_NUM = 4;
const unsigned MAX_LENGTH = 3;
const unsigned EMBEDDING_SIZE =/* 256;*/  1024;
const unsigned HIDDEN_SIZE = /*EMBEDDING_SIZE * 3;*/ 1024;


extern llvm::cl::opt<BackendKind> ExecutionBackend;
extern llvm::cl::opt<unsigned> batchSizeOpt;
extern llvm::cl::OptionCategory debugCat;
extern llvm::cl::OptionCategory quantizationCat;
extern llvm::cl::OptionCategory en2grCat;
extern llvm::cl::opt<std::string> dumpGraphDAGFileOpt;


struct Vocabulary {
  std::vector<std::string> index2word_;
  std::unordered_map<std::string, int64_t> word2index_;

  void addWord(llvm::StringRef word) {
    word2index_[word] = index2word_.size();
    index2word_.push_back(word);
  }

  Vocabulary() = default;

  void loadVocabularyFromFile(llvm::StringRef filename) {
    std::ifstream file(filename);
    assert(file.is_open() == true && "file not open");
    std::string word;
    while (getline(file, word)) {
    	addWord(word);
    }
  }
};

struct Model {
  unsigned batchSize_;
  ExecutionEngine EE_{ExecutionBackend};
  Function *F_;
  Vocabulary en_, gr_, tok_;
  Placeholder *input_;
  Placeholder *output_;
  PlaceholderBindings bindings;
  LoweredInfoMap loweredMap_;

  void loadLanguages();
  void loadTokens();
  void loadEncoder();
  std::vector<Node *> loadAttention(std::vector<Node *> AttentionQuery);
  void loadDecoder();
  void translate(const std::vector<std::string> &batch);

  Model(unsigned batchSize) : batchSize_(batchSize) {
    F_ = EE_.getModule().createFunction("main");
  }

  void dumpGraphDAG(const char *filename) { F_->dumpDAG(filename); }

  void compile();

private:
  Placeholder *embedding_gr_, *embedding_en_ , *embedding_tok_;
  Node *encoderHiddenOutput_;
  Node *attetionOutput_;

  Placeholder *loadEmbedding(llvm::StringRef langPrefix, size_t langSize);
};



#endif /* EXAMPLES_EN2GR_MODEL_MODEL_H_ */
