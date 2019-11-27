

#ifndef META_COMMON_H
#define META_COMMON_H


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
const size_t NUM_TRACE = 2;

#ifdef __cplusplus
extern "C"
{
#endif


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

struct AttentionParams{
  	NodeValue attention_out;
  	NodeValue scores;
};

struct BeamDecoderOutput{
	Node *candidates;
	Node *scores;
};

struct metadata{

	unsigned int offset;
	std::string name;
	std::string type;
	std::vector<int> m_size;

	void get_size(std::map<int , std::shared_ptr<int>> size);
	void write_to_file(std::ofstream& output);
	void read_from_file(std::ifstream& file);
};


class En2grCommon {

public:
	En2grCommon(){};
	void loadMatrixFromFile(llvm::StringRef filename, Tensor &result);
	void loadMatrixAndTransposeFromFile(llvm::StringRef filename, Tensor &result);
	void loadMatrixAndSplitFromFile(llvm::StringRef filename, std::vector<Constant *> result, uint numOfSlices);
	void loadMatrixAndSplitAndTransposeFromFile(llvm::StringRef filename, std::vector<Constant *> result, uint numOfSlices);
	void print_weights(std::vector<Constant *> weights);
	void init_hidden(std::vector<std::vector<Node *>> &hidden , uint length , Node *node);

};

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* COMMON_H_ */
