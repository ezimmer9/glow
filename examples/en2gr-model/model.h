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
const unsigned EMBEDDING_SIZE = 256; // 1024;
const unsigned HIDDEN_SIZE = EMBEDDING_SIZE * 3; //1024;


extern llvm::cl::opt<BackendKind> ExecutionBackend;
extern llvm::cl::opt<unsigned> batchSizeOpt;
extern llvm::cl::OptionCategory debugCat;
extern llvm::cl::OptionCategory quantizationCat;
extern llvm::cl::OptionCategory en2grCat;
extern llvm::cl::opt<std::string> dumpGraphDAGFileOpt;

struct metadata{
	unsigned int offset;
	std::string name;
	std::string type;
	std::vector<int> m_size;

	void get_size(std::map<int , std::shared_ptr<int>> size){
		int arr_size = size.begin()->first;
		for (int i = 0 ; i < arr_size ; i++){
			m_size.push_back(*((size.begin()->second).get() + i));
		}
		for (int j = 0 ; j < m_size.size() ; j++){
			std::cout << "vector size " << j << ":  = " << m_size[j] << "\n";
		}
	}

	void write_to_file(std::ofstream& output){
		offset = 3 * sizeof(size_t) + m_size.size()*sizeof(int) + name.size() + type.size() +sizeof(unsigned int);
		output.write(reinterpret_cast<char*>(&offset), sizeof(unsigned int));
		size_t name_len = name.size();
		output.write(reinterpret_cast<char*>(&name_len), sizeof(name_len));
		output.write(reinterpret_cast<const char*>(name.c_str()), name_len);
		size_t type_len = type.size();
		output.write(reinterpret_cast<char*>(&type_len), sizeof(type_len));
		output.write(reinterpret_cast<const char*>(type.c_str()), type_len);
		size_t m_size_len = m_size.size();
		output.write(reinterpret_cast<char*>(&m_size_len), sizeof(m_size_len));
		for (int i = 0 ; i < m_size_len ; i++){
			output.write(reinterpret_cast<char*>(&m_size[i]), sizeof(int));
		}
	}

	void read_from_file(std::ifstream& file){
		unsigned int offset;
		file.read(reinterpret_cast<char*>(&offset) , sizeof(unsigned int));
		std::cout << "offset: " << offset << "\n";
		size_t len;
		file.read(reinterpret_cast<char*>(&len) , sizeof(len));
		char *_name = new char [len];
		file.read(reinterpret_cast<char*>(_name) , len);
		name = std::string(_name);
		free(_name);
		std::cout << "name: " << name.c_str() << "\n";
		size_t len1;
		file.read(reinterpret_cast<char*>(&len1) , sizeof(len1));
		char * _type = new char [len1];
		file.read(reinterpret_cast<char*>(_type) , len1);
		type = std::string(_type);
		free(_type);
		std::cout << "type: " << type.c_str() << "\n";
		size_t len2;
		file.read(reinterpret_cast<char*>(&len2) , sizeof(len2));
		for (int i = 0 ; i < len2 ; i++){
			int temp;
			file.read(reinterpret_cast<char*>(&temp), sizeof(int));
			m_size.push_back(temp);
			std::printf("m_size[%d]: ", i);
			std::cout << m_size[i]<< ", ";
		}
		std::cout <<"\n";
	}
};

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
  Placeholder *seqLength_;
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
