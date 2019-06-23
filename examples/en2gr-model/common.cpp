/*
 * common.cpp
 *
 *  Created on: Jun 20, 2019
 *      Author: ezimmer9
 */

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
		for (unsigned int j = 0 ; j < m_size.size() ; j++){
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
		for (unsigned int i = 0 ; i < m_size_len ; i++){
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
		for (unsigned int i = 0 ; i < len2 ; i++){
			int temp;
			file.read(reinterpret_cast<char*>(&temp), sizeof(int));
			m_size.push_back(temp);
			std::printf("m_size[%d]: ", i);
			std::cout << m_size[i]<< ", ";
		}
		std::cout <<"\n";
	}
};

/// Loads tensor of floats from binary file.
void loadMatrixFromFile(llvm::StringRef filename, Tensor &result) {
  std::ifstream file(filename.str(), std::ios::binary);
  assert(file.is_open() == true);
  metadata meta;
  meta.read_from_file(file);
  if (!file.read(result.getUnsafePtr(), result.size() * sizeof(float))) {
	  std::cout << "Error: only " << file.gcount() << " could be read\n";
	  std::cout << "Error reading file: " << filename.str() << '\n'
			  << "Need to be downloaded by calling:\n"
              << "python ../glow/utils/download_test_db.py -d fr2en\n";
    exit(1);
  }
}

/// Loads tensor of floats from binary file.
void loadMatrixAndSplitFromFile(llvm::StringRef filename, std::vector<Constant *> result, uint numOfSlices) {
  std::ifstream file(filename.str(), std::ios::binary);
  assert(file.is_open() == true);
  metadata meta;
  meta.read_from_file(file);
  for (uint i = 0 ; i < numOfSlices ; ++i){
	  if (!file.read(result[i]->getPayload().getUnsafePtr(), meta.m_size[0]/numOfSlices * sizeof(float))) {
		  std::cout << "Error: only " << file.gcount() << " could be read\n";
		  std::cout << "Error reading file: " << filename.str() << '\n'
				  << "Need to be downloaded by calling:\n"
				  << "python ../glow/utils/download_test_db.py -d fr2en\n";
		exit(1);
	  }
  }
  //std::cout << *(float *)(result[0]->getPayload().getUnsafePtr());
  //std::cout << " ****** End loadMatrixAndSplitFromFile ******\n" ;
}
