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
const size_t NUM_TRACE = 2;
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
	  std::cout << "Error reading file: " << filename.str() << '\n';
    exit(1);
  }
  file.close();
  if (false){
	  float *f_ptr = (float *)result.getUnsafePtr();
	  for (int i = 0 ; i < 100 ; i++){
		  std::cout << i << ": " <<
				  f_ptr[2000+i] << ", ";
	  }
  }
}

/// Loads tensor of floats from binary file.
void loadMatrixAndTransposeFromFile(llvm::StringRef filename, Tensor &result) {
	if (result.dims().size() != 2){
		std::cout << "[Error] the load from file and trnspose only with 2 dimension\n";
		exit(1);
	}
	Tensor src(ElemKind::FloatTy, {result.dims()[1], result.dims()[0]});
	Tensor dst(ElemKind::FloatTy, {result.dims()[0], result.dims()[1]});
	std::ifstream file(filename.str(), std::ios::binary);
	assert(file.is_open() == true);
	metadata meta;
	meta.read_from_file(file);
	if (!file.read(src.getUnsafePtr(), result.size() * sizeof(float))) {
		std::cout << "Error: only " << file.gcount() << " could be read\n";
		std::cout << "Error reading file: " << filename.str() << '\n';
		exit(1);
	}
	src.transpose(&dst , {1,0});
	std::memcpy(result.getUnsafePtr(), dst.getUnsafePtr(), result.size()*sizeof(float));
	file.close();
}

/// Loads tensor of floats from binary file.
void loadMatrixAndSplitFromFile(llvm::StringRef filename, std::vector<Constant *> result, uint numOfSlices) {
	std::ifstream file(filename.str(), std::ios::binary);
	assert(file.is_open() == true);
	metadata meta;
	meta.read_from_file(file);
	int file_size = 1;
	for (auto size : meta.m_size){
		file_size = file_size * size;
	}
	for (uint i = 0 ; i < numOfSlices ; i++){
		if (result[i]->dims().size() == 2){
			if (!file.read(result[i]->getPayload().getUnsafePtr(), (file_size/numOfSlices) * sizeof(float))) {
				std::cout << "Error: only " << file.gcount() << " could be read\n";
				std::cout << "Error reading file: " << filename.str() << '\n';
				exit(1);
			}
			if (false){
				std::cout << "******************src*************\n";
				for (uint i = 0 ; i < result[i]->dims()[0] ; i++){
					for (uint j= 0 ; j < result[i]->dims()[1] ; j++){
						std::cout <<i << " " << j << ":" <<
								result[i]->getHandle<float_t>().at({i , j}) << " ";
					}
					std::cout << "\n";
				}
				std::cout << "\n\n";
			}
		}
		else if (result[i]->dims().size() == 1){
			if (!file.read(result[i]->getPayload().getUnsafePtr(), (file_size/numOfSlices) * sizeof(float))) {
				std::cout << "Error: only " << file.gcount() << " could be read\n";
				std::cout << "Error reading file: " << filename.str() << '\n';
				exit(1);
			}
		}
	}
	file.close();
//  std::cout << *(float *)(result[0]->getPayload().getUnsafePtr());
//  std::cout << *(float *)(result[0]->getPayload().getUnsafePtr() + 4);
//  std::cout << " ****** End loadMatrixAndSplitFromFile ******\n" ;
}

void loadMatrixAndSplitAndTransposeFromFile(llvm::StringRef filename, std::vector<Constant *> result, uint numOfSlices) {
	std::ifstream file(filename.str(), std::ios::binary);
	assert(file.is_open() == true);
	metadata meta;
	meta.read_from_file(file);
	size_t file_size = 1;
	for (auto size : meta.m_size){
		file_size = file_size * size;
	}
	for (uint i = 0 ; i < numOfSlices ; i++){
		if (result[i]->dims().size() != 2){
			std::cout << "[Error] the load from file and trnspose only with 2 dimension\n";
			exit(1);
		}
		if (result[i]->dims().size() == 2){
			Tensor src(ElemKind::FloatTy, {result[i]->dims()[1],result[i]->dims()[0]});
			Tensor dst(ElemKind::FloatTy, {(result[i]->dims()[0]),result[i]->dims()[1]});
			if (!file.read(src.getUnsafePtr(), (file_size/numOfSlices) * sizeof(float))) {
				std::cout << "Error: only " << file.gcount() << " could be read\n";
				std::cout << "Error reading file: " << filename.str() << '\n';
				exit(1);
			}
			src.transpose(&dst , {1,0});
//			for (int i = 1; i < 5 ; i++){
//				for (int j = 1; j < 5 ; j++){
//					std::cout << src.getHandle<float_t>().at({i,j}) <<  " "
//							<< dst.getHandle<float_t>().at({j,i}) << "\n";
//				}
//			}
			std::memcpy(result[i]->getPayload().getUnsafePtr(),
					dst.getUnsafePtr(), (file_size/numOfSlices)*sizeof(float));
		}
	}
	file.close();
//  std::cout << *(float *)(result[0]->getPayload().getUnsafePtr());
//  std::cout << *(float *)(result[0]->getPayload().getUnsafePtr() + 4);
//  std::cout << " ****** End loadMatrixAndSplitFromFile ******\n" ;
}
