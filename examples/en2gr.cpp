

#include "en2gr-model/model.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>


int main(int argc, char **argv){
	std::printf("*********** en2gr workload ************\n\n");
	std::array<const llvm::cl::OptionCategory *, 3> showCategories = {
	  {&debugCat, &quantizationCat, &en2grCat}};
	llvm::cl::HideUnrelatedOptions(showCategories);
	llvm::cl::ParseCommandLineOptions(argc, argv, "Translate sentences from English to German");

	Model gnmt(batchSizeOpt, beamSizeOpt, maxLengthOpt);
	gnmt.loadTokens();
	gnmt.loadEncoder();
	if (beamSizeOpt == 1){
		gnmt.loadGreadyDecoder();
	}
	else {
		gnmt.loadBeamDecoder();
	}
	gnmt.compile();

	if (!dumpGraphDAGFileOpt.empty()) {
	    gnmt.dumpGraphDAG(dumpGraphDAGFileOpt.c_str());
	}

	if (debugMode){
		std::vector<std::string> debug = {"They are"};
		gnmt.translate(debug);
		return 0;

	}

	if (inputOpt != " "){
		std::vector<std::string> vstrings = {inputOpt};
		gnmt.translate(vstrings);
		return 0;
	}

	if (file != " "){
		std::vector<std::string> sentences;
		std::string temp_sentence;
		std::ifstream infile(file.c_str());
		assert(infile.is_open() == true);
		while (std::getline(infile, temp_sentence)){
			std::cout << temp_sentence.c_str() << "\n";
			std::vector<std::string> vstrings = {temp_sentence};
			gnmt.translate(vstrings);
		}
		return 0;
	}

	std::vector<std::string> batch;
	do {
		std::cout << "Please enter a sentence in English " << std::endl;
		batch.clear();
	    for (size_t i = 0; i < batchSizeOpt; i++) {
	    	std::string sentence;
	    	if (!getline(std::cin, sentence)) {
	    		break;
	    	}
	    	batch.push_back(sentence);
	    }
	    if (!batch.empty()) {
	      gnmt.translate(batch);
	    }
	} while (batch.size() == batchSizeOpt);

	return 0;
}
