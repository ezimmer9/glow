

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

	Model seq2seq(batchSizeOpt);
	seq2seq.loadTokens();
	seq2seq.loadEncoder();
	seq2seq.loadDecoder();
	seq2seq.compile();

	if (!dumpGraphDAGFileOpt.empty()) {
	    seq2seq.dumpGraphDAG(dumpGraphDAGFileOpt.c_str());
	}

	if (debugMode){
		std::vector<std::string> debug = {"They are"};
		seq2seq.translate(debug);
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
	      seq2seq.translate(batch);
	    }
	} while (batch.size() == batchSizeOpt);

	return 0;
}
