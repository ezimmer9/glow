

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
	seq2seq.loadLanguages();
	seq2seq.loadEncoder();
	seq2seq.loadAttention();
	seq2seq.loadDecoder();
	seq2seq.compile();

	if (!dumpGraphDAGFileOpt.empty()) {
	    seq2seq.dumpGraphDAG(dumpGraphDAGFileOpt.c_str());
	}
	return 0;
}
