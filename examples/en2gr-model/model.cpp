/*
 * model.cpp
 *
 *  Created on: May 5, 2019
 *      Author: ezimmer9
 */


#include "model.h"
using namespace glow;


/// Debugging options.
llvm::cl::OptionCategory debugCat("Glow Debugging Options");

/// Translator options.
llvm::cl::OptionCategory en2grCat("English-to-German Translator Options");

llvm::cl::opt<unsigned> batchSizeOpt(
    "batchsize", llvm::cl::desc("Process batches of N sentences at a time."),
    llvm::cl::init(1), llvm::cl::value_desc("N"), llvm::cl::cat(en2grCat));
llvm::cl::alias batchSizeA("b", llvm::cl::desc("Alias for -batchsize"),
                           llvm::cl::aliasopt(batchSizeOpt),
                           llvm::cl::cat(en2grCat));

llvm::cl::opt<bool>
    timeOpt("time",
            llvm::cl::desc("Print timer data detailing how long it "
                           "takes for the program to execute translate phase. "
                           "This option will be useful if input is read from "
                           "the file directly."),
            llvm::cl::Optional, llvm::cl::cat(en2grCat));

llvm::cl::opt<BackendKind> ExecutionBackend(
    llvm::cl::desc("Backend to use:"), llvm::cl::Optional,
    llvm::cl::values(clEnumValN(BackendKind::Interpreter, "interpreter",
                                "Use interpreter"),
                     clEnumValN(BackendKind::CPU, "cpu", "Use CPU"),
                     clEnumValN(BackendKind::OpenCL, "opencl", "Use OpenCL")),
    llvm::cl::init(BackendKind::Interpreter), llvm::cl::cat(en2grCat));

/// Quantization options.
llvm::cl::OptionCategory quantizationCat("Quantization Options");

llvm::cl::opt<std::string> dumpProfileFileOpt(
    "dump-profile",
    llvm::cl::desc("Perform quantization profiling for a given graph "
                   "and dump result to the file."),
    llvm::cl::value_desc("profile.yaml"), llvm::cl::Optional,
    llvm::cl::cat(quantizationCat));

llvm::cl::opt<std::string> loadProfileFileOpt(
    "load-profile",
    llvm::cl::desc("Load quantization profile file and quantize the graph"),
    llvm::cl::value_desc("profile.yaml"), llvm::cl::Optional,
    llvm::cl::cat(quantizationCat));

llvm::cl::opt<std::string> dumpGraphDAGFileOpt(
    "dump-graph-DAG",
    llvm::cl::desc("Dump the graph to the given file in DOT format."),
    llvm::cl::value_desc("file.dot"), llvm::cl::cat(debugCat));

// namespace

/// Loads tensor of floats from binary file.
void loadMatrixFromFile(llvm::StringRef filename, Tensor &result) {
  std::ifstream file(filename.str(), std::ios::binary);
  if (!file.read(result.getUnsafePtr(), result.size() * sizeof(float))) {
    std::cout << "Error reading file: " << filename.str() << '\n'
              << "Need to be downloaded by calling:\n"
              << "python ../glow/utils/download_test_db.py -d fr2en\n";
    exit(1);
  }
}


void Model::loadLanguages(){
	std::printf("*** loadLanguages ***\n\n");
	gr_.loadVocabularyFromFile("en2gr/fr_vocabulary.txt");
	en_.loadVocabularyFromFile("en2gr/en_vocabulary.txt");
	embedding_gr_ = loadEmbedding("fr", gr_.index2word_.size());
	embedding_en_ = loadEmbedding("en", en_.index2word_.size());
}

void Model::loadEncoder(){
	std::printf("*** loadEncoder ***\n\n");

	auto &mod = EE_.getModule();

	input_ = mod.createPlaceholder(ElemKind::Int64ITy, {batchSize_, MAX_LENGTH}, "encoder.inputsentence", false);
	bindings.allocate(input_);

	seqLength_ = mod.createPlaceholder(ElemKind::Int64ITy, {batchSize_}, "encoder.seqLength", false);
	bindings.allocate(seqLength_);

	auto *hiddenInit = mod.createPlaceholder(ElemKind::FloatTy, {batchSize_, EMBEDDING_SIZE}, "encoder.hiddenInit", false);
	auto *hiddenInitTensor = bindings.allocate(hiddenInit);
	hiddenInitTensor->zero();

	Node *hidden = hiddenInit;

	auto *wIh = mod.createPlaceholder(
			ElemKind::FloatTy, {EMBEDDING_SIZE, HIDDEN_SIZE}, "encoder.w_ih", false);
	auto *bIh = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE},
									 "encoder.b_ih", false);
	auto *wHh = mod.createPlaceholder(
			ElemKind::FloatTy, {EMBEDDING_SIZE, HIDDEN_SIZE}, "encoder.w_hh", false);
	auto *bHh = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE},
									 "encoder.b_hh", false);

	loadMatrixFromFile("en2gr/encoder_w_ih.bin", *bindings.allocate(wIh));
	loadMatrixFromFile("en2gr/encoder_b_ih.bin", *bindings.allocate(bIh));
	loadMatrixFromFile("en2gr/encoder_w_hh.bin", *bindings.allocate(wHh));
	loadMatrixFromFile("en2gr/encoder_b_hh.bin", *bindings.allocate(bHh));

	Node *inputEmbedded =
	   F_->createGather("encoder.embedding", embedding_gr_, input_);

	std::vector<NodeValue> hidenOutputs;
	std::vector<NodeValue> outputs;
	for (unsigned word_inx=0 ; word_inx < MAX_LENGTH; word_inx++){
		Node *inputSlice = F_->createSlice("encoder.slice"+ std::to_string(word_inx),
				inputEmbedded , {0,word_inx,0}, {batchSize_ , word_inx+1 , EMBEDDING_SIZE});
		Node *reshape = F_->createReshape("encofer.reshape" + std::to_string(word_inx),
				inputSlice,{batchSize_, EMBEDDING_SIZE});
		F_->createLSTM(bindings, "encoder.lstm", reshape , batchSize_, HIDDEN_SIZE , EMBEDDING_SIZE, hidenOutputs);

		F_->createLSTM(bindings, "encoder.lstm1", hidenOutputs[0].getNode() , batchSize_, HIDDEN_SIZE , EMBEDDING_SIZE, hidenOutputs);

		F_->createLSTM(bindings, "encoder.lstm2", hidenOutputs[1].getNode() , batchSize_, HIDDEN_SIZE , EMBEDDING_SIZE, hidenOutputs);
		Node *add1 = F_->createAdd("encoder.residual1",hidenOutputs[1].getNode(), hidenOutputs[2].getNode());
		F_->createLSTM(bindings, "encoder.lstm3", add1 , batchSize_, HIDDEN_SIZE , EMBEDDING_SIZE, outputs);
		hidenOutputs.clear();
	}

	Node *output = F_->createConcat("encoder.output", outputs, 1);

	encoderHiddenOutput_ = F_->createGather("encoder.outputNth", output, seqLength_);

}

void Model::loadAttention(){
	std::printf("*** loadAttention ***\n\n");

}

void Model::loadDecoder(){
	std::printf("*** loadDecoder ***\n\n");

	auto &mod = EE_.getModule();

//	auto *input = mod.createPlaceholder(ElemKind::Int64ITy, {4*EMBEDDING_SIZE}, "encoder.inputsentence", false);
//	bindings.allocate(input);
//
//	auto *hiddenInit = mod.createPlaceholder(ElemKind::FloatTy, {batchSize_, EMBEDDING_SIZE}, "encoder.hiddenInit", false);
//	auto *hiddenInitTensor = bindings.allocate(hiddenInit);
//	hiddenInitTensor->zero();
//
//	Node *hidden = hiddenInit;
//
//	auto *wIh = mod.createPlaceholder(ElemKind::FloatTy, {EMBEDDING_SIZE, HIDDEN_SIZE},
//			"dencoder.w_ih", false);
//	auto *bIh = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE},
//			"dencoder.b_ih", false);
//	auto *wHh = mod.createPlaceholder(ElemKind::FloatTy, {EMBEDDING_SIZE, HIDDEN_SIZE},
//			"dencoder.w_hh", false);
//	auto *bHh = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE},
//			"dencoder.b_hh", false);
//
//	loadMatrixFromFile("en2gr/encoder_w_ih.bin", *bindings.allocate(wIh));
//	loadMatrixFromFile("en2gr/encoder_b_ih.bin", *bindings.allocate(bIh));
//	loadMatrixFromFile("en2gr/encoder_w_hh.bin", *bindings.allocate(wHh));
//	loadMatrixFromFile("en2gr/encoder_b_hh.bin", *bindings.allocate(bHh));

	Node *inputSlice = F_->createSlice("dencoder.slice", encoderHiddenOutput_ ,
			{0,0}, {batchSize_ ,4 * EMBEDDING_SIZE});

	std::vector<NodeValue> outputs;
	F_->createLSTM(bindings, "dencoder.lstm", inputSlice , batchSize_, HIDDEN_SIZE , EMBEDDING_SIZE, outputs);

	F_->createLSTM(bindings, "dencoder.lstm1", outputs[0].getNode() , batchSize_, HIDDEN_SIZE , EMBEDDING_SIZE, outputs);

	F_->createLSTM(bindings, "dencoder.lstm2", outputs[1].getNode() , batchSize_, HIDDEN_SIZE , EMBEDDING_SIZE, outputs);

	F_->createLSTM(bindings, "dencoder.lstm3", outputs[2].getNode() , batchSize_, HIDDEN_SIZE , EMBEDDING_SIZE, outputs);

	Node *output = F_->createConcat("dencoder.output", outputs, 1);
	Placeholder *S = mod.createPlaceholder(ElemKind::Int64ITy, {batchSize_, 1}, "S", false);
	auto *SM = F_->createSoftMax("softmax", output, S);
	auto *save = F_->createSave("decoder.output", SM);
	output_ = save->getPlaceholder();
	bindings.allocate(output_);
}

void Model::translate(const std::vector<std::string> &batch){
	std::printf("*** translate ***\n\n");
}

Placeholder *Model::loadEmbedding(llvm::StringRef langPrefix, size_t langSize) {
	  auto &mod = EE_.getModule();
	  auto *result =
			  mod.createPlaceholder(ElemKind::FloatTy, {langSize, EMBEDDING_SIZE},
					  "embedding." + langPrefix.str(), false);
	  loadMatrixFromFile("en2gr/" + langPrefix.str() + "_embedding.bin",
                       *bindings.allocate(result));

	  return result;
}

void Model::compile() {
	std::printf("*** compile ***\n\n");
    if (!dumpProfileFileOpt.empty()) {
      // Perform the high-level optimizations before instrumenting the graph.
      // This optimization phase will remove stuff like repetitive transpose
      // operations perform CSE, etc.
      ::optimize(F_, glow::CompilationMode::Infer);

      // Lower everything for profile and log lowered info in loweredMap_. Used
      // later when creating quantization infos.
      ::lower(F_, &loweredMap_);

      // Instrument the graph to capture profiles for nodes' outputs.
      glow::profileQuantization(bindings, F_);
    }

    // Load the quantization profile and transform the graph.
    if (!loadProfileFileOpt.empty()) {
      // The profiled graph was optimized before it was instrumentated. In this
      // part of the code we repeat the same transformation in order to create
      // the same graph structure.
      glow::optimize(F_, CompilationMode::Infer);

      // Lower however the backend prefers.
      ::lower(F_, &loweredMap_, EE_.getBackend());

      quantization::QuantizationConfiguration quantConfig{
          deserializeFromYaml(loadProfileFileOpt)};

      // Quantize the graph based on the captured profile.
      quantization::quantizeFunction(F_, quantConfig, *EE_.getBackend(),
                                     loweredMap_);
    }

    // Do not create constants if we're profiling; the newly allocate histogram
    // vars will erroneously become constants.
    if (dumpProfileFileOpt.empty()) {
      ::glow::convertPlaceholdersToConstants(F_, bindings,
                                             {input_, seqLength_, output_});
    }
    EE_.compile(CompilationMode::Infer, F_);
}
