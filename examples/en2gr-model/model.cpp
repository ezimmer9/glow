/*
 * model.cpp
 *
 *  Created on: May 5, 2019
 *      Author: ezimmer9
 */

#include "model.h"

using namespace glow;

int debug_counter = 0;

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

llvm::cl::opt<std::string> ExecutionBackend(
		"backend",
	    llvm::cl::desc("Backend to use, e.g., Interpreter, CPU, OpenCL:"),
	    llvm::cl::Optional, llvm::cl::init("Interpreter"), llvm::cl::cat(en2grCat));

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

llvm::cl::opt<bool> debugMode(
		"debug",
		llvm::cl::desc("print debug placeholders"),
		llvm::cl::init(false),
		llvm::cl::cat(en2grCat));

llvm::cl::opt<bool> trace(
		"trace",
		llvm::cl::desc("tracing and output"),
		llvm::cl::init(false),
		llvm::cl::cat(en2grCat));

llvm::cl::opt<int> beamSizeOpt(
     "beamsize", llvm::cl::desc("beam size option"),
     llvm::cl::init(1), llvm::cl::value_desc("N"), llvm::cl::cat(en2grCat));

llvm::cl::opt<uint> maxLengthOpt(
     "maxlength", llvm::cl::desc("max length option"),
     llvm::cl::init(4), llvm::cl::value_desc("N"), llvm::cl::cat(en2grCat));

llvm::cl::opt<std::string> inputOpt(
		"input",
	    llvm::cl::desc("input sentence"),
	    llvm::cl::init(" "), llvm::cl::cat(en2grCat));

llvm::cl::opt<std::string> file(
		"file",
	    llvm::cl::desc("input file with sentences"),
	    llvm::cl::init(" "), llvm::cl::cat(en2grCat));

void debug_size_print(Node *candident){
	std::cout << " " << candident->getDebugDesc() << std::endl << std::endl;
}

Placeholder *Model::loadEmbedding(llvm::StringRef langPrefix, size_t langSize) {
	  auto &mod = EE_.getModule();
	  auto *result =
			  mod.createPlaceholder(ElemKind::FloatTy, {langSize, EMBEDDING_SIZE},
					  "embedding." + langPrefix.str(), false);
	  auto test = (*context.getPlaceholderBindings()).allocate(result);
	  common_->loadMatrixFromFile("en2gr/" + langPrefix.str() + "_embedding.bin",*test);
	  return result;
}

void Model::loadTokens(){
	std::printf("*** loadTokens ***\n\n");
	tok_.loadVocabularyFromFile("../examples/en2gr-model/vocab.bpe.32000");
	embedding_tok_ = loadEmbedding("encoder.embedder.weight", tok_.index2word_.size()+4);
}

void Model::loadEncoder(){
	auto &mod = EE_.getModule();

	input_ = mod.createPlaceholder(ElemKind::Int64ITy, {batchSize_, max_length}, "encoder.inputsentence", false);
	(*context.getPlaceholderBindings()).allocate(input_);
	encoderHiddenOutput_ = m_encoder->loadEncoder(embedding_tok_ , input_);
}

void Model::loadBeamDecoder(){
	m_decoder->init();
	BeamDecoderOutput decoderout = m_decoder->loadBeamDecoder();
	auto *save = F_->createSave("decoder.output.candidents" , decoderout.candidates);
	output_ = save->getPlaceholder();
	(*context.getPlaceholderBindings()).allocate(output_);
	auto *save1 = F_->createSave("decoder.output.scores" , decoderout.scores);
	scores_ = save1->getPlaceholder();
	(*context.getPlaceholderBindings()).allocate(scores_);
}

void Model::loadGreadyDecoder(){
	m_decoder->init();
	Node * decoderout = m_decoder->loadGreadyDecoder();
	auto *save = F_->createSave("decoder.output", decoderout);
	output_ = save->getPlaceholder();
	(*context.getPlaceholderBindings()).allocate(output_);
}

void Model::translate(const std::vector<std::string> &batch){
	std::printf("*** translate ***\n\n");
	Tensor input(ElemKind::Int64ITy, {batchSize_, max_length});
	Tensor attention_mask(ElemKind::FloatTy , {beam_size,max_length , 1});
	input.zero();
	size_t words_size;

	std::cout << "En: \n";
	for (size_t j = 0; j < batch.size(); j++) {
		std::istringstream iss(batch[j]);
	    std::vector<std::string> words;
	    std::string word;
	    while (iss >> word)
	      words.push_back(word);

	    assert(words.size() <= (max_length-2) && "sentence is too long.");
	    words_size = words.size()+2;
	    input.getHandle<int64_t>().at({j, 0}) = BOS ;
	    std::cout << "BOS: " << input.getHandle<int64_t>().at({j, 0}) << "  ";
	    for (size_t i = 0; i < words.size(); i++) {
	      auto iter = tok_.word2index_.find(words[i]);
	      if (iter == tok_.word2index_.end()){
	    	  std::cout << "Unknown word, can't find it on the vocab!\n\n";
	    	  return;
	      }
	      std::cout << iter->first << ": " << iter->second +4 << "  ";
	      input.getHandle<int64_t>().at({j, i+1}) = (iter->second)+4;
	    }
	    input.getHandle<int64_t>().at({j, words.size()+1}) = EOS ;
	    std::cout << "EOS: " << ": " << input.getHandle<int64_t>().at({j, words.size()+1}) << "  ";
	    size_t i=words.size()+2;
	    while(i!=max_length){
	    	input.getHandle<int64_t>().at({j, i}) = PAD ;
		    std::cout << "PAD: " << ": " << input.getHandle<int64_t>().at({j, i}) << "  ";
	    	i++;
	    }
	  }
	std::cout << "\n\n";

	for(size_t i=0; i< beam_size; i++){
		for(size_t j=0; j< (max_length); j++){
			if(j>=words_size) //TODO read value from placeholder, src_len or (srclenTensor->getHandle<int64_t>().at({0,0}))+2)
				attention_mask.getHandle<float_t>().at({i,j,0})=-1000000.f;
	    	else
	    		attention_mask.getHandle<float_t>().at({i,j,0})=0.f;
		}
	}

	updateInputPlaceholders(*(context.getPlaceholderBindings()),
			{input_, attention_mask_}, {&input , &attention_mask});
	EE_.run(context);

	auto OH = context.getPlaceholderBindings()->get(output_)->getHandle<int64_t>();
	std::cout << "Gr: \n";
	if (beam_size != 1){
		std::vector<float_t> scores;
		auto OSH = context.getPlaceholderBindings()->get(scores_)->getHandle<float_t>();
		for (unsigned i = 0; i < beam_size; i++) {
			float_t score = (float_t)OSH.at({i});
//			std::cout << "the score: " << score << "\n";
			scores.push_back(score);
		}

		int maxElementIndex = std::max_element(scores.begin(),scores.end()) - scores.begin();
//		std::cout << "the index is: " << maxElementIndex << "\n";

		for (unsigned j = 0; j < batch.size(); j++) {
			for (unsigned i = 0; i < max_length; i++) {
				int64_t wordIdx = (int64_t)OH.at({i, maxElementIndex});
				if (wordIdx == 3)
					break;

				if (wordIdx)
					std::cout << ' ';
					std::cout << wordIdx << ": " << tok_.index2word_[wordIdx-4];
			}
			std::cout << "\n\n";
		}

	}
	else{
		for (unsigned j = 0; j < batch.size(); j++) {
			for (unsigned i = 0; i < max_length; i++) {
				int64_t wordIdx = (int64_t)OH.at({i, j});
				if (wordIdx == 3)
					break;

				if (wordIdx)
					std::cout << ' ';
					std::cout << wordIdx << ": " << tok_.index2word_[wordIdx-4];
			}
			std::cout << "\n\n";
		}
	}

	if (trace){
		std::vector<TraceEvent>& events = context.getTraceContext()->getTraceEvents();
		TraceEvent::dumpTraceEvents(events, "glow-trace.json", "glow");
	}

	if (!dumpProfileFileOpt.empty()) {
		std::vector<NodeQuantizationInfo> QI =
	        quantization::generateNodeQuantizationInfos(*context.getPlaceholderBindings(), F_, loweredMap_);
		serializeToYaml(dumpProfileFileOpt, QI);
	}
}

void Model::compile() {
	std::printf("*** compile ***\n\n");

    std::vector<Placeholder *> placeHolderVec;
    placeHolderVec.push_back(input_);
    placeHolderVec.push_back(output_);
    placeHolderVec.push_back(attention_mask_);
    if (beam_size != 1){
    	placeHolderVec.push_back(scores_);
    }
    ::glow::convertPlaceholdersToConstants(F_, *context.getPlaceholderBindings(), placeHolderVec);
//    ConstList consts = F_->findConstants();
//    	for (auto con : consts){
//    		std::cout << "constant: " << con->getName().str ()<< "\n";
//   	}
	m_encoder->loadEncoderWieghts();
	m_encoder->loadEncoderBiases();
	m_encoder->loadEncoderReverse();
    m_decoder->loadDecoderWieghts();
    m_decoder->loadDecoderBiases();

    CompilationContext cctx{&*context.getPlaceholderBindings(), &loweredMap_};
    if (trace){
    	cctx.backendOpts.autoInstrument = true;
    }
    PrecisionConfiguration &precConfig = cctx.precisionConfig;

    if (!dumpProfileFileOpt.empty()) {
      precConfig.quantMode = QuantizationMode::Profile;
    }

    // Load the quantization profile and transform the graph.
    if (!loadProfileFileOpt.empty()) {
      precConfig.quantMode = QuantizationMode::Quantize;
      precConfig.quantConfig.infos = deserializeFromYaml(loadProfileFileOpt);
      precConfig.quantConfig.assertAllNodesQuantized = true;
      if (ExecutionBackend == "CPU" || ExecutionBackend == "NNPI"){
    	  precConfig.precisionModeKindSet.insert(Kinded::Kind::SoftMaxNodeKind);
    	  precConfig.precisionModeKindSet.insert(Kinded::Kind::PowNodeKind);
      }
    }


    context.setTraceContext(llvm::make_unique<TraceContext>(TraceLevel::STANDARD));
    EE_.compile(cctx);
    F_ = EE_.getModule().getFunctions().front();
//    FunctionList debug = EE_.getModule().getFunctions();
//    for (auto F : debug){
//    	F->dump();
//    }
}
