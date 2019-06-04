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


void debug_size_print(Node *candident){
	TypeRef t_1 =  candident->getType(0);std::cout << candident->getName().str() << " dims:  ";
	for (uint i=0 ; i < t_1->dims().size(); i++){
		std::cout << t_1->dims()[i] << " " ;
	}
	std::cout << " " << std::endl << std::endl;
}

void Model::loadEncoder(){
	std::printf("*** loadEncoder ***\n\n");

	auto &mod = EE_.getModule();

	input_ = mod.createPlaceholder(ElemKind::Int64ITy, {batchSize_, MAX_LENGTH}, "encoder.inputsentence", false);
	bindings.allocate(input_);

	seqLength_ = mod.createPlaceholder(ElemKind::Int64ITy, {batchSize_}, "encoder.seqLength", false);
	bindings.allocate(seqLength_);

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
	   F_->createGather("encoder.embedding", embedding_en_, input_);

	std::vector<NodeValue> hidenOutputs0;
	std::vector<Node *> hiddenOut0;
	std::vector<NodeValue> hidenOutputs1;
	std::vector<Node *> hiddenOut1;
	std::vector<NodeValue> hidenOutputs2;
	std::vector<Node *> hiddenOut2;
	std::vector<NodeValue> hidenOutputs3;
	std::vector<Node *> hiddenOut3;

	std::vector<Node *> enc_seq;

	for (unsigned word_inx=0 ; word_inx < MAX_LENGTH; word_inx++){
		Node *inputSlice = F_->createSlice("encoder.slice."+ std::to_string(word_inx),
				inputEmbedded , {0,word_inx,0}, {batchSize_ , word_inx+1 , EMBEDDING_SIZE});
		Node *reshape = F_->createReshape("encofer.reshape." + std::to_string(word_inx),
				inputSlice,{batchSize_, EMBEDDING_SIZE});
		enc_seq.push_back(reshape);
	}

	for (int word_inx=0 ; word_inx < 1; word_inx++){
		F_->createLSTM(bindings, "encoder."+std::to_string(word_inx)+".lstm", enc_seq ,
				batchSize_, EMBEDDING_SIZE , HIDDEN_SIZE, hidenOutputs0);
		for (uint i = 0 ; i < hidenOutputs0.size() ; i++){
			hiddenOut0.push_back(hidenOutputs0[i].getNode());
		}

		F_->createLSTM(bindings,"encoder."+std::to_string(word_inx)+".lstm1",hiddenOut0 ,
				batchSize_, HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs1);
		for (uint i = 0 ; i < hidenOutputs1.size() ; i++){
			hiddenOut1.push_back(hidenOutputs1[i].getNode());
		}

		F_->createLSTM(bindings,"encoder."+std::to_string(word_inx)+".lstm2",hiddenOut1 ,
				batchSize_, HIDDEN_SIZE  , HIDDEN_SIZE, hidenOutputs2);
		for (uint i = 0 ; i < hidenOutputs2.size() ; i++){
			hiddenOut2.push_back(hidenOutputs2[i].getNode());
		}

		GLOW_ASSERT(hidenOutputs2.size() == hidenOutputs1.size() && "LSTM outputs doesn't align");
		for (uint i = 0 ; i < hidenOutputs2.size() ; i++){
			Node *add1 = F_->createAdd("encoder."+std::to_string(i)+".residual",hidenOutputs2[i], hidenOutputs1[i]);
			hiddenOut3.push_back(add1);
		}
		F_->createLSTM(bindings,"encoder."+std::to_string(word_inx)+".lstm3",hiddenOut3 ,
				batchSize_, HIDDEN_SIZE  , HIDDEN_SIZE, hidenOutputs3);

		hidenOutputs0.clear();
		hidenOutputs1.clear();
		hidenOutputs2.clear();
		hiddenOut0.clear();
		hiddenOut1.clear();
		hiddenOut2.clear();
		hiddenOut3.clear();
	}

	Node *output = F_->createConcat("encoder.output", hidenOutputs3, 1);
	hidenOutputs3.clear();
	// TODO: maybe no need here the reshape concat 3rd arg is dimension
	encoderHiddenOutput_ = F_->createReshape("encoder.output.reshape", output,
			{batchSize_, MAX_LENGTH , HIDDEN_SIZE});

	//  ******** example how to change the lstm weights *******
	//  *******************************************************
	//auto *test = mod.getPlaceholderByName("encoder_inputsentence");
	//  convert all placeholders to constants except test
	//::convertPlaceholdersToConstants(F_ , bindings , {test});
	//ConstList consts = F_->findConstants();
	//for (auto con : consts){
	//	std::cout << "constant: " << con->getName().str ()<< "\n";
	//}
	//Constant *test1 = mod.getConstantByName("initial_cell_state391");
	//std::cout << "tets1 pointer: " << test1 << "\n";
	//llvm::ArrayRef<size_t> aref = test1->getPayload().dims();
	//for (uint i=0; i<aref.size(); i++){
	//	std::cout << " aref[i]: " << aref[i];
	//}
	//std::cout << "\n";
	//std::vector<float> ten(768, 1);
	//test1->getPayload().getHandle() = ten;
	//std::printf("Ten in 1st place: %f\n", ten[0]);

}

std::vector<Node *> Model::loadAttention(std::vector<Node *> AttentionQuery){
	std::printf("*** loadAttention ***\n\n");

	auto &mod = EE_.getModule();
	auto Wa = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE , HIDDEN_SIZE},
			"attention.1.Wfc1" , false);
	bindings.allocate(Wa)->zero();

	auto Bwa = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE},
				"attention.1.Bfc1" , false);
	bindings.allocate(Bwa)->zero();

	auto Ua = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE , HIDDEN_SIZE},
			"attention.2.Winside" , false);
	bindings.allocate(Ua)->zero();

	auto Bua = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE},
			"attention.2.Binside" , false);
	bindings.allocate(Bua)->zero();

	auto We = mod.createPlaceholder(ElemKind::FloatTy, {MAX_LENGTH,HIDDEN_SIZE}, "attention.1.Wout", false);
	bindings.allocate(We)->zero();
	auto Bwe = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE}, "attention.1.Bout", false);
	bindings.allocate(Bwe)->zero();

	Placeholder *Vt = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE,1}, "attention.Vt", false);
	Node *NVt = Vt;
	bindings.allocate(Vt)->zero();

	Placeholder *SM = mod.createPlaceholder(ElemKind::Int64ITy, {1 , 1}, "attention.softmax.ph", false);

//	Node *expandKeys = F_->createExpandDims("encoder.expand.output", encoderHiddenOutput_ ,
//			{1});
//	debug_size_print(expandKeys);

	std::vector<Node *> attentionOut;
	std::vector<NodeValue> Qexpand;
	for (uint i=0 ; i < AttentionQuery.size() ; i++){
		debug_size_print(AttentionQuery[i]);

		//TODO: not need to do every iteration
		// this is instead of fullyconected that support only 2 dimentions.
		Node *BroadKeys = F_->createBroadcastedBatchMatMul("attention.broadkeys",
				encoderHiddenOutput_,Wa);
		Node *BroadForAdd = F_->createBroadcast("attention.BroadForAdd",
				Bwa , {batchSize_, MAX_LENGTH , HIDDEN_SIZE} , 2);
		Node *AddKeys = F_->createAdd("attention.addkeys", BroadKeys, BroadForAdd);
		debug_size_print(AddKeys);

		Node *QueryExpand = F_->createExpandDims("attention."+ std::to_string(i)+ ".expandquery",
				AttentionQuery[i], {1});;
		debug_size_print(QueryExpand);

		// this is instead of fullyconected that support only 2 dimentions.
		Node *BroadQuery = F_->createBroadcastedBatchMatMul("attention.broadquery",
				QueryExpand ,Ua);
		Node *BroadForQureAdd = F_->createBroadcast("attention.BroadForQureAdd",
				Bua , {batchSize_, 1 , HIDDEN_SIZE} , 2);
		Node *AddQuery = F_->createAdd("attention.addquery", BroadQuery, BroadForQureAdd);
		debug_size_print(AddQuery);

		Node *copyQuery = F_->createConcat("attention.concatquery",{AddQuery,AddQuery,AddQuery},1);
		debug_size_print(copyQuery);
		Node *THinside = F_->createTanh("attention.TanH.inside" , F_->createAdd (
				"attention.add." + std::to_string(i) + ".inside", AddKeys, copyQuery));
		debug_size_print(THinside);
		Node *Amul = F_->createBroadcastedBatchMatMul("attention."+ std::to_string(i) + ".matmul"
				,THinside,NVt);
		debug_size_print(Amul);
		Node *Asoftmax = F_->createSoftMax("attention."+std::to_string(i) + ".softmax" , Amul , SM);
		debug_size_print(Asoftmax);
		// **** need to check where is the transpose happen ****
//		Node *ATranspose = F_->createTranspose("attention."+ std::to_string(i) + ".transpose",Asoftmax , {1 ,0});
//		debug_size_print(ATranspose);
		Node *AFCout1 = F_->createFullyConnected("attention."+ std::to_string(i) + ".fc1.out" , Asoftmax, We, Bwe);
		debug_size_print(AFCout1);
		attentionOut.push_back(AFCout1);
		Qexpand.clear();
	}


	return attentionOut;

}

void Model::loadDecoder(){
	std::printf("*** loadDecoder ***\n\n");
	auto &mod = EE_.getModule();

	std::vector<Node *> dec_seq;
	for (uint i=0 ; i < MAX_LENGTH ; i++){
		Placeholder *hiddenInit = mod.createPlaceholder(ElemKind::FloatTy,
				{batchSize_, EMBEDDING_SIZE}, "decoder."+std::to_string(i)+".hiddenInit", false);
		bindings.allocate(hiddenInit)->zero();
		Node *seqInit = hiddenInit;
		dec_seq.push_back(seqInit);
	}

	//TODO: init the 1st dec_seq with </s> embedding
	// the Initial vector fot the decoder will be:
	// { embedding(</s>), 0 , 0 ,0 ...}

	std::vector<NodeValue> hidenOutputs0;
	std::vector<Node *> hiddenOut0;
	std::vector<NodeValue> hidenOutputs1;
	std::vector<Node *> hiddenOut1;
	std::vector<NodeValue> hidenOutputs2;
	std::vector<Node *> hiddenOut2;
	std::vector<NodeValue> hidenOutputs3;
	std::vector<Node *> hiddenOut3;

	for (uint word_inx=0 ; word_inx < MAX_LENGTH; word_inx++){
		std::printf("***************** word_ind %u***************\n", word_inx);
		F_->createLSTM(bindings, "decoder.lstm.0."+std::to_string(word_inx), dec_seq ,
				batchSize_, HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs0);
		for (uint i = 0 ; i < hidenOutputs0.size() ; i++){
			hiddenOut0.push_back(hidenOutputs0[i].getNode());
		}

		std::vector<Node *> attentionOut = loadAttention(hiddenOut0);
		for (uint i=0 ; i < hiddenOut0.size() ; i++){
			NodeValue attentionOutNV(attentionOut[i]);
			debug_size_print(attentionOutNV);
			NodeValue hiddenOut0NV(hiddenOut0[i]);
			debug_size_print(hiddenOut0NV);
			attentionOut[i] = F_->createConcat("decoder.concat", {attentionOutNV, hiddenOut0NV},1);
		}

		F_->createLSTM(bindings, "decoder.lstm.1."+std::to_string(word_inx), attentionOut ,
				batchSize_, /* 2*/HIDDEN_SIZE , EMBEDDING_SIZE, hidenOutputs1);
		for (uint i = 0 ; i < hidenOutputs1.size() ; i++){
			hiddenOut1.push_back(hidenOutputs1[i].getNode());
		}

		F_->createLSTM(bindings, "decoder.lstm.2."+std::to_string(word_inx), hiddenOut1 ,
				batchSize_, /* 2*/HIDDEN_SIZE , EMBEDDING_SIZE, hidenOutputs2);
		for (uint i = 0 ; i < hidenOutputs2.size() ; i++){
			hiddenOut2.push_back(hidenOutputs2[i].getNode());
		}

		GLOW_ASSERT(hidenOutputs2.size() == hidenOutputs1.size() && "LSTM outputs doesn't align");
		for (uint i = 0 ; i < hidenOutputs2.size() ; i++){
			Node *add1 = F_->createAdd("decoder."+std::to_string(i)+".residual",hidenOutputs2[i], hidenOutputs1[i]);
			hiddenOut3.push_back(add1);
		}
		F_->createLSTM(bindings, "decoder.lstm.3."+std::to_string(word_inx), hiddenOut3 , batchSize_,
				/* 2*/HIDDEN_SIZE , EMBEDDING_SIZE, hidenOutputs3);

		hidenOutputs0.clear();
		hidenOutputs1.clear();
		hidenOutputs2.clear();
		hiddenOut0.clear(); hiddenOut1.clear(); hiddenOut2.clear(); hiddenOut3.clear();
		// TODO: think how to avoid the last allocation for dec_seq
		if (word_inx+1 < MAX_LENGTH){
			dec_seq[word_inx+1] = hidenOutputs3[word_inx];
		}
		hidenOutputs3.clear();

	}
	std::vector<NodeValue> lstmOutput;
	for (uint i=0 ; i<dec_seq.size() ; i++){
		NodeValue tmpNodeValue(dec_seq[i]);
		lstmOutput.push_back(tmpNodeValue);
	}
	Node *output = F_->createConcat("dencoder.output", lstmOutput, 1);
	Placeholder *S = mod.createPlaceholder(ElemKind::Int64ITy, {batchSize_, 1}, "S", false);
	auto *SM = F_->createSoftMax("softmax", output, S);
	auto *save = F_->createSave("decoder.output", SM);
	output_ = save->getPlaceholder();
	bindings.allocate(output_);
}

void Model::translate(const std::vector<std::string> &batch){
	std::printf("*** translate ***\n\n");
	Tensor input(ElemKind::Int64ITy, {batchSize_, MAX_LENGTH});
	Tensor seqLength(ElemKind::Int64ITy, {batchSize_});
	input.zero();

	for (size_t j = 0; j < batch.size(); j++) {
		std::istringstream iss(batch[j]);
	    std::vector<std::string> words;
	    std::string word;
	    while (iss >> word)
	      words.push_back(word);
	    words.push_back("EOS");

	    GLOW_ASSERT(words.size() <= MAX_LENGTH && "sentence is too long.");

	    for (size_t i = 0; i < words.size(); i++) {
	      auto iter = en_.word2index_.find(words[i]);
	      GLOW_ASSERT(iter != en_.word2index_.end() && "Unknown word.");
	      input.getHandle<int64_t>().at({j, i}) = iter->second;
	    }
	    seqLength.getHandle<int64_t>().at({j}) =
	        (words.size() - 1) + j * MAX_LENGTH;
	  }
	updateInputPlaceholders(bindings, {input_, seqLength_}, {&input, &seqLength});
	EE_.run(bindings);

	auto OH = bindings.get(output_)->getHandle<float_t>();
	for (unsigned j = 0; j < batch.size(); j++) {
		for (unsigned i = 0; i < MAX_LENGTH; i++) {
			int64_t wordIdx = (int64_t)OH.at({i, j});
			if (wordIdx == gr_.word2index_["EOS"])
				break;

			if (i)
				std::cout << ' ';
				std::cout << wordIdx;
		}
		std::cout << "\n\n";
	}

	if (!dumpProfileFileOpt.empty()) {
		std::vector<NodeQuantizationInfo> QI =
	        quantization::generateNodeQuantizationInfos(bindings, F_, loweredMap_);
		serializeToYaml(dumpProfileFileOpt, QI);
	}
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
