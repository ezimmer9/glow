/*
 * model.cpp
 *
 *  Created on: May 5, 2019
 *      Author: ezimmer9
 */


#include "model.h"
#include "common.cpp"
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


void Model::loadLanguages(){
	std::printf("*** loadLanguages ***\n\n");
	gr_.loadVocabularyFromFile("en2gr/fr_vocabulary.txt");
	en_.loadVocabularyFromFile("en2gr/en_vocabulary.txt");
	embedding_gr_ = loadEmbedding("fr", gr_.index2word_.size());
	embedding_en_ = loadEmbedding("en", en_.index2word_.size());
}

void Model::loadTokens(){
	std::printf("*** loadTokens ***\n\n");
	tok_.loadVocabularyFromFile("/home/ezimmer9/Desktop/glow/examples/en2gr-model/vocab.bpe.32000");
	embedding_tok_ = loadEmbedding("encoder.embedder.weight", tok_.index2word_.size());
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


	Node *inputEmbedded =
	   F_->createGather("encoder.embedding", embedding_tok_, input_);

	std::vector<NodeValue> hidenOutputs0;
	std::vector<Node *> hiddenOut0;
	std::vector<NodeValue> oppHidenOutputs0;
	std::vector<Node *> oppHiddenOut0;
	std::vector<NodeValue> hidenOutputs1;
	std::vector<Node *> hiddenOut1;
	std::vector<NodeValue> hidenOutputs2;
	std::vector<Node *> hiddenOut2;
	std::vector<NodeValue> hidenOutputs3;
	std::vector<Node *> hiddenOut3;
	std::vector<NodeValue> encOut;

	std::vector<Node *> enc_seq;
	std::vector<Node *> opposite_seq;

	for (unsigned word_inx=0 ; word_inx < MAX_LENGTH; word_inx++){
		Node *inputSlice = F_->createSlice("encoder.slice.",
				inputEmbedded , {0,word_inx,0}, {batchSize_ , word_inx+1 , EMBEDDING_SIZE});
		Node *reshape = F_->createReshape("encoder.reshape." + std::to_string(word_inx),
				inputSlice,{batchSize_, EMBEDDING_SIZE});
		enc_seq.push_back(reshape);

		Node *oppInputSlice = F_->createSlice("opp.encoder.slice.",
				inputEmbedded , {0,MAX_LENGTH-word_inx-1,0}, {batchSize_ ,MAX_LENGTH-word_inx, EMBEDDING_SIZE});
		Node *oppReshape = F_->createReshape("opp.encoder.reshape." + std::to_string(word_inx),
				oppInputSlice,{batchSize_, EMBEDDING_SIZE});
		opposite_seq.push_back(oppReshape);
	}

	// Bi-Directional LSTM.
	F_->createLSTM(bindings, "encoder.lstm0", enc_seq ,
			batchSize_, EMBEDDING_SIZE , HIDDEN_SIZE, hidenOutputs0);
	for (uint i = 0 ; i < hidenOutputs0.size() ; i++){
		hiddenOut0.push_back(hidenOutputs0[i].getNode());
	}

	F_->createLSTM(bindings, "encoder.opp.lstm", opposite_seq ,
			batchSize_, EMBEDDING_SIZE , HIDDEN_SIZE, oppHidenOutputs0);
	for (uint i = 0 ; i < oppHidenOutputs0.size() ; i++){
		oppHiddenOut0.push_back(oppHidenOutputs0[i].getNode());
	}
	std::vector<Node *> lstm0Concat;
	for (uint i=0 ; i < hiddenOut0.size() ; i++){
		NodeValue opphiddenOut0NV(oppHiddenOut0[i]);
		NodeValue hiddenOut0NV(hiddenOut0[i]);
		lstm0Concat.push_back(F_->createConcat("encoder."+std::to_string(i)+".concat",
				{opphiddenOut0NV, hiddenOut0NV},1));
	}

	// ---------------- end Bi-Directional LSTM --------------

	F_->createLSTM(bindings,"encoder.lstm1",lstm0Concat ,
			batchSize_, /* 2(?)*/HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs1);
	for (uint i = 0 ; i < hidenOutputs1.size() ; i++){
		hiddenOut1.push_back(hidenOutputs1[i].getNode());
	}

	F_->createLSTM(bindings,"encoder.lstm2",hiddenOut1 ,
			batchSize_, HIDDEN_SIZE  , HIDDEN_SIZE, hidenOutputs2);
	for (uint i = 0 ; i < hidenOutputs2.size() ; i++){
		hiddenOut2.push_back(hidenOutputs2[i].getNode());
	}

	GLOW_ASSERT(hidenOutputs2.size() == hidenOutputs1.size() && "LSTM outputs doesn't align");
	for (uint i = 0 ; i < hidenOutputs2.size() ; i++){
		Node *add1 = F_->createAdd("encoder."+std::to_string(i)+".residual.",hidenOutputs2[i], hidenOutputs1[i]);
		hiddenOut3.push_back(add1);
	}
	F_->createLSTM(bindings,"encoder.lstm3",hiddenOut3 ,
			batchSize_, HIDDEN_SIZE  , HIDDEN_SIZE, hidenOutputs3);

	GLOW_ASSERT(hidenOutputs3.size() == hiddenOut3.size() && "LSTM outputs doesn't align");
	for (uint i = 0 ; i < hidenOutputs2.size() ; i++){
		Node *add2 = F_->createAdd("encoder."+std::to_string(i)+".residual1.",hiddenOut3[i], hidenOutputs3[i]);
		NodeValue add2NV(add2);
		encOut.push_back(add2NV);
	}

	hidenOutputs0.clear();
	hidenOutputs1.clear();
	hidenOutputs2.clear();
	hidenOutputs3.clear();
	hiddenOut0.clear();
	hiddenOut1.clear();
	hiddenOut2.clear();
	hiddenOut3.clear();

	Node *output = F_->createConcat("encoder.output", encOut, 1);
	// TODO: maybe no need here the reshape concat 3rd arg is dimension
	encoderHiddenOutput_ = F_->createReshape("encoder.output.reshape", output,
			{batchSize_, MAX_LENGTH , HIDDEN_SIZE});

	//  ******** example how to change the lstm weights *******
	//  *******************************************************
//	auto *test = mod.getPlaceholderByName("encoder_inputsentence");
//	//  convert all placeholders to constants except test
//	::convertPlaceholdersToConstants(F_ , bindings , {test});
//	ConstList consts = F_->findConstants();
//	for (auto con : consts){
//		std::cout << "constant: " << con->getName().str ()<< "\n";
//	}
//	std::vector<Constant *> ConstTest;
//	Constant *test1 = mod.getConstantByName("encoder_lstm_Whf1");
//	ConstTest.push_back(test1);
//
//	Constant *test2 = mod.getConstantByName("encoder_lstm_Whi1");
//	ConstTest.push_back(test2);
//
//	Constant *test3 = mod.getConstantByName("encoder_lstm_Who1");
//	ConstTest.push_back(test3);
//
//	Constant *test4 = mod.getConstantByName("encoder_lstm_Whc1");
//	ConstTest.push_back(test4);
//
//	loadMatrixAndSplitFromFile("en2gr/encoder.rnn_layers.0.weight_hh_l0.bin" , ConstTest,4);
//	//std::cout << "tets1 pointer: " << test1 << "\n";
//	//llvm::ArrayRef<size_t> aref = test1->getPayload().dims();
//	//for (uint i=0; i<aref.size(); i++){
//	//	std::cout << " aref[i]: " << aref[i];
//	//}
//	//std::cout << "\n";
//	//std::vector<float> ten(768, 1);
//	//test1->getPayload().getHandle() = ten;
//	//std::printf("Ten in 1st place: %f\n", ten[0]);

}

std::vector<Node *> Model::loadAttention(std::vector<Node *> AttentionQuery){
	std::printf("*** loadAttention ***\n\n");

	auto &mod = EE_.getModule();
	auto Wa = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE , HIDDEN_SIZE},
			"attention.1.Wfc1_keys" , false);
	loadMatrixFromFile("en2gr/decoder.att_rnn.attn.linear_k.weight.bin", *bindings.allocate(Wa));

	auto Bwa = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE},
				"attention.1.Bfc1_keys" , false);
	bindings.allocate(Bwa)->zero();

	auto Ua = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE , HIDDEN_SIZE},
			"attention.2.Wfc2_query" , false);
	loadMatrixFromFile("en2gr/decoder.att_rnn.attn.linear_q.weight.bin", *bindings.allocate(Ua));

	auto Bua = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE},
			"attention.2.Bfc2_query" , false);
	bindings.allocate(Bua)->zero();

	auto We = mod.createPlaceholder(ElemKind::FloatTy, {MAX_LENGTH,HIDDEN_SIZE}, "attention.1.Wout", false);
	bindings.allocate(We)->zero();
	auto Bwe = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE}, "attention.1.Bout", false);
	bindings.allocate(Bwe)->zero();

	Placeholder *Vt = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE,1}, "attention.Vt", false);
	Node *NVt = Vt;
	loadMatrixFromFile("en2gr/decoder.att_rnn.attn.linear_att.bin", *bindings.allocate(Vt));

	Placeholder *SM = mod.createPlaceholder(ElemKind::Int64ITy, {1 , 1}, "attention.softmax.ph", false);

//	Node *expandKeys = F_->createExpandDims("encoder.expand.output", encoderHiddenOutput_ ,
//			{1});
//	debug_size_print(expandKeys);

	std::vector<Node *> attentionOut;
	std::vector<NodeValue> Qexpand;
	for (uint i=0 ; i < AttentionQuery.size() ; i++){
		//debug_size_print(AttentionQuery[i]);

		//TODO: not need to do every iteration
		// this is instead of fullyconected that support only 2 dimentions.
		Node *BroadKeys = F_->createBroadcastedBatchMatMul("attention.broadkeys",
				encoderHiddenOutput_,Wa);
		Node *BroadForAdd = F_->createBroadcast("attention.BroadForAdd",
				Bwa , {batchSize_, MAX_LENGTH , HIDDEN_SIZE} , 2);
		Node *AddKeys = F_->createAdd("attention.addkeys", BroadKeys, BroadForAdd);
		//debug_size_print(AddKeys);

		Node *QueryExpand = F_->createExpandDims("attention."+ std::to_string(i)+ ".expandquery",
				AttentionQuery[i], {1});;
		//debug_size_print(QueryExpand);

		// this is instead of fullyconected that support only 2 dimentions.
		Node *BroadQuery = F_->createBroadcastedBatchMatMul("attention.broadquery",
				QueryExpand ,Ua);
		Node *BroadForQureAdd = F_->createBroadcast("attention.BroadForQureAdd",
				Bua , {batchSize_, 1 , HIDDEN_SIZE} , 2);
		Node *AddQuery = F_->createAdd("attention.addquery", BroadQuery, BroadForQureAdd);
		//debug_size_print(AddQuery);
		auto vect = [](Node *AddQuery)
		{
			std::vector<NodeValue> vec;
			for (uint i=0 ; i < MAX_LENGTH ; i++){
			vec.push_back(AddQuery);
			}
			return vec;
		};
		//TODO: pass to the concat none hardcoded length
		Node *copyQuery = F_->createConcat("attention.concatquery",vect(AddQuery),1);
		//debug_size_print(copyQuery);
		Node *THinside = F_->createTanh("attention.TanH.inside" , F_->createAdd (
				"attention.add." + std::to_string(i) + ".inside", AddKeys, copyQuery));
		//debug_size_print(THinside);
		Node *Amul = F_->createBroadcastedBatchMatMul("attention."+ std::to_string(i) + ".matmul"
				,THinside,NVt);
		//debug_size_print(Amul);
		Node *Asoftmax = F_->createSoftMax("attention."+std::to_string(i) + ".softmax" , Amul , SM);
		//debug_size_print(Asoftmax);
		// **** need to check where is the transpose happen ****
//		Node *ATranspose = F_->createTranspose("attention."+ std::to_string(i) + ".transpose",Asoftmax , {1 ,0});
//		debug_size_print(ATranspose);
		Node *AFCout1 = F_->createFullyConnected("attention."+ std::to_string(i) + ".fc1.out" , Asoftmax, We, Bwe);
		//debug_size_print(AFCout1);
		attentionOut.push_back(AFCout1);
		Qexpand.clear();
	}


	return attentionOut;

}

void Model::loadDecoder(){
	std::printf("*** loadDecoder ***\n\n");
	auto &mod = EE_.getModule();

	std::vector<Node *> dec_seq;
	std::vector<Node *> dec_seq_embedding;
	for (uint i=0 ; i < MAX_LENGTH ; i++){
		Placeholder *hiddenInit = mod.createPlaceholder(ElemKind::Int64ITy,
				{batchSize_}, "decoder."+std::to_string(i)+".hiddenInit", false);
		auto *inputTensor = bindings.allocate(hiddenInit);
		for (size_t i = 0; i < batchSize_; i++) {
			inputTensor->getHandle<int64_t>().at({i}) = tok_.word2index_["SOS"];
		}
		Node *seqInit = hiddenInit;
		dec_seq.push_back(seqInit);

		Placeholder *hiddenInitEmbed = mod.createPlaceholder(ElemKind::Int64ITy,
				{batchSize_, EMBEDDING_SIZE}, "decoder."+std::to_string(i)+".hiddenInitEmbed", false);
		bindings.allocate(hiddenInitEmbed)->zero();
		Node *seqInitEmbed = hiddenInitEmbed;
		dec_seq_embedding.push_back(seqInitEmbed);
	}


	Placeholder *classifier_w = mod.createPlaceholder(ElemKind::FloatTy,
			{tok_.index2word_.size(),EMBEDDING_SIZE}, "decoder.classifier_w", false);
	loadMatrixFromFile("en2gr/decoder.classifier.classifier.weight.bin", *bindings.allocate(classifier_w));

	Placeholder *classifier_b = mod.createPlaceholder(ElemKind::FloatTy,
				{tok_.index2word_.size()}, "decoder.classifier_w", false);
	loadMatrixFromFile("en2gr/decoder.classifier.classifier.bias.bin", *bindings.allocate(classifier_b));
	Node *classifier_w_T = F_->createTranspose("decoder.classifier_w.transpose", (Node *)classifier_w , {1,0});

	Placeholder *S = mod.createPlaceholder(ElemKind::Int64ITy, {batchSize_, 1}, "S", false);

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

	std::vector<Node *> output_vec;

	for (uint word_inx=0 ; word_inx < MAX_LENGTH; word_inx++){
		std::printf("***************** word_ind %u***************\n", word_inx);


		//TODO: embedding need to be here

		Node *decoderEmbedded =
		   F_->createGather("decoder.embedding", embedding_tok_, dec_seq[word_inx]);
		debug_size_print(decoderEmbedded);

		dec_seq_embedding[word_inx] = decoderEmbedded;

		// lstm 0
		F_->createLSTM(bindings, "decoder.lstm.0."+std::to_string(word_inx), dec_seq_embedding ,
				batchSize_, HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs0);
		for (uint i = 0 ; i < hidenOutputs0.size() ; i++){
			hiddenOut0.push_back(hidenOutputs0[i].getNode());
		}

		std::vector<Node *> currentOut;
		// attention
		std::vector<Node *> attentionOut = loadAttention(hiddenOut0);
		for (uint i=0 ; i < hiddenOut0.size() ; i++){
			NodeValue attentionOutNV(attentionOut[i]);
			debug_size_print(attentionOutNV);
			NodeValue hiddenOut0NV(hiddenOut0[i]);
			debug_size_print(hiddenOut0NV);
		//concat 0
			currentOut.push_back(F_->createConcat("decoder."+std::to_string(i)+".concat",
					{attentionOutNV, hiddenOut0NV},1));
		}

		// lstm 1
		F_->createLSTM(bindings, "decoder.lstm.1."+std::to_string(word_inx), currentOut ,
				batchSize_, HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs1);
		for (uint i = 0 ; i < hidenOutputs1.size() ; i++){
			hiddenOut1.push_back(hidenOutputs1[i].getNode());
		}

		currentOut.clear();
		for (uint i=0 ; i < hiddenOut1.size() ; i++){
			NodeValue attentionOutNV(attentionOut[i]);
			debug_size_print(attentionOutNV);
			NodeValue hiddenOut1NV(hiddenOut1[i]);
			debug_size_print(hiddenOut1NV);
		// concat 1  - concat lstm1 with attentionout
			currentOut.push_back(F_->createConcat("decoder."+std::to_string(i)+".concat",
					{attentionOutNV, hiddenOut1NV},1));
		}

		//lstm 2
		F_->createLSTM(bindings, "decoder.lstm.2."+std::to_string(word_inx), currentOut ,
				batchSize_, HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs2);
		for (uint i = 0 ; i < hidenOutputs2.size() ; i++){
			hiddenOut2.push_back(hidenOutputs2[i].getNode());
		}
		// add residual 0 - lstm2 + attention-out
		std::vector<Node *> addResidual1;
		for (uint i=0 ; i < hiddenOut2.size() ; i++){
			addResidual1.push_back(F_->createAdd("add.residual1."+std::to_string(i),
					hiddenOut1[i] , hiddenOut2[i]));
		}

		currentOut.clear();
		for (uint i=0 ; i < addResidual1.size() ; i++){
			NodeValue attentionOutNV(attentionOut[i]);
			debug_size_print(attentionOutNV);
			NodeValue hiddenOut2NV(addResidual1[i]);
			debug_size_print(hiddenOut2NV);
		// concat 2 - concat addResidual1 with attentionout
			currentOut.push_back(F_->createConcat("decoder."+std::to_string(i)+".concat",
					{attentionOutNV, hiddenOut2NV},1));
		}

		// lstm 3
		F_->createLSTM(bindings, "decoder.lstm.3."+std::to_string(word_inx), currentOut , batchSize_,
				 HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs3);
		for (uint i = 0 ; i < hidenOutputs3.size() ; i++){
			hiddenOut3.push_back(hidenOutputs3[i].getNode());
		}

		// add residual 1 - LSTM2 + LSTM3
		std::vector<Node *> addResidual2;
		for (uint i=0 ; i < hiddenOut3.size() ; i++){
			addResidual2.push_back(F_->createAdd("add.residual2."+std::to_string(i),
					addResidual1[i] , hiddenOut3[i]));
		}

		hidenOutputs0.clear();
		hidenOutputs1.clear();
		hidenOutputs2.clear();
		hidenOutputs3.clear();
		hiddenOut0.clear(); hiddenOut1.clear(); hiddenOut2.clear(); hiddenOut3.clear();
		addResidual1.clear();
		// TODO: think how to avoid the last allocation for dec_seq

		// classifier
		Node *ClassifierW = F_->createFullyConnected("decoder.classifier.w",
				addResidual2[word_inx] , (Placeholder*)classifier_w_T, classifier_b);
		debug_size_print(ClassifierW);
		addResidual2.clear();

		// softmax
		Node *DecSM = F_->createSoftMax("decoder.softmax", ClassifierW, S);
		debug_size_print(DecSM);
		// topk
		auto *topK = F_->createTopK("decoder.topK", DecSM, 1);
		debug_size_print(topK);
	    Node *lastWordIdx =
	        F_->createReshape("decoder.reshape", topK->getIndices(), {batchSize_});
		debug_size_print(lastWordIdx);

		if (word_inx+1 < MAX_LENGTH){
			dec_seq[word_inx+1] = lastWordIdx;
		}
		output_vec.push_back(lastWordIdx);
	}

	std::vector<NodeValue> lstmOutput;
	for (uint i=0 ; i<output_vec.size() ; i++){
		debug_size_print(output_vec[i]);
		NodeValue tmpNodeValue(output_vec[i]);
		lstmOutput.push_back(output_vec[i]);
	}
	Node *output = F_->createConcat("dencoder.output", lstmOutput, 0);
	debug_size_print(output);
	Node *reshape = F_->createReshape("decoder.output.reshape", output,
	                                  {MAX_LENGTH, batchSize_});
	auto *save = F_->createSave("decoder.output", reshape);
	output_ = save->getPlaceholder();
	bindings.allocate(output_);
}

void Model::translate(const std::vector<std::string> &batch){
	std::printf("*** translate ***\n\n");
	Tensor input(ElemKind::Int64ITy, {batchSize_, MAX_LENGTH});
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
	      auto iter = tok_.word2index_.find(words[i]);
	      GLOW_ASSERT(iter != tok_.word2index_.end() && "Unknown word.");
	      input.getHandle<int64_t>().at({j, i}) = iter->second;
	    }
	  }
	updateInputPlaceholders(bindings, {input_}, {&input});
	EE_.run(bindings);

	auto OH = bindings.get(output_)->getHandle<int64_t>();
	for (unsigned j = 0; j < batch.size(); j++) {
		for (unsigned i = 0; i < MAX_LENGTH; i++) {
			int64_t wordIdx = (int64_t)OH.at({i, j});
			if (wordIdx == tok_.word2index_["EOS"])
				break;

			if (i)
				std::cout << ' ';
				std::cout << wordIdx << ": " << tok_.index2word_[wordIdx];
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
	  auto test = bindings.allocate(result);
	  loadMatrixFromFile("en2gr/" + langPrefix.str() + "_embedding.bin",
                       *test);

	  return result;
}

void Model::loadEncoderWieghts(){
	auto mod = F_->getParent();
    ConstList consts = F_->findConstants();
    for (auto con : consts){
    	std::cout << "constant: " << con->getName().str ()<< "\n";
    }
    std::vector<Constant *> ConstVecH, ConstVecX;
	std::vector<Constant *> ConstVecOppH, ConstVecOppX;
    for (uint j = 0 ; j < ENCODER_LSTMS_NUM ; ++j)
    {
		for (uint i = 0 ; i < LSTM_LEVELS ; ++i)
		{
			std::string const_name_h = "encoder_lstm"+ std::to_string(j) +"_Wh" + files_indices[i]+"1";
			std::string const_name_x = "encoder_lstm"+ std::to_string(j) +"_Wx" + files_indices[i]+"1";
			Constant *ConstH = mod->getConstantByName(const_name_h);
			Constant *ConstX = mod->getConstantByName(const_name_x);
			ConstVecH.push_back(ConstH);
			ConstVecX.push_back(ConstX);
			if (j == 0 ){
				std::string const_name_opp_h = "encoder_opp_lstm_Wh" + files_indices[i] +"1";
				std::string const_name_opp_x = "encoder_opp_lstm_Wx" + files_indices[i] +"1";
				Constant *ConstOppH = mod->getConstantByName(const_name_opp_h);
				Constant *ConstOppX = mod->getConstantByName(const_name_opp_x);
				ConstVecOppH.push_back(ConstOppH);
				ConstVecOppX.push_back(ConstOppX);

			}
		}
	   	loadMatrixAndSplitFromFile(
	   			"en2gr/encoder.rnn_layers."+ std::to_string(j) +".weight_hh_l0.bin" ,
				ConstVecH , LSTM_LEVELS);
	   	ConstVecH.clear();
	   	loadMatrixAndSplitFromFile(
	   			"en2gr/encoder.rnn_layers."+ std::to_string(j) +".weight_ih_l0.bin" ,
				ConstVecX , LSTM_LEVELS);
	   	ConstVecX.clear();
	   	if (j == 0){
	   		loadMatrixAndSplitFromFile(
		   			"en2gr/encoder.rnn_layers.0.weight_hh_l0_reverse.bin" ,
					ConstVecOppH , LSTM_LEVELS);
		   	ConstVecOppH.clear();
		   	loadMatrixAndSplitFromFile(
		   			"en2gr/encoder.rnn_layers.0.weight_ih_l0_reverse.bin" ,
					ConstVecOppX , LSTM_LEVELS);
		   	ConstVecOppX.clear();
	   	}
    }

}

void Model::loadEncoderBiases(){
	auto mod = F_->getParent();
    std::vector<Constant *> ConstVecHb, ConstVecXb;
	std::vector<Constant *> ConstVecOppHb, ConstVecOppXb;
    for (uint j = 0 ; j < ENCODER_LSTMS_NUM ; ++j)
    {
		for (uint i = 0 ; i < LSTM_LEVELS ; ++i)
		{
			std::string const_name_h_b = "encoder_lstm"+ std::to_string(j) +"_b" + files_indices[i]+"11";
			std::string const_name_x_b = "encoder_lstm"+ std::to_string(j) +"_b" + files_indices[i]+"21";
			Constant *ConstHb = mod->getConstantByName(const_name_h_b);
			Constant *ConstXb = mod->getConstantByName(const_name_x_b);
			ConstVecHb.push_back(ConstHb);
			ConstVecXb.push_back(ConstXb);
			if (j == 0 ){
				std::string const_name_opp_h_b = "encoder_opp_lstm_b" + files_indices[i] +"11";
				std::string const_name_opp_x_b = "encoder_opp_lstm_b" + files_indices[i] +"21";
				Constant *ConstOppHb = mod->getConstantByName(const_name_opp_h_b);
				Constant *ConstOppXb = mod->getConstantByName(const_name_opp_x_b);
				ConstVecOppHb.push_back(ConstOppHb);
				ConstVecOppXb.push_back(ConstOppXb);

			}
		}
	   	loadMatrixAndSplitFromFile(
	   			"en2gr/encoder.rnn_layers."+ std::to_string(j) +".bias_hh_l0.bin" ,
				ConstVecHb , LSTM_LEVELS);
	   	ConstVecHb.clear();
	   	loadMatrixAndSplitFromFile(
	   			"en2gr/encoder.rnn_layers."+ std::to_string(j) +".bias_ih_l0.bin" ,
				ConstVecXb , LSTM_LEVELS);
	   	ConstVecXb.clear();
	   	if (j == 0){
	   		loadMatrixAndSplitFromFile(
		   			"en2gr/encoder.rnn_layers.0.bias_hh_l0_reverse.bin" ,
					ConstVecOppHb , LSTM_LEVELS);
		   	ConstVecOppHb.clear();
		   	loadMatrixAndSplitFromFile(
		   			"en2gr/encoder.rnn_layers.0.bias_ih_l0_reverse.bin" ,
					ConstVecOppXb , LSTM_LEVELS);
		   	ConstVecOppXb.clear();
	   	}
    }

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
                                             {input_, output_});
    }
    loadEncoderWieghts();
    loadEncoderBiases();
    EE_.compile(CompilationMode::Infer, F_);
}
