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
	std::printf("*** loadBeamDecoder ***\n\n");
	auto &mod = EE_.getModule();

	std::vector<Node *> dec_seq;
	std::vector<NodeValue> dec_seq_embedding;
	std::vector<NodeValue> attentionOut;

	for (uint i=0 ; i < max_length ; i++){
		Placeholder *hiddenInit = mod.createPlaceholder(ElemKind::Int64ITy,
				{batchSize_ * beam_size}, "decoder."+std::to_string(i)+".hiddenInit", false);
		auto *inputTensor = (*context.getPlaceholderBindings()).allocate(hiddenInit);
		for (size_t j = 0; j < batchSize_ * beam_size; j++) {
			inputTensor->getHandle<int64_t>().at({j}) = BOS;
		}
		Node *seqInit = hiddenInit;
		dec_seq.push_back(seqInit);

		Placeholder *hiddenInitEmbed = mod.createPlaceholder(ElemKind::FloatTy,
				{batchSize_ * beam_size, EMBEDDING_SIZE}, "decoder."+std::to_string(i)+".hiddenInitEmbed", false);
		(*context.getPlaceholderBindings()).allocate(hiddenInitEmbed)->zero();
		Node *seqInitEmbed = hiddenInitEmbed;
		dec_seq_embedding.push_back(seqInitEmbed);

		Placeholder *attentionInint = mod.createPlaceholder(ElemKind::FloatTy,
				{batchSize_ * beam_size, HIDDEN_SIZE}, "attentionInit."+std::to_string(i)+".hiddenInitEmbed", false);
		(*context.getPlaceholderBindings()).allocate(attentionInint)->zero();
		Node *attentionInit_ = attentionInint;
		NodeValue temp(attentionInit_);
		attentionOut.push_back(temp);
	}

	Placeholder *classifier_w = mod.createPlaceholder(ElemKind::FloatTy,
			{EMBEDDING_SIZE, tok_.index2word_.size()+4}, "decoder.classifier_w", false);
	common_->loadMatrixAndTransposeFromFile("en2gr/decoder.classifier.classifier.weight.bin", *(*context.getPlaceholderBindings()).allocate(classifier_w));

	Placeholder *classifier_b = mod.createPlaceholder(ElemKind::FloatTy,
				{tok_.index2word_.size()+4}, "decoder.classifier_w", false);
	common_->loadMatrixFromFile("en2gr/decoder.classifier.classifier.bias.bin", *(*context.getPlaceholderBindings()).allocate(classifier_b));

	Placeholder *S = mod.createPlaceholder(ElemKind::Int64ITy, {batchSize_ * beam_size, 1}, "S", false);

	Placeholder *scoresPH = mod.createPlaceholder(ElemKind::FloatTy, {beam_size} , "scores", false);
	(*context.getPlaceholderBindings()).allocate(scoresPH)->zero();
	Node *scores = scoresPH;

	Tensor newScoreMaskT(ElemKind::FloatTy, {1, beam_size * beam_size});
	for (uint j = 0; j < beam_size*beam_size; j++) {
		if (j < beam_size){
			newScoreMaskT.getHandle<float_t>().at({0,j}) = 0;
		}
		else{
			newScoreMaskT.getHandle<float_t>().at({0,j}) = -1000000.f;
		}
	}
	Constant *newScoreMaskC = mod.createConstant("newScoresMask", newScoreMaskT);
	Node *newScoresMask = newScoreMaskC;

	// TODO: for sourcebeam feature - for future.
//	Placeholder *beamSizeNode = mod.createPlaceholder(ElemKind::FloatTy,
//			{1, batchSize_ * beam_size}, "decoder.beamSizeNode", false);
//	auto *beamTensor = (*context.getPlaceholderBindings()).allocate(beamSizeNode);

	std::vector<Node *> hiddenOut0;

	std::vector<std::vector<NodeValue>> hidenOutputs0Vec;
	std::vector<std::vector<NodeValue>> hidenOutputs1Vec;
	std::vector<std::vector<NodeValue>> hidenOutputs2Vec;
	std::vector<std::vector<NodeValue>> hidenOutputs3Vec;

	std::vector<Node *> output_vec;
	std::vector<std::vector<Node *>> hidden;
	common_->init_hidden(hidden , max_length , dec_seq[0]);

	Node *activeScores;

	for (uint word_inx=0 ; word_inx < max_length; word_inx++){
		Node *decoderEmbedded =
		   F_->createGather("decoder."+std::to_string(word_inx)+".embedding", embedding_tok_, dec_seq[word_inx]);
		NodeValue decoderEmbeddedNV(decoderEmbedded);

		dec_seq_embedding[word_inx] = decoderEmbeddedNV;
		std::vector<NodeValue> lstm0input;
		lstm0input.push_back(dec_seq_embedding[word_inx]);

		// --------------- lstm 0 ---------------
		decoderLayer0->createDecoderInferPytorchLSTM(*context.getPlaceholderBindings(), "decoder.lstm.0."+std::to_string(word_inx), lstm0input ,
				batchSize_*beam_size, HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs0Vec , hidden[0] , word_inx);

		hidden[0][1] = hidenOutputs0Vec[0][1].getNode();
		hidden[0][2] = hidenOutputs0Vec[0][2].getNode();

		std::vector<NodeValue> currentOut;

		// ------- attention -------
		attention = m_attention->loadAttention(hidenOutputs0Vec[0][0].getNode());
		attentionOut[word_inx] = attention.attention_out;
		currentOut.push_back(F_->createConcat("decoder."+std::to_string(word_inx)+".concat",
				{hidenOutputs0Vec[0][0], attention.attention_out},1));

		// --------------- lstm 1 ---------------
		decoderLayer1->createDecoderInferPytorchLSTM(*context.getPlaceholderBindings(), "decoder.lstm.1."+std::to_string(word_inx), currentOut ,
				batchSize_ * beam_size, HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs1Vec, hidden[1] , word_inx);
		currentOut.clear();

		hidden[1][1] = hidenOutputs1Vec[0][1].getNode();
		hidden[1][2] = hidenOutputs1Vec[0][2].getNode();

		// concat 1  - concat lstm1 with attentionout
		currentOut.push_back(F_->createConcat("decoder1."+std::to_string(word_inx)+".concat",
				{hidenOutputs1Vec[0][0], attention.attention_out},1));


		// --------------- lstm 2 ---------------
		decoderLayer2->createDecoderInferPytorchLSTM(*context.getPlaceholderBindings(), "decoder.lstm.2."+std::to_string(word_inx), currentOut ,
				batchSize_*beam_size, HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs2Vec, hidden[2] , word_inx);

		hidden[2][1] = hidenOutputs2Vec[0][1].getNode();
		hidden[2][2] = hidenOutputs2Vec[0][2].getNode();

		// ------- add residual 0 - lstm2 + attention-out -------
		std::vector<NodeValue> addResidual1;
		addResidual1.push_back(F_->createAdd("add.residual1",
				hidenOutputs1Vec[0][0] , hidenOutputs2Vec[0][0]));
		currentOut.clear();

		// ------- concat 2 - concat addResidual1 with attentionout -------
		currentOut.push_back(F_->createConcat("decoder2."+std::to_string(word_inx)+".concat",
				{addResidual1[0], attention.attention_out},1));

		// --------------- lstm 3 ---------------
		decoderLayer3->createDecoderInferPytorchLSTM(*context.getPlaceholderBindings(), "decoder.lstm.3."+std::to_string(word_inx), currentOut ,
				batchSize_*beam_size,HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs3Vec, hidden[3] , word_inx);

		hidden[3][1] = hidenOutputs3Vec[0][1].getNode();
		hidden[3][2] = hidenOutputs3Vec[0][2].getNode();

		// ------- add residual 1 - LSTM2 + LSTM3 -------
		std::vector<Node *> addResidual2;
		addResidual2.push_back(F_->createAdd("add.residual2.", addResidual1[0] , hidenOutputs3Vec[0][0]));

		hidenOutputs0Vec.clear();
		hidenOutputs1Vec.clear();
		hidenOutputs2Vec.clear();
		hidenOutputs3Vec.clear();
		hiddenOut0.clear();
		addResidual1.clear();

		// classifier
		auto *ClasiffierFC = F_->createFullyConnected("decoder.classifier.fc", addResidual2[0] , classifier_w , classifier_b);
		addResidual2.clear();

		// softmax
		Node *DecSM = F_->createSoftMax("decoder_t.softmax", ClasiffierFC, S);
		Node *logSoftmax = F_->createLog("decoder_t.logsoftmax", DecSM);
		// topk
		auto *topK = F_->createTopK("decoder_t.topK_", logSoftmax, beam_size);
		Node *words_indexs = F_->createReshape("decoder_t.words_indexs", topK->getIndices(), {1 , beam_size , beam_size});
		auto *logprobs = F_->createExpandDims("decoder_t.logprobs.reshape" ,topK->getValues(), {0});
		activeScores = F_->createReshape("decoder_t.active.score",scores ,{beam_size ,1});
		activeScores = F_->createReshape("decoder_t.active.score.active",activeScores ,{1, beam_size ,1});
		activeScores = F_->createBroadcast("decoder_t.active.score.broadcast", activeScores , {1 , beam_size , beam_size} , 0);
		Node *newScores = F_->createAdd("decoder_t.new.score", activeScores , logprobs);
		newScores = F_->createReshape("decoder_t.new.score.reshape" , newScores	, {1, beam_size * beam_size});
		if (word_inx == 0){
			newScores = F_->createAdd("decoder_t.index0.add.new_scores", newScores , newScoresMask);
		}
		auto *topK1 = F_->createTopK("decoder_t.topK1_", newScores, beam_size);
		// TODO: sourcebeam for future features
		// auto *sourceBeam = F_->createDiv("decoder_t.source.beam", topK1->getIndices(), (Node *)beamSizeNode);
		auto *bestScores = F_->createGather("decoder_t.gather.best.score",
				F_->createReshape("decoder_t.bestscore.dummy.reshape", newScores , beam_size*beam_size),
				F_->createReshape("decoder_t.bestscore.dummy1.reshape" , topK1->getIndices(), {beam_size}));
		scores = bestScores;
		Node *words = F_->createReshape("decoder_t.words.reshape" , words_indexs
				, {1, beam_size * beam_size});
		words = F_->createGather("decoder_t.gather.words" ,
				F_->createReshape("decoder_t.dummy.reshape" , words, {beam_size*beam_size}) ,
				F_->createReshape("decoder_t.dummy1.reshape" , topK1->getIndices(),{beam_size}));
		if (word_inx+1 < max_length){
			dec_seq[word_inx+1] = words;
		}
		output_vec.push_back(
				F_->createReshape("decoder_t.final.reshape" , words , {1, beam_size}));
	}
	std::vector<NodeValue> decoderOutput;
	for (uint i=0 ; i<output_vec.size() ; i++){
		NodeValue tmpNodeValue(output_vec[i]);
		decoderOutput.push_back(output_vec[i]);
	}
	Node *outputCandidents = F_->createConcat("decoder.output", decoderOutput, 0);
	auto *save = F_->createSave("decoder.output.candidents" , outputCandidents);
	output_ = save->getPlaceholder();
	(*context.getPlaceholderBindings()).allocate(output_);
	auto *save1 = F_->createSave("decoder.output.scores" , scores);
	scores_ = save1->getPlaceholder();
	(*context.getPlaceholderBindings()).allocate(scores_);
}

void Model::loadGreadyDecoder(){
	std::printf("*** loadGreadyDecoder ***\n\n");
	auto &mod = EE_.getModule();

	std::vector<Node *> dec_seq;
	std::vector<NodeValue> dec_seq_embedding;
	std::vector<NodeValue> attentionOut;

	for (uint i=0 ; i < max_length ; i++){
		Placeholder *hiddenInit = mod.createPlaceholder(ElemKind::Int64ITy,
				{batchSize_}, "decoder."+std::to_string(i)+".hiddenInit", false);
		auto *inputTensor = (*context.getPlaceholderBindings()).allocate(hiddenInit);
		for (size_t j = 0; j < batchSize_; j++) {
			inputTensor->getHandle<int64_t>().at({j}) = BOS;
		}
		Node *seqInit = hiddenInit;
		dec_seq.push_back(seqInit);

		Placeholder *hiddenInitEmbed = mod.createPlaceholder(ElemKind::FloatTy,
				{batchSize_, EMBEDDING_SIZE}, "decoder."+std::to_string(i)+".hiddenInitEmbed", false);
		(*context.getPlaceholderBindings()).allocate(hiddenInitEmbed)->zero();
		Node *seqInitEmbed = hiddenInitEmbed;
		dec_seq_embedding.push_back(seqInitEmbed);

		Placeholder *attentionInint = mod.createPlaceholder(ElemKind::FloatTy,
				{batchSize_, HIDDEN_SIZE}, "attentionInit."+std::to_string(i)+".hiddenInitEmbed", false);
		(*context.getPlaceholderBindings()).allocate(attentionInint)->zero();
		Node *attentionInit_ = attentionInint;
		NodeValue temp(attentionInit_);
		attentionOut.push_back(temp);
	}


	Placeholder *classifier_w = mod.createPlaceholder(ElemKind::FloatTy,
			{EMBEDDING_SIZE, tok_.index2word_.size()+4}, "decoder.classifier_w", false);
	common_->loadMatrixAndTransposeFromFile("en2gr/decoder.classifier.classifier.weight.bin", *(*context.getPlaceholderBindings()).allocate(classifier_w));

	Placeholder *classifier_b = mod.createPlaceholder(ElemKind::FloatTy,
				{tok_.index2word_.size()+4}, "decoder.classifier_w", false);
	common_->loadMatrixFromFile("en2gr/decoder.classifier.classifier.bias.bin", *(*context.getPlaceholderBindings()).allocate(classifier_b));

	Placeholder *S = mod.createPlaceholder(ElemKind::Int64ITy, {batchSize_, 1}, "S", false);

	std::vector<Node *> hiddenOut0;

	std::vector<std::vector<NodeValue>> hidenOutputs0Vec;
	std::vector<std::vector<NodeValue>> hidenOutputs1Vec;
	std::vector<std::vector<NodeValue>> hidenOutputs2Vec;
	std::vector<std::vector<NodeValue>> hidenOutputs3Vec;

	std::vector<Node *> output_vec;
	std::vector<std::vector<Node *>> hidden;
	common_->init_hidden(hidden , max_length , dec_seq[0]);

	for (uint word_inx=0 ; word_inx < max_length; word_inx++){
		Node *decoderEmbedded =
		   F_->createGather("decoder."+std::to_string(word_inx)+".embedding", embedding_tok_, dec_seq[word_inx]);
		NodeValue decoderEmbeddedNV(decoderEmbedded);

		dec_seq_embedding[word_inx] = decoderEmbeddedNV;
		std::vector<NodeValue> lstm0input;
		lstm0input.push_back(dec_seq_embedding[word_inx]);

		// --------------- lstm 0 ---------------
		decoderLayer0->createDecoderInferPytorchLSTM(*context.getPlaceholderBindings(), "decoder.lstm.0."+std::to_string(word_inx), lstm0input ,
				batchSize_, HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs0Vec , hidden[0] , word_inx);

		hidden[0][1] = hidenOutputs0Vec[0][1].getNode();
		hidden[0][2] = hidenOutputs0Vec[0][2].getNode();

		std::vector<NodeValue> currentOut;

		// ------- attention -------
		attention = m_attention->loadAttention(hidenOutputs0Vec[0][0].getNode());
		attentionOut[word_inx] = attention.attention_out;

		currentOut.push_back(F_->createConcat("decoder."+std::to_string(word_inx)+".concat", {hidenOutputs0Vec[0][0], attention.attention_out},1));

		// --------------- lstm 1 ---------------
		decoderLayer1->createDecoderInferPytorchLSTM(*context.getPlaceholderBindings(), "decoder.lstm.1."+std::to_string(word_inx), currentOut ,
				batchSize_, HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs1Vec , hidden[1] , word_inx);
		currentOut.clear();

		hidden[1][1] = hidenOutputs1Vec[0][1].getNode();
		hidden[1][2] = hidenOutputs1Vec[0][2].getNode();

		currentOut.push_back(F_->createConcat("decoder1."+std::to_string(word_inx)+".concat", {hidenOutputs1Vec[0][0], attention.attention_out},1));

		// --------------- lstm 2 ---------------
		decoderLayer2->createDecoderInferPytorchLSTM(*context.getPlaceholderBindings(), "decoder.lstm.2."+std::to_string(word_inx), currentOut ,
				batchSize_, HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs2Vec, hidden[2] , word_inx);

		hidden[2][1] = hidenOutputs2Vec[0][1].getNode();
		hidden[2][2] = hidenOutputs2Vec[0][2].getNode();

		// ------- add residual 0 - lstm2 + attention-out -------
		std::vector<NodeValue> addResidual1;
		addResidual1.push_back(F_->createAdd("add."+std::to_string(word_inx)+".residual1", hidenOutputs1Vec[0][0] , hidenOutputs2Vec[0][0]));
		currentOut.clear();

		// ------- concat 2 - concat addResidual1 with attentionout -------
		currentOut.push_back(F_->createConcat("decoder2."+std::to_string(word_inx)+".concat", {addResidual1[0], attention.attention_out},1));

		// --------------- lstm 3 ---------------
		decoderLayer3->createDecoderInferPytorchLSTM(*context.getPlaceholderBindings(), "decoder.lstm.3."+std::to_string(word_inx), currentOut ,
				batchSize_, HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs3Vec, hidden[3] , word_inx);

		hidden[3][1] = hidenOutputs3Vec[0][1].getNode();
		hidden[3][2] = hidenOutputs3Vec[0][2].getNode();

		// ------- add residual 1 - LSTM2 + LSTM3 -------
		std::vector<Node *> addResidual2;
		addResidual2.push_back(F_->createAdd("add."+std::to_string(word_inx)+".residual2.", addResidual1[0] , hidenOutputs3Vec[0][0]));

		hidenOutputs0Vec.clear();
		hidenOutputs1Vec.clear();
		hidenOutputs2Vec.clear();
		hidenOutputs3Vec.clear();
		hiddenOut0.clear();
		addResidual1.clear();

		// ------- classifier -------
		auto *ClasiffierFC = F_->createFullyConnected("decoder.classifier.fc."+std::to_string(word_inx), addResidual2[0] , classifier_w , classifier_b);
		addResidual2.clear();

		// ------- softmax -------
		Node *DecSM = F_->createSoftMax("decoder."+std::to_string(word_inx)+".softmax", ClasiffierFC, S);
		Node *logSoftmax = F_->createLog("decoder_t."+std::to_string(word_inx)+".logsoftmax", DecSM);
		// topk
		auto *topK = F_->createTopK("decoder."+std::to_string(word_inx)+".topK", logSoftmax, 1);

		Node *lastWordIdx = F_->createReshape("decoder."+std::to_string(word_inx)+".reshape",
	    		topK->getIndices(), {batchSize_});

		if (word_inx+1 < max_length){
			dec_seq[word_inx+1] = lastWordIdx;
		}
		output_vec.push_back(lastWordIdx);
	}

	std::vector<NodeValue> lstmOutput;
	for (uint i=0 ; i<output_vec.size() ; i++){
		//debug_size_print(output_vec[i]);
		NodeValue tmpNodeValue(output_vec[i]);
		lstmOutput.push_back(output_vec[i]);
	}
	Node *output = F_->createConcat("decoder.output", lstmOutput, 0);
	Node *reshape = F_->createReshape("decoder.output.reshape", output, {max_length, batchSize_});
	auto *save = F_->createSave("decoder.output", reshape);
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

Placeholder *Model::loadEmbedding(llvm::StringRef langPrefix, size_t langSize) {
	  auto &mod = EE_.getModule();
	  auto *result =
			  mod.createPlaceholder(ElemKind::FloatTy, {langSize, EMBEDDING_SIZE},
					  "embedding." + langPrefix.str(), false);
	  auto test = (*context.getPlaceholderBindings()).allocate(result);
	  common_->loadMatrixFromFile("en2gr/" + langPrefix.str() + "_embedding.bin",*test);
	  return result;
}

void Model::loadDecoderWieghts(){
	auto mod = F_->getParent();
    std::vector<Constant *> ConstVecH, ConstVecX;
    for (uint j = 0 ; j < DECODER_LSTMS_NUM ; ++j)
    {
    	for (uint i = 0 ; i < LSTM_LEVELS ; ++i)
    	{
    		std::string const_name_h = "decoder_lstm_"+ std::to_string(j) +"_0_" + files_gates[i]+"_wh1";
    		std::string const_name_x = "decoder_lstm_"+ std::to_string(j) +"_0_" + files_gates[i]+"_wi1";
    		Constant *ConstH = mod->getConstantByName(const_name_h);
    		Constant *ConstX = mod->getConstantByName(const_name_x);
    		ConstVecH.push_back(ConstH);
    		ConstVecX.push_back(ConstX);
    	}
    	if (j == 0){
    		common_->loadMatrixAndSplitAndTransposeFromFile(
    				"en2gr/decoder.att_rnn.rnn.weight_hh_l0.bin" ,
					ConstVecH , LSTM_LEVELS);

    		common_->loadMatrixAndSplitAndTransposeFromFile(
    				"en2gr/decoder.att_rnn.rnn.weight_ih_l0.bin" ,
					ConstVecX , LSTM_LEVELS);
    		ConstVecH.clear();
    		ConstVecX.clear();
    	}
    	else
    	{
    		common_->loadMatrixAndSplitAndTransposeFromFile(
    				"en2gr/decoder.rnn_layers."+ std::to_string(j-1) +".weight_hh_l0.bin" ,
					ConstVecH , LSTM_LEVELS);
    		common_->loadMatrixAndSplitAndTransposeFromFile(
    				"en2gr/decoder.rnn_layers."+ std::to_string(j-1) +".weight_ih_l0.bin" ,
					ConstVecX , LSTM_LEVELS);
    		ConstVecH.clear();
    		ConstVecX.clear();
    	}
    }
}

void Model::loadDecoderBiases(){
	auto mod = F_->getParent();
    std::vector<Constant *> ConstVecH, ConstVecX;
    for (uint j = 0 ; j < ENCODER_LSTMS_NUM ; ++j)
    {
    	for (uint i = 0 ; i < LSTM_LEVELS ; ++i)
    	{
    		std::string const_name_h = "decoder_lstm_"+ std::to_string(j) +"_0_" + files_gates[i]+"_bh1";
    		std::string const_name_x = "decoder_lstm_"+ std::to_string(j) +"_0_" + files_gates[i]+"_bi1";
    		Constant *ConstH = mod->getConstantByName(const_name_h);
    		Constant *ConstX = mod->getConstantByName(const_name_x);
    		ConstVecH.push_back(ConstH);
    		ConstVecX.push_back(ConstX);
    	}
    	if (j == 0){
    		common_->loadMatrixAndSplitFromFile(
    				"en2gr/decoder.att_rnn.rnn.bias_hh_l0.bin" ,
					ConstVecH , LSTM_LEVELS);
    		common_->loadMatrixAndSplitFromFile(
    				"en2gr/decoder.att_rnn.rnn.bias_ih_l0.bin" ,
					ConstVecX , LSTM_LEVELS);
    		ConstVecH.clear();
    		ConstVecX.clear();
    	}
    	else
    	{
    		common_->loadMatrixAndSplitFromFile(
    				"en2gr/decoder.rnn_layers."+ std::to_string(j-1) +".bias_hh_l0.bin" ,
					ConstVecH , LSTM_LEVELS);
    		common_->loadMatrixAndSplitFromFile(
    				"en2gr/decoder.rnn_layers."+ std::to_string(j-1) +".bias_ih_l0.bin" ,
					ConstVecX , LSTM_LEVELS);
    		ConstVecH.clear();
    		ConstVecX.clear();
		}
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
    loadDecoderWieghts();
    loadDecoderBiases();

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
    EE_.compile(F_, cctx);
//    FunctionList debug = EE_.getModule().getFunctions();
//    for (auto F : debug){
//    	F->dump();
//    }
}
