/*
 * decoder.cpp
 *
 *  Created on: Nov 26, 2019
 *      Author: ezimmer9
 */
#include "decoder.h"

void Decoder::init() {
	auto &mod = m_model->EE_.getModule();
	decoderLayer0 = std::make_shared<LSTM>(m_model);
	decoderLayer1 = std::make_shared<LSTM>(m_model);
	decoderLayer2 = std::make_shared<LSTM>(m_model);
	decoderLayer3 = std::make_shared<LSTM>(m_model);
	m_attention = std::make_shared<Attention>(m_batch_size , m_beam_size , m_model);

	classifier_w = mod.createPlaceholder(ElemKind::FloatTy,{EMBEDDING_SIZE, m_model->tok_.index2word_.size()+4}, "decoder.classifier_w", false);
	m_model->common_->loadMatrixAndTransposeFromFile("en2gr/decoder.classifier.classifier.weight.bin",*(*m_model->context.getPlaceholderBindings()).allocate(classifier_w));
	classifier_b = mod.createPlaceholder(ElemKind::FloatTy, {m_model->tok_.index2word_.size()+4}, "decoder.classifier_w", false);
	m_model->common_->loadMatrixFromFile("en2gr/decoder.classifier.classifier.bias.bin", *(*m_model->context.getPlaceholderBindings()).allocate(classifier_b));
}

BeamDecoderOutput Decoder::loadBeamDecoder(){
	std::printf("*** loadBeamDecoder ***\n\n");
	auto &mod = m_model->EE_.getModule();

	std::vector<Node *> dec_seq;
	std::vector<NodeValue> dec_seq_embedding;
	std::vector<NodeValue> attentionOut;

	for (uint i=0 ; i < m_model->max_length ; i++){
		Placeholder *hiddenInit = mod.createPlaceholder(ElemKind::Int64ITy,
				{get_batch_size() * get_beam_size()}, "decoder."+std::to_string(i)+".hiddenInit", false);
		auto *inputTensor = (*m_model->context.getPlaceholderBindings()).allocate(hiddenInit);
		for (size_t j = 0; j < get_batch_size() * get_beam_size(); j++) {
			inputTensor->getHandle<int64_t>().at({j}) = BOS;
		}
		Node *seqInit = hiddenInit;
		dec_seq.push_back(seqInit);

		Placeholder *hiddenInitEmbed = mod.createPlaceholder(ElemKind::FloatTy,
				{get_batch_size() * get_beam_size(), EMBEDDING_SIZE}, "decoder."+std::to_string(i)+".hiddenInitEmbed", false);
		(*m_model->context.getPlaceholderBindings()).allocate(hiddenInitEmbed)->zero();
		Node *seqInitEmbed = hiddenInitEmbed;
		dec_seq_embedding.push_back(seqInitEmbed);

		Placeholder *attentionInint = mod.createPlaceholder(ElemKind::FloatTy,
				{get_batch_size() * get_beam_size(), HIDDEN_SIZE}, "attentionInit."+std::to_string(i)+".hiddenInitEmbed", false);
		(*m_model->context.getPlaceholderBindings()).allocate(attentionInint)->zero();
		Node *attentionInit_ = attentionInint;
		NodeValue temp(attentionInit_);
		attentionOut.push_back(temp);
	}


	Placeholder *S = mod.createPlaceholder(ElemKind::Int64ITy, {get_batch_size() * get_beam_size(), 1}, "S", false);

	Placeholder *scoresPH = mod.createPlaceholder(ElemKind::FloatTy, {get_beam_size()} , "scores", false);
	(*m_model->context.getPlaceholderBindings()).allocate(scoresPH)->zero();
	Node *scores = scoresPH;

	Tensor newScoreMaskT(ElemKind::FloatTy, {1, get_beam_size() * get_beam_size()});
	for (uint j = 0; j < get_beam_size()*get_beam_size(); j++) {
		if (j < get_beam_size()){
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
//			{1, get_batch_size() * get_beam_size()}, "decoder.beamSizeNode", false);
//	auto *beamTensor = (*context.getPlaceholderBindings()).allocate(beamSizeNode);

	std::vector<Node *> hiddenOut0;

	std::vector<std::vector<NodeValue>> hidenOutputs0Vec;
	std::vector<std::vector<NodeValue>> hidenOutputs1Vec;
	std::vector<std::vector<NodeValue>> hidenOutputs2Vec;
	std::vector<std::vector<NodeValue>> hidenOutputs3Vec;

	std::vector<Node *> output_vec;
	std::vector<std::vector<Node *>> hidden;
	m_model->common_->init_hidden(hidden , m_model->max_length , dec_seq[0]);

	Node *activeScores;

	for (uint word_inx=0 ; word_inx < m_model->max_length; word_inx++){
		Node *decoderEmbedded =
				m_model->F_->createGather("decoder."+std::to_string(word_inx)+".embedding", m_model->get_embedding_tok(), dec_seq[word_inx]);
		NodeValue decoderEmbeddedNV(decoderEmbedded);

		dec_seq_embedding[word_inx] = decoderEmbeddedNV;
		std::vector<NodeValue> lstm0input;
		lstm0input.push_back(dec_seq_embedding[word_inx]);

		// --------------- lstm 0 ---------------
		decoderLayer0->createDecoderInferPytorchLSTM(*m_model->context.getPlaceholderBindings(), "decoder.lstm.0."+std::to_string(word_inx), lstm0input ,
				get_batch_size()*get_beam_size(), HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs0Vec , hidden[0] , word_inx);

		hidden[0][1] = hidenOutputs0Vec[0][1].getNode();
		hidden[0][2] = hidenOutputs0Vec[0][2].getNode();

		std::vector<NodeValue> currentOut;

		// ------- attention -------
		attention = m_attention->loadAttention(hidenOutputs0Vec[0][0].getNode());
		attentionOut[word_inx] = attention.attention_out;
		currentOut.push_back(m_model->F_->createConcat("decoder."+std::to_string(word_inx)+".concat",
				{hidenOutputs0Vec[0][0], attention.attention_out},1));

		// --------------- lstm 1 ---------------
		decoderLayer1->createDecoderInferPytorchLSTM(*m_model->context.getPlaceholderBindings(), "decoder.lstm.1."+std::to_string(word_inx), currentOut ,
				get_batch_size() * get_beam_size(), HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs1Vec, hidden[1] , word_inx);
		currentOut.clear();

		hidden[1][1] = hidenOutputs1Vec[0][1].getNode();
		hidden[1][2] = hidenOutputs1Vec[0][2].getNode();

		// concat 1  - concat lstm1 with attentionout
		currentOut.push_back(m_model->F_->createConcat("decoder1."+std::to_string(word_inx)+".concat",
				{hidenOutputs1Vec[0][0], attention.attention_out},1));


		// --------------- lstm 2 ---------------
		decoderLayer2->createDecoderInferPytorchLSTM(*m_model->context.getPlaceholderBindings(), "decoder.lstm.2."+std::to_string(word_inx), currentOut ,
				get_batch_size()*get_beam_size(), HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs2Vec, hidden[2] , word_inx);

		hidden[2][1] = hidenOutputs2Vec[0][1].getNode();
		hidden[2][2] = hidenOutputs2Vec[0][2].getNode();

		// ------- add residual 0 - lstm2 + attention-out -------
		std::vector<NodeValue> addResidual1;
		addResidual1.push_back(m_model->F_->createAdd("add.residual1",
				hidenOutputs1Vec[0][0] , hidenOutputs2Vec[0][0]));
		currentOut.clear();

		// ------- concat 2 - concat addResidual1 with attentionout -------
		currentOut.push_back(m_model->F_->createConcat("decoder2."+std::to_string(word_inx)+".concat",
				{addResidual1[0], attention.attention_out},1));

		// --------------- lstm 3 ---------------
		decoderLayer3->createDecoderInferPytorchLSTM(*m_model->context.getPlaceholderBindings(), "decoder.lstm.3."+std::to_string(word_inx), currentOut ,
				get_batch_size()*get_beam_size(),HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs3Vec, hidden[3] , word_inx);

		hidden[3][1] = hidenOutputs3Vec[0][1].getNode();
		hidden[3][2] = hidenOutputs3Vec[0][2].getNode();

		// ------- add residual 1 - LSTM2 + LSTM3 -------
		std::vector<Node *> addResidual2;
		addResidual2.push_back(m_model->F_->createAdd("add.residual2.", addResidual1[0] , hidenOutputs3Vec[0][0]));

		hidenOutputs0Vec.clear();
		hidenOutputs1Vec.clear();
		hidenOutputs2Vec.clear();
		hidenOutputs3Vec.clear();
		hiddenOut0.clear();
		addResidual1.clear();

		// classifier
		auto *ClasiffierFC = m_model->F_->createFullyConnected("decoder.classifier.fc", addResidual2[0] , classifier_w , classifier_b);
		addResidual2.clear();

		// softmax
		Node *DecSM = m_model->F_->createSoftMax("decoder_t.softmax", ClasiffierFC, S);
		Node *logSoftmax = m_model->F_->createLog("decoder_t.logsoftmax", DecSM);
		// topk
		auto *topK = m_model->F_->createTopK("decoder_t.topK_", logSoftmax, get_beam_size());
		Node *words_indexs = m_model->F_->createReshape("decoder_t.words_indexs", topK->getIndices(), {1 , get_beam_size() , get_beam_size()});
		auto *logprobs = m_model->F_->createExpandDims("decoder_t.logprobs.reshape" ,topK->getValues(), {0});
		activeScores = m_model->F_->createReshape("decoder_t.active.score",scores ,{get_beam_size() ,1});
		activeScores = m_model->F_->createReshape("decoder_t.active.score.active",activeScores ,{1, get_beam_size() ,1});
		activeScores = m_model->F_->createBroadcast("decoder_t.active.score.broadcast", activeScores , {1 , get_beam_size() , get_beam_size()} , 0);
		Node *newScores = m_model->F_->createAdd("decoder_t.new.score", activeScores , logprobs);
		newScores = m_model->F_->createReshape("decoder_t.new.score.reshape" , newScores	, {1, get_beam_size() * get_beam_size()});
		if (word_inx == 0){
			newScores = m_model->F_->createAdd("decoder_t.index0.add.new_scores", newScores , newScoresMask);
		}
		auto *topK1 = m_model->F_->createTopK("decoder_t.topK1_", newScores, get_beam_size());
		// TODO: sourcebeam for future features
		// auto *sourceBeam = F_->createDiv("decoder_t.source.beam", topK1->getIndices(), (Node *)beamSizeNode);
		auto *bestScores = m_model->F_->createGather("decoder_t.gather.best.score",
				m_model->F_->createReshape("decoder_t.bestscore.dummy.reshape", newScores , get_beam_size()*get_beam_size()),
				m_model->F_->createReshape("decoder_t.bestscore.dummy1.reshape" , topK1->getIndices(), {get_beam_size()}));
		scores = bestScores;
		Node *words = m_model->F_->createReshape("decoder_t.words.reshape" , words_indexs
				, {1, get_beam_size() * get_beam_size()});
		words = m_model->F_->createGather("decoder_t.gather.words" ,
				m_model->F_->createReshape("decoder_t.dummy.reshape" , words, {get_beam_size()*get_beam_size()}) ,
				m_model->F_->createReshape("decoder_t.dummy1.reshape" , topK1->getIndices(),{get_beam_size()}));
		if (word_inx+1 < m_model->max_length){
			dec_seq[word_inx+1] = words;
		}
		output_vec.push_back(
				m_model->F_->createReshape("decoder_t.final.reshape" , words , {1, get_beam_size()}));
	}
	std::vector<NodeValue> decoderOutput;
	for (uint i=0 ; i<output_vec.size() ; i++){
		NodeValue tmpNodeValue(output_vec[i]);
		decoderOutput.push_back(output_vec[i]);
	}
	BeamDecoderOutput output;
	Node *outputCandidents = m_model->F_->createConcat("decoder.output", decoderOutput, 0);
	output.candidates = outputCandidents;
	output.scores = scores;
	return output;
}

Node* Decoder::loadGreadyDecoder(){
	std::printf("*** loadGreadyDecoder ***\n\n");
	auto &mod = m_model->EE_.getModule();

	std::vector<Node *> dec_seq;
	std::vector<NodeValue> dec_seq_embedding;
	std::vector<NodeValue> attentionOut;

	for (uint i=0 ; i < m_model->max_length ; i++){
		Placeholder *hiddenInit = mod.createPlaceholder(ElemKind::Int64ITy,
				{get_batch_size()}, "decoder."+std::to_string(i)+".hiddenInit", false);
		auto *inputTensor = (*m_model->context.getPlaceholderBindings()).allocate(hiddenInit);
		for (size_t j = 0; j < get_batch_size(); j++) {
			inputTensor->getHandle<int64_t>().at({j}) = BOS;
		}
		Node *seqInit = hiddenInit;
		dec_seq.push_back(seqInit);

		Placeholder *hiddenInitEmbed = mod.createPlaceholder(ElemKind::FloatTy,
				{get_batch_size(), EMBEDDING_SIZE}, "decoder."+std::to_string(i)+".hiddenInitEmbed", false);
		(*m_model->context.getPlaceholderBindings()).allocate(hiddenInitEmbed)->zero();
		Node *seqInitEmbed = hiddenInitEmbed;
		dec_seq_embedding.push_back(seqInitEmbed);

		Placeholder *attentionInint = mod.createPlaceholder(ElemKind::FloatTy,
				{get_batch_size(), HIDDEN_SIZE}, "attentionInit."+std::to_string(i)+".hiddenInitEmbed", false);
		(*m_model->context.getPlaceholderBindings()).allocate(attentionInint)->zero();
		Node *attentionInit_ = attentionInint;
		NodeValue temp(attentionInit_);
		attentionOut.push_back(temp);
	}

	Placeholder *S = mod.createPlaceholder(ElemKind::Int64ITy, {get_batch_size(), 1}, "S", false);

	std::vector<Node *> hiddenOut0;

	std::vector<std::vector<NodeValue>> hidenOutputs0Vec;
	std::vector<std::vector<NodeValue>> hidenOutputs1Vec;
	std::vector<std::vector<NodeValue>> hidenOutputs2Vec;
	std::vector<std::vector<NodeValue>> hidenOutputs3Vec;

	std::vector<Node *> output_vec;
	std::vector<std::vector<Node *>> hidden;
	m_model->common_->init_hidden(hidden , m_model->max_length , dec_seq[0]);

	for (uint word_inx=0 ; word_inx < m_model->max_length; word_inx++){
		Node *decoderEmbedded =
				m_model->F_->createGather("decoder."+std::to_string(word_inx)+".embedding", m_model->get_embedding_tok(), dec_seq[word_inx]);
		NodeValue decoderEmbeddedNV(decoderEmbedded);

		dec_seq_embedding[word_inx] = decoderEmbeddedNV;
		std::vector<NodeValue> lstm0input;
		lstm0input.push_back(dec_seq_embedding[word_inx]);

		// --------------- lstm 0 ---------------
		decoderLayer0->createDecoderInferPytorchLSTM(*m_model->context.getPlaceholderBindings(), "decoder.lstm.0."+std::to_string(word_inx), lstm0input ,
				get_batch_size(), HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs0Vec , hidden[0] , word_inx);

		hidden[0][1] = hidenOutputs0Vec[0][1].getNode();
		hidden[0][2] = hidenOutputs0Vec[0][2].getNode();

		std::vector<NodeValue> currentOut;

		// ------- attention -------
		attention = m_attention->loadAttention(hidenOutputs0Vec[0][0].getNode());
		attentionOut[word_inx] = attention.attention_out;

		currentOut.push_back(m_model->F_->createConcat("decoder."+std::to_string(word_inx)+".concat", {hidenOutputs0Vec[0][0], attention.attention_out},1));

		// --------------- lstm 1 ---------------
		decoderLayer1->createDecoderInferPytorchLSTM(*m_model->context.getPlaceholderBindings(), "decoder.lstm.1."+std::to_string(word_inx), currentOut ,
				get_batch_size(), HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs1Vec , hidden[1] , word_inx);
		currentOut.clear();

		hidden[1][1] = hidenOutputs1Vec[0][1].getNode();
		hidden[1][2] = hidenOutputs1Vec[0][2].getNode();

		currentOut.push_back(m_model->F_->createConcat("decoder1."+std::to_string(word_inx)+".concat", {hidenOutputs1Vec[0][0], attention.attention_out},1));

		// --------------- lstm 2 ---------------
		decoderLayer2->createDecoderInferPytorchLSTM(*m_model->context.getPlaceholderBindings(), "decoder.lstm.2."+std::to_string(word_inx), currentOut ,
				get_batch_size(), HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs2Vec, hidden[2] , word_inx);

		hidden[2][1] = hidenOutputs2Vec[0][1].getNode();
		hidden[2][2] = hidenOutputs2Vec[0][2].getNode();

		// ------- add residual 0 - lstm2 + attention-out -------
		std::vector<NodeValue> addResidual1;
		addResidual1.push_back(m_model->F_->createAdd("add."+std::to_string(word_inx)+".residual1", hidenOutputs1Vec[0][0] , hidenOutputs2Vec[0][0]));
		currentOut.clear();

		// ------- concat 2 - concat addResidual1 with attentionout -------
		currentOut.push_back(m_model->F_->createConcat("decoder2."+std::to_string(word_inx)+".concat", {addResidual1[0], attention.attention_out},1));

		// --------------- lstm 3 ---------------
		decoderLayer3->createDecoderInferPytorchLSTM(*m_model->context.getPlaceholderBindings(), "decoder.lstm.3."+std::to_string(word_inx), currentOut ,
				get_batch_size(), HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs3Vec, hidden[3] , word_inx);

		hidden[3][1] = hidenOutputs3Vec[0][1].getNode();
		hidden[3][2] = hidenOutputs3Vec[0][2].getNode();

		// ------- add residual 1 - LSTM2 + LSTM3 -------
		std::vector<Node *> addResidual2;
		addResidual2.push_back(m_model->F_->createAdd("add."+std::to_string(word_inx)+".residual2.", addResidual1[0] , hidenOutputs3Vec[0][0]));

		hidenOutputs0Vec.clear();
		hidenOutputs1Vec.clear();
		hidenOutputs2Vec.clear();
		hidenOutputs3Vec.clear();
		hiddenOut0.clear();
		addResidual1.clear();

		// ------- classifier -------
		auto *ClasiffierFC = m_model->F_->createFullyConnected("decoder.classifier.fc."+std::to_string(word_inx), addResidual2[0] , classifier_w , classifier_b);
		addResidual2.clear();

		// ------- softmax -------
		Node *DecSM = m_model->F_->createSoftMax("decoder."+std::to_string(word_inx)+".softmax", ClasiffierFC, S);
		Node *logSoftmax = m_model->F_->createLog("decoder_t."+std::to_string(word_inx)+".logsoftmax", DecSM);
		// topk
		auto *topK = m_model->F_->createTopK("decoder."+std::to_string(word_inx)+".topK", logSoftmax, 1);

		Node *lastWordIdx = m_model->F_->createReshape("decoder."+std::to_string(word_inx)+".reshape",
	    		topK->getIndices(), {get_batch_size()});

		if (word_inx+1 < m_model->max_length){
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
	Node *output = m_model->F_->createConcat("decoder.output", lstmOutput, 0);
	Node *reshape = m_model->F_->createReshape("decoder.output.reshape", output, {m_model->max_length, get_batch_size()});
	return reshape;
}

void Decoder::loadDecoderWieghts(){
	auto mod = m_model->F_->getParent();
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
    		m_model->common_->loadMatrixAndSplitAndTransposeFromFile("en2gr/decoder.att_rnn.rnn.weight_hh_l0.bin" ,
					ConstVecH , LSTM_LEVELS);

    		m_model->common_->loadMatrixAndSplitAndTransposeFromFile("en2gr/decoder.att_rnn.rnn.weight_ih_l0.bin" ,
					ConstVecX , LSTM_LEVELS);
    		ConstVecH.clear();
    		ConstVecX.clear();
    	}
    	else
    	{
    		m_model->common_->loadMatrixAndSplitAndTransposeFromFile("en2gr/decoder.rnn_layers."+ std::to_string(j-1) +".weight_hh_l0.bin" ,
					ConstVecH , LSTM_LEVELS);
    		m_model->common_->loadMatrixAndSplitAndTransposeFromFile("en2gr/decoder.rnn_layers."+ std::to_string(j-1) +".weight_ih_l0.bin" ,
					ConstVecX , LSTM_LEVELS);
    		ConstVecH.clear();
    		ConstVecX.clear();
    	}
    }
}

void Decoder::loadDecoderBiases(){
	auto mod = m_model->F_->getParent();
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
    		m_model->common_->loadMatrixAndSplitFromFile("en2gr/decoder.att_rnn.rnn.bias_hh_l0.bin" ,
					ConstVecH , LSTM_LEVELS);
    		m_model->common_->loadMatrixAndSplitFromFile("en2gr/decoder.att_rnn.rnn.bias_ih_l0.bin" ,
					ConstVecX , LSTM_LEVELS);
    		ConstVecH.clear();
    		ConstVecX.clear();
    	}
    	else
    	{
    		m_model->common_->loadMatrixAndSplitFromFile("en2gr/decoder.rnn_layers."+ std::to_string(j-1) +".bias_hh_l0.bin" ,
					ConstVecH , LSTM_LEVELS);
    		m_model->common_->loadMatrixAndSplitFromFile("en2gr/decoder.rnn_layers."+ std::to_string(j-1) +".bias_ih_l0.bin" ,
					ConstVecX , LSTM_LEVELS);
    		ConstVecH.clear();
    		ConstVecX.clear();
		}
    }
}
