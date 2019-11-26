/*
 * encoder.cpp
 *
 *  Created on: Nov 24, 2019
 *      Author: ezimmer9
 */

#include "encoder.h"

void Encoder::init() {
	encoderLayer0 = std::make_shared<LSTM>(m_model);
	encoderLayer1 = std::make_shared<LSTM>(m_model);
	encoderLayer2 = std::make_shared<LSTM>(m_model);
	encoderLayer3 = std::make_shared<LSTM>(m_model);
	encoderLayerOpp = std::make_shared<LSTM>(m_model);
}



Node* Encoder::loadEncoder(Placeholder *embedding_tok_, Placeholder *input_){
	std::printf("*** loadEncoder ***\n\n");

	Node *inputEmbedded =
	   m_model->F_->createGather("encoder.embedding", embedding_tok_, input_);

	std::vector<NodeValue> hidenOutputs0 , oppHidenOutputs0 , hidenOutputs1 , hidenOutputs2 , hidenOutputs3;
	std::vector<NodeValue> hiddenOut2 , encOut;

	std::vector<NodeValue> enc_seq , opposite_seq;
	std::vector<SliceNode *> enc_seq__;
	m_model->F_->createSplit("encoder.split", inputEmbedded , m_model->max_length, 1 ,{}, enc_seq__);

	for (unsigned word_inx=0 ; word_inx < m_model->max_length; word_inx++){
		Node *reshape = m_model->F_->createReshape("encoder."+ std::to_string(word_inx)+".reshape",
				enc_seq__[word_inx],{get_batch_size(), EMBEDDING_SIZE});
		NodeValue reshapeNV(reshape);
		enc_seq.push_back(reshapeNV);

		Node *oppReshape = m_model->F_->createReshape("opp.encoder."+ std::to_string(word_inx)+".reshape" ,
				enc_seq__[m_model->max_length-1 - word_inx],{get_batch_size(), EMBEDDING_SIZE});
		NodeValue oppReshapeNV(oppReshape);
		opposite_seq.push_back(oppReshapeNV);
	}
	enc_seq__.clear();
	// Bi-Directional LSTM.
	encoderLayer0->createInferPytorchLSTM(*m_model->context.getPlaceholderBindings(), "encoder.0.lstm", enc_seq ,
			get_batch_size(), EMBEDDING_SIZE , HIDDEN_SIZE, hidenOutputs0);
	encoderLayerOpp->createInferPytorchLSTM(*m_model->context.getPlaceholderBindings(), "encoder.opp.lstm", opposite_seq ,
			get_batch_size(), EMBEDDING_SIZE , HIDDEN_SIZE, oppHidenOutputs0);
	std::vector<NodeValue> lstm0Concat;
	for (uint i=0 ; i < hidenOutputs0.size() ; i++){
		Node *concat = m_model->F_->createConcat("encoder."+std::to_string(i)+".concat",
				{hidenOutputs0[i], oppHidenOutputs0[m_model->max_length-1-i]},1);
		NodeValue concatNV(concat);
		lstm0Concat.push_back(concatNV);
	}
	// ---------------- end Bi-Directional LSTM --------------
	// lstm 1
	encoderLayer1->createInferPytorchLSTM(*m_model->context.getPlaceholderBindings(),"encoder.1.lstm",lstm0Concat ,
			get_batch_size(), HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs1);
	// lstm 2
	encoderLayer2->createInferPytorchLSTM(*m_model->context.getPlaceholderBindings(),"encoder.2.lstm",hidenOutputs1 ,
			get_batch_size(), HIDDEN_SIZE  , HIDDEN_SIZE, hidenOutputs2);
	// add 0
	assert(hidenOutputs2.size() == hidenOutputs1.size() && "LSTM outputs doesn't align");
	for (uint i = 0 ; i < hidenOutputs1.size() ; i++){
		Node *add1 = m_model->F_->createAdd("encoder."+std::to_string(i)+".residual.",hidenOutputs2[i], hidenOutputs1[i]);
		NodeValue add1NV(add1);
		hiddenOut2.push_back(add1NV);
	}

	// lstm 3
	encoderLayer3->createInferPytorchLSTM(*m_model->context.getPlaceholderBindings(),"encoder.3.lstm",hiddenOut2 ,
			get_batch_size(), HIDDEN_SIZE  , HIDDEN_SIZE, hidenOutputs3);

	// add 1
	assert(hidenOutputs3.size() == hiddenOut2.size() && "LSTM outputs doesn't align");
	for (uint i = 0 ; i < hidenOutputs2.size() ; i++){
		Node *add2 = m_model->F_->createAdd("encoder."+std::to_string(i)+".residual1.",hiddenOut2[i], hidenOutputs3[i]);
		NodeValue add2NV(add2);
		encOut.push_back(add2NV);
	}

	hidenOutputs0.clear();
	hidenOutputs1.clear();
	hidenOutputs2.clear();
	hidenOutputs3.clear();
	hiddenOut2.clear();

	Node *output = m_model->F_->createConcat("encoder.output", encOut, 0);
	Node *encoderHiddenOutput = m_model->F_->createReshape("encoder.output.reshape", output,
			{get_batch_size(), m_model->max_length , HIDDEN_SIZE});
	return encoderHiddenOutput;
}

void Encoder::loadEncoderWieghts(){
	auto mod = m_model->F_->getParent();
    std::vector<Constant *> ConstVecH, ConstVecX;
	std::vector<Constant *> ConstVecOppH, ConstVecOppX;
    for (uint j = 0 ; j < ENCODER_LSTMS_NUM ; ++j)
    {
		for (uint i = 0 ; i < LSTM_LEVELS ; ++i)
		{
			std::string const_name_h = "encoder_"+ std::to_string(j) +"_lstm_" + files_gates[i]+"_wh1";
			std::string const_name_x = "encoder_"+ std::to_string(j) +"_lstm_" + files_gates[i]+"_wi1";
			Constant *ConstH = mod->getConstantByName(const_name_h);
			Constant *ConstX = mod->getConstantByName(const_name_x);
			ConstVecH.push_back(ConstH);
			ConstVecX.push_back(ConstX);
		}
		m_model->common_->loadMatrixAndSplitAndTransposeFromFile(
	   			"en2gr/encoder.rnn_layers."+ std::to_string(j) +".weight_hh_l0.bin" ,
				ConstVecH , LSTM_LEVELS);
	   	ConstVecH.clear();

	   	m_model->common_->loadMatrixAndSplitAndTransposeFromFile(
	   			"en2gr/encoder.rnn_layers."+ std::to_string(j) +".weight_ih_l0.bin" ,
				ConstVecX , LSTM_LEVELS);
	   	ConstVecX.clear();
    }

}

void Encoder::loadEncoderBiases(){
	auto mod = m_model->F_->getParent();
    std::vector<Constant *> ConstVecHb, ConstVecXb;
	std::vector<Constant *> ConstVecOppHb, ConstVecOppXb;
    for (uint j = 0 ; j < DECODER_LSTMS_NUM ; ++j)
    {
		for (uint i = 0 ; i < LSTM_LEVELS ; ++i)
		{
			std::string const_name_h_b = "encoder_"+ std::to_string(j) +"_lstm_" + files_gates[i]+"_bh1";
			std::string const_name_x_b = "encoder_"+ std::to_string(j) +"_lstm_" + files_gates[i]+"_bi1";
			Constant *ConstHb = mod->getConstantByName(const_name_h_b);
			Constant *ConstXb = mod->getConstantByName(const_name_x_b);
			ConstVecHb.push_back(ConstHb);
			ConstVecXb.push_back(ConstXb);
		}
		m_model->common_->loadMatrixAndSplitFromFile(
	   			"en2gr/encoder.rnn_layers."+ std::to_string(j) +".bias_hh_l0.bin" ,
				ConstVecHb , LSTM_LEVELS);
	   	ConstVecHb.clear();
	   	m_model->common_->loadMatrixAndSplitFromFile(
	   			"en2gr/encoder.rnn_layers."+ std::to_string(j) +".bias_ih_l0.bin" ,
				ConstVecXb , LSTM_LEVELS);
	   	ConstVecXb.clear();
    }

}

void Encoder::loadEncoderReverse(){
	auto mod = m_model->F_->getParent();
	std::vector<Constant *> ConstVecOppH, ConstVecOppX;
	std::vector<Constant *> ConstVecOppHb, ConstVecOppXb;
	for (uint i =0 ; i < LSTM_LEVELS ; i++){
		std::string const_name_opp_h = "encoder_opp_lstm_" + files_gates[i] +"_wh1";
		std::string const_name_opp_x = "encoder_opp_lstm_" + files_gates[i] +"_wi1";
		Constant *ConstOppH = mod->getConstantByName(const_name_opp_h);
		Constant *ConstOppX = mod->getConstantByName(const_name_opp_x);
		ConstVecOppH.push_back(ConstOppH);
		ConstVecOppX.push_back(ConstOppX);
	}
	m_model->common_->loadMatrixAndSplitAndTransposeFromFile(
			"en2gr/encoder.rnn_layers.0.weight_hh_l0_reverse.bin" ,
			ConstVecOppH , LSTM_LEVELS);
	ConstVecOppH.clear();
	m_model->common_->loadMatrixAndSplitAndTransposeFromFile(
			"en2gr/encoder.rnn_layers.0.weight_ih_l0_reverse.bin" ,
			ConstVecOppX , LSTM_LEVELS);
	ConstVecOppX.clear();

	for (uint i =0 ; i < LSTM_LEVELS ; i++){
		std::string const_name_opp_h_b = "encoder_opp_lstm_" + files_gates[i] +"_bh1";
		std::string const_name_opp_x_b = "encoder_opp_lstm_" + files_gates[i] +"_bi1";
		Constant *ConstOppHb = mod->getConstantByName(const_name_opp_h_b);
		Constant *ConstOppXb = mod->getConstantByName(const_name_opp_x_b);
		ConstVecOppHb.push_back(ConstOppHb);
		ConstVecOppXb.push_back(ConstOppXb);

	}
	m_model->common_->loadMatrixAndSplitFromFile(
   			"en2gr/encoder.rnn_layers.0.bias_hh_l0_reverse.bin" ,
			ConstVecOppHb , LSTM_LEVELS);
   	ConstVecOppHb.clear();
   	m_model->common_->loadMatrixAndSplitFromFile(
   			"en2gr/encoder.rnn_layers.0.bias_ih_l0_reverse.bin" ,
			ConstVecOppXb , LSTM_LEVELS);
	ConstVecOppXb.clear();

}
