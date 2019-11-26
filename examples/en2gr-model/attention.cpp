/*
 * attention.cpp
 *
 *  Created on: Nov 14, 2019
 *      Author: ezimmer9
 */

//#include "common.h"
#include "attention.h"

void Attention::init(){
	auto &mod = _model->EE_.getModule();

	Wa = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE , HIDDEN_SIZE}, "attention.1.Wfc1_keys" , false);
	_model->common_->loadMatrixFromFile("en2gr/decoder.att_rnn.attn.linear_k.weight.bin", *(*_model->context.getPlaceholderBindings()).allocate(Wa));
	Bwa = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE}, "attention.1.Bfc1_keys" , false);
		//no bias here - so allocate as zero
	(*_model->context.getPlaceholderBindings()).allocate(Bwa)->zero();

	Ua = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE , HIDDEN_SIZE}, "attention.2.Wfc2_query" , false);
	_model->common_->loadMatrixFromFile("en2gr/decoder.att_rnn.attn.linear_q.weight.bin", *(*_model->context.getPlaceholderBindings()).allocate(Ua));
	Bua = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE}, "attention.2.Bfc2_query" , false);
	//no bias here - so allocate as zero
	(*_model->context.getPlaceholderBindings()).allocate(Bua)->zero();

	Vt = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE}, "attention.Vt", false);
	_model->common_->loadMatrixFromFile("en2gr/decoder.att_rnn.attn.linear_att.bin", *(*_model->context.getPlaceholderBindings()).allocate(Vt));

	NormBias = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE}, "norm_bias", false);
	_model->common_->loadMatrixFromFile("en2gr/decoder.att_rnn.attn.normalize_bias.bin", *(*_model->context.getPlaceholderBindings()).allocate(NormBias));

	NormScalar = mod.createPlaceholder(ElemKind::FloatTy, {1}, "norm_scalar", false);
	_model->common_->loadMatrixFromFile("en2gr/decoder.att_rnn.attn.normalize_scalar.bin", *(*_model->context.getPlaceholderBindings()).allocate(NormScalar));

	SM = mod.createPlaceholder(ElemKind::Int64ITy, {get_batch_size(), _model->max_length , 1}, "attention.softmax.ph", false);
}


AttentionParams Attention::loadAttention(Node *AttentionQuery){
	std::printf("*** loadAttention ***\n\n");

	uint batchBeam = get_batch_size() * get_beam_size();

	auto &mod = _model->EE_.getModule();
	Node *NVt = Vt;

	std::vector<NodeValue> Qexpand;
	const size_t querySize = 1;
	const size_t keySize = _model->max_length;

	//TODO: not need to do every iteration
	/* ************************************************************* */
	/*                    Keys Fully Connected                       */
	/* ************************************************************* */
	// this is instead of fullyconnected that support only 2 dimentions.
	Node *BroadKeyW = _model->F_->createBroadcast("attention.BroadKeyW",
			Wa , {batchBeam, HIDDEN_SIZE , HIDDEN_SIZE} , 1);
	auto *EncoderOutTranspoae = _model->F_->createTranspose("attention.encoder.transpose",
			_model->encoderHiddenOutput_ , {0 ,2 ,1});
	Node *BroadKeys = _model->F_->createBatchMatMul("attention.broadkeys",
			BroadKeyW , EncoderOutTranspoae);
	auto *MatMulKeyTranspose = _model->F_->createTranspose("attention.matmul.transpose",
			BroadKeys , {0 , 2 ,1});
	Node *BroadForAdd = _model->F_->createBroadcast("attention.BroadForAdd",
			Bwa , {batchBeam, _model->max_length , HIDDEN_SIZE} , 2);
	Node *AddKeys = _model->F_->createAdd("attention.addkeys", MatMulKeyTranspose, BroadForAdd);
	/* ****************************END****************************** */


	/* ************************************************************* */
	/*                    Query Fully Connected                       */
	/* ************************************************************* */
	Node *QueryExpand = _model->F_->createExpandDims("attention.expandquery",
			AttentionQuery, {1});
	auto *QueryTranspose = _model->F_->createTranspose("attention.query.transpose",
			QueryExpand , {0,2,1});

	Node *BroadQureyW = _model->F_->createBroadcast("attention.BroadQurey",
			Ua , {batchBeam, HIDDEN_SIZE , HIDDEN_SIZE} , 1);

	// this is instead of fullyconected that support only 2 dimentions.
	Node *MatMulQuery = _model->F_->createBatchMatMul("attention.Mat.Mul.query",
			BroadQureyW ,QueryTranspose);
	auto *MatMulQueryTranspose = _model->F_->createTranspose("attention.matmul.transpose",
			MatMulQuery , {0 , 2 ,1});

	Node *BroadForQureAdd = _model->F_->createBroadcast("attention.BroadForQureAdd",
			Bua , {batchBeam, 1 , HIDDEN_SIZE} , 2);
	Node *AddQuery = _model->F_->createAdd("attention.addquery", MatMulQueryTranspose, BroadForQureAdd);
	auto vect = [](Node *AddQuery , uint max_length){
		std::vector<NodeValue> vec;
		for (uint i=0 ; i < max_length ; i++){
			vec.push_back(AddQuery);
		}
		return vec;
	};
	Node *copyQuery = _model->F_->createConcat("attention.concatquery",vect(AddQuery, _model->max_length),1);
	/* ****************************END****************************** */


	/* ************************************************************* */
	/*                           Calc Score                          */
	/* ************************************************************* */
	Node *KeysUnsqueeze = _model->F_->createExpandDims("attention.calc.score.expandim", AddKeys , {1});
	Node *KeyExpand = _model->F_->createBroadcast("attention.calc.score.KeyBrod",
			KeysUnsqueeze , {batchBeam, querySize , keySize , HIDDEN_SIZE}, 0);

	// query usqueeze + expand (b , querySize , 1 hidden)
	Node *QueryUnsqueese = _model->F_->createExpandDims("attention.calc.score.queryexpandim", copyQuery , {1});
	Node *CalcScoreAdd = _model->F_->createAdd ("attention.add.inside", QueryUnsqueese, KeyExpand);

	// --------------------------- normalize ------------------------
	Node *BroadForNormBiasAdd = _model->F_->createBroadcast("attention.BroadForQureAdd",
			NormBias , {batchBeam, querySize, keySize , HIDDEN_SIZE} , 3);
	Node *SumQK = _model->F_->createAdd("attention.norm.batchadd", CalcScoreAdd,BroadForNormBiasAdd);
	Node *ReshSumQK = _model->F_->createReshape("attention.norm.reshap",
			SumQK,{batchBeam*querySize , keySize , HIDDEN_SIZE});

	Node *PowSqure = _model->F_->createPow("attention.pow" , NVt , 2);
	Node *Root = _model->F_->createPow("attention.root" , _model->F_->createBatchedReduceAdd("attention.reduce.add" , PowSqure , {0}), 0.5);
	Node *BroadcastRoot = _model->F_->createBroadcast("attention.root.broadcast", Root, {HIDDEN_SIZE}, 0);
	Node *LinearAtte = _model->F_->createDiv("attention.linear.atte.div", NVt , BroadcastRoot);
	Node *BNormScalar = _model->F_->createBroadcast("attention.linear.atten.broad", NormScalar , {HIDDEN_SIZE} , 0);
	LinearAtte = _model->F_->createMul("attention.linear.atte.mul" , LinearAtte , BNormScalar);
	Node *BlinearAtteExpand = _model->F_->createExpandDims("attention.linear.atte.expanddim" , LinearAtte , {0});
	Node *BlinearAtte = _model->F_->createBroadcast("attention.linear.atte.broadcast" ,
			BlinearAtteExpand , {batchBeam*querySize , HIDDEN_SIZE} , 0);
	Node *BlinearAtteExpand2 = _model->F_->createExpandDims("attention.linear.atte.expanddim2" , BlinearAtte , {2});
	// --------------------------------------------------------------

	Node *THinside = _model->F_->createTanh("attention.TanH.inside" , ReshSumQK);
	Node *Amul = _model->F_->createBatchMatMul("attention.matmul", THinside,BlinearAtteExpand2);
	/* ****************************END****************************** */

    /****************************** MASK PADDING *********************************/
	if (!_model->alloc_mask){
		_model->attention_mask_= mod.createPlaceholder(ElemKind::FloatTy, {get_beam_size(), _model->max_length, 1}, "values", false);
		(*_model->context.getPlaceholderBindings()).allocate(_model->attention_mask_);
		_model->alloc_mask = true;
	}

    Node* AmulMaskNode = _model->attention_mask_;
    Node* newAmul = _model->F_->createAdd("new Amul",Amul , AmulMaskNode);
	Node *Asoftmax = _model->F_->createSoftMax("attention.softmax" , newAmul , SM);
	Node *SoftmaxReshape = _model->F_->createReshape("attention.softmax.reshape", Asoftmax , {batchBeam, keySize});
	auto *SoftmaxExpand = _model->F_->createExpandDims("attention.softmax.expanddims", SoftmaxReshape , {1});
	Node *Bmm = _model->F_->createBatchMatMul("attention.bmm", SoftmaxExpand , _model->encoderHiddenOutput_);
	Node *bmmReshape = _model->F_->createReshape("attention.bmm.reshape", Bmm , {batchBeam*querySize, HIDDEN_SIZE});
	NodeValue bmmReshapeNV(bmmReshape);
	Qexpand.clear();
	attention_out.attention_out = bmmReshapeNV;
	attention_out.scores = SoftmaxReshape;

	return attention_out;
}
