/*
 * lstm.cpp
 *
 *  Created on: Nov 12, 2019
 *      Author: ezimmer9
 */
#include "lstm.h"

void Gate::setParams(std::string name, unsigned inputSize, unsigned hiddenSize){
	_name = name;
	_inputSize = inputSize;
	_hiddenSize = hiddenSize;
}

void Gate::setPlaceholders(){
	Wi = _model->F_->getParent()->createPlaceholder(
			ElemKind::FloatTy, {_inputSize,_hiddenSize}, _name + ".wi", false);
	Wh = _model->F_->getParent()->createPlaceholder(
			ElemKind::FloatTy, {_hiddenSize, _hiddenSize}, _name + ".wh" , false);
	Bi = _model->F_->getParent()->createPlaceholder(
			ElemKind::FloatTy, {_hiddenSize}, _name + ".bi" , false);
	Bh = _model->F_->getParent()->createPlaceholder(
			ElemKind::FloatTy, {_hiddenSize}, _name + ".bh" , false);
}


void Gate::allocate(){
	(_model->context.getPlaceholderBindings())->allocate(Wi);
	(_model->context.getPlaceholderBindings())->allocate(Wh);
	(_model->context.getPlaceholderBindings())->allocate(Bi);
	(_model->context.getPlaceholderBindings())->allocate(Bh);
}


void LSTM::setParams(std::string name_, unsigned inputSize_, unsigned hiddenSize_){
	_name = name_;
	inputSize = inputSize_;
	hiddenSize = hiddenSize_;
}
void LSTM::updateGates(){
	forget.setParams(_name + ".forget" ,inputSize , hiddenSize);
	input.setParams(_name + ".input" ,inputSize , hiddenSize);
	output.setParams(_name + ".output" ,inputSize , hiddenSize);
	cell.setParams(_name + ".cell" ,inputSize , hiddenSize);
}

void LSTM::createInferPytorchLSTM(PlaceholderBindings &bindings,
                          llvm::StringRef namePrefix,
                          llvm::ArrayRef<NodeValue> inputs, unsigned batchSize,
                          unsigned hiddenSize, unsigned outputSize,
                          std::vector<NodeValue> &outputs) {
	std::string nameBase = namePrefix;
	const unsigned timeSteps = inputs.size();
	assert(timeSteps > 0 && "empty input");
	const unsigned inputSize = inputs.front().dims().back();
	assert(inputSize > 0 && "input dimensionality is zero");

	// Initialize the hidden and cell states to zero.
	Placeholder *HInit =
	  _model->F_->getParent()->createPlaceholder(ElemKind::FloatTy, {hiddenSize,batchSize},
			  nameBase+".initial_hidden_state", false);
	bindings.allocate(HInit)->zero();
	Node *Ht = HInit;

	Placeholder *CInit = _model->F_->getParent()->createPlaceholder(ElemKind::FloatTy,
		  {hiddenSize, batchSize}, nameBase+".initial_cell_state.", false);
	bindings.allocate(CInit)->zero();
	Node *Ct = CInit;

	// Forget gate:
	//    f_t <- sigmoid([Wif * x^T + bif] + [Whf * h_t + bhf])
	// Input gate:
	//    i_t <- sigmoid([Wii * x^T + bii] + [Whi * h_t + bhi])
	// Output gate:
	//    o_t <- sigmoid([Wio * x^T + bio] + [Who * h_t + bho])
	// Gated
	//    g_t <- tanh([Wig  * x^T + big] + [Whg * h_t + bhg])
	// Cell state:
	//    c_t <- f_t (mul) c_(t-1) + i_t (mul) g_t
	// Hidden state:
	//    h <- o_t (mul) tanh(c_t)
	this->setParams(nameBase, inputSize , hiddenSize);
	this->updateGates();
	this->forget.setPlaceholders();
	this->forget.allocate();

	this->input.setPlaceholders();
	this->input.allocate();

	this->output.setPlaceholders();
	this->output.allocate();

	this->cell.setPlaceholders();
	this->cell.allocate();

	//  ****************** Ht Transpose ***************
	auto *Ht_Transpose = _model->F_->createTranspose(nameBase+".Ht.transpose." , Ht , {1,0});
	//  ***********************************************

	std::vector<Node *> outputNodes;
	for (unsigned t = 0; t < timeSteps; t++) {

		auto fc1Name = nameBase + ".fc1." + std::to_string(t)+".";
		auto fc2Name = nameBase + ".fc2." + std::to_string(t)+".";
		auto add1Name = nameBase + ".add1." + std::to_string(t)+".";
		auto sigmoid1Name = nameBase + ".sigmoid1." + std::to_string(t)+".";

		auto *FCFtInput = _model->F_->createFullyConnected(fc1Name , inputs[t] ,this->forget.Wi, this->forget.Bi);
		auto *FCFtInputTranspose = _model->F_->createTranspose(fc1Name+".FCFtInput.Transpose", FCFtInput , {1,0});

		auto *FCFt = _model->F_->createFullyConnected(fc2Name , Ht_Transpose , this->forget.Wh, this->forget.Bh);
		auto *FCFtTranspose = _model->F_->createTranspose(fc2Name+".FCFt.Transpose", FCFt , {1,0});

		auto *Ft = _model->F_->createSigmoid(sigmoid1Name, _model->F_->createAdd(add1Name, FCFtInputTranspose,FCFtTranspose));

		auto fc3Name = nameBase + ".fc3." + std::to_string(t)+".";
		auto fc4Name = nameBase + ".fc4." + std::to_string(t)+".";
		auto add2Name = nameBase + ".add2." + std::to_string(t)+".";
		auto sigmoid2Name = nameBase + ".sigmoid2." + std::to_string(t)+".";

		auto *FCItInput = _model->F_->createFullyConnected(fc3Name , inputs[t] ,this->input.Wi, this->input.Bi);
		auto *FCItInputTranspose = _model->F_->createTranspose(fc3Name+".FCItInput.Transpose", FCItInput , {1,0});

		auto *FCIt = _model->F_->createFullyConnected(fc4Name , Ht_Transpose , this->input.Wh, this->input.Bh);
		auto *FCItTranspose = _model->F_->createTranspose(fc3Name+".FCIt.Transpose", FCIt , {1,0});

		auto *It = _model->F_->createSigmoid(sigmoid2Name, _model->F_->createAdd(add2Name, FCItInputTranspose, FCItTranspose));

		auto fc5Name = nameBase + ".fc5." + std::to_string(t)+".";
		auto fc6Name = nameBase + ".fc6." + std::to_string(t)+".";
		auto add3Name = nameBase + ".add3." + std::to_string(t)+".";
		auto sigmoid3Name = nameBase + ".sigmoid3." + std::to_string(t)+".";

		auto *FCOtInput = _model->F_->createFullyConnected(fc5Name , inputs[t] ,this->output.Wi, this->output.Bi);
		auto *FCOtInputTranspose = _model->F_->createTranspose(fc5Name+".FCOtInput.Transpose", FCOtInput , {1,0});

		auto *FCOt = _model->F_->createFullyConnected(fc6Name , Ht_Transpose , this->output.Wh, this->output.Bh);
		auto *FCOtTranspose = _model->F_->createTranspose(fc6Name+".FCOt.Transpose", FCOt , {1,0});

		auto *Ot = _model->F_->createSigmoid(sigmoid3Name, _model->F_->createAdd(add3Name, FCOtInputTranspose,FCOtTranspose));

		auto fc7Name = nameBase + ".fc7." + std::to_string(t)+".";
		auto fc8Name = nameBase + ".fc8." + std::to_string(t)+".";
		auto add4Name = nameBase + ".add4." + std::to_string(t)+".";
		auto tanh1Name = nameBase + ".tanh1." + std::to_string(t)+".";

		auto *FCGtInput = _model->F_->createFullyConnected(fc7Name , inputs[t] ,this->cell.Wi, this->cell.Bi);
		auto *FCGtInputTranspose = _model->F_->createTranspose(fc7Name+".FCGtInput.Transpose", FCGtInput , {1,0});

		auto *FCGt = _model->F_->createFullyConnected(fc8Name , Ht_Transpose , this->cell.Wh, this->cell.Bh);
		auto *FCGtTranspose = _model->F_->createTranspose(fc8Name+".FCOt.Transpose", FCGt , {1,0});

		auto *Gt = _model->F_->createTanh(tanh1Name,_model->F_->createAdd(add4Name, FCGtInputTranspose,FCGtTranspose));

		auto mul1Name = nameBase + ".mul1." + std::to_string(t)+".";
		auto mul2Name = nameBase + ".mul2." + std::to_string(t)+".";
		Ct = _model->F_->createAdd(nameBase + ".C." + std::to_string(t)+".", _model->F_->createMul(mul1Name, Ft, Ct), _model->F_->createMul(mul2Name, It, Gt));

		auto htName = nameBase + ".H." + std::to_string(t)+".";
		auto tanh2Name = nameBase + ".tanh2." + std::to_string(t)+".";
		Ht = _model->F_->createMul(htName, Ot, _model->F_->createTanh(tanh2Name, Ct));
		Ht_Transpose = _model->F_->createTranspose(nameBase +".lstm.out.transpose."+ std::to_string(t)+".", Ht , {1,0});
		outputs.push_back(Ht_Transpose);
	}
};

void LSTM::createDecoderInferPytorchLSTM(PlaceholderBindings &bindings,
                          llvm::StringRef namePrefix,
                          llvm::ArrayRef<NodeValue> inputs, unsigned batchSize,
                          unsigned hiddenSize, unsigned outputSize,
                          std::vector<std::vector<NodeValue>> &outputs ,
						  std::vector<Node *> hidden , uint word_inx) {
  std::string nameBase = namePrefix;
  const unsigned timeSteps = inputs.size();
  assert(timeSteps > 0 && "empty input");
  const unsigned inputSize = inputs.front().dims().back();
  assert(inputSize > 0 && "input dimensionality is zero");

  // Initialize the hidden and cell states to zero.
  Node *Ht;
  Node *Ct;
  if (word_inx == 0){
	  Placeholder *HInit = _model->F_->getParent()->createPlaceholder(ElemKind::FloatTy, {hiddenSize,batchSize},
	    		  nameBase+".initial_hidden_state", false);
	  bindings.allocate(HInit)->zero();
	  Ht = HInit;

	  Placeholder *CInit = _model->F_->getParent()->createPlaceholder(ElemKind::FloatTy,
			  {hiddenSize, batchSize}, nameBase+".initial_cell_state", false);
	  bindings.allocate(CInit)->zero();
	  Ct = CInit;
  }
  else{
	  Ht = (Node *)hidden[1];
	  Ct = (Node *)hidden[2];
  }

  // Forget gate:
  //    f_t <- sigmoid([Wif * x^T + bif] + [Whf * h_t + bhf])
  // Input gate:
  //    i_t <- sigmoid([Wii * x^T + bii] + [Whi * h_t + bhi])
  // Output gate:
  //    o_t <- sigmoid([Wio * x^T + bio] + [Who * h_t + bho])
  // Gated
  //    g_t <- tanh([Wig  * x^T + big] + [Whg * h_t + bhg])
  // Cell state:
  //    c_t <- f_t (mul) c_(t-1) + i_t (mul) g_t
  // Hidden state:
  //    h <- o_t (mul) tanh(c_t)
    this->setParams(nameBase, inputSize , hiddenSize);
    this->updateGates();
    if (word_inx == 0){
    	this->forget.setPlaceholders();
    	this->forget.allocate();

    	this->input.setPlaceholders();
    	this->input.allocate();

    	this->output.setPlaceholders();
    	this->output.allocate();

    	this->cell.setPlaceholders();
    	this->cell.allocate();
    }

  //  ****************** Ht Transpose ***************
  auto *Ht_Transpose = _model->F_->createTranspose(nameBase+".Ht.transpose." , Ht , {1,0});
  //  ***********************************************

  std::vector<Node *> outputNodes;
  for (unsigned t = 0; t < timeSteps; t++) {

    auto fc1Name = nameBase + ".fc1." + std::to_string(t)+".";
    auto fc2Name = nameBase + ".fc2." + std::to_string(t)+".";
    auto add1Name = nameBase + ".add1." + std::to_string(t)+".";
    auto sigmoid1Name = nameBase + ".sigmoid1." + std::to_string(t)+".";

    Node *FCFtInput = _model->F_->createFullyConnected(fc1Name + ".Input" , inputs[t] ,this->forget.Wi, this->forget.Bi);
    Node *FCFt = _model->F_->createFullyConnected(fc2Name + ".Input" , Ht_Transpose , this->forget.Wh, this->forget.Bh);
    auto *FCFtTranspose = _model->F_->createTranspose(fc2Name+".FCFt.Transpose", FCFt , {1,0});
    auto *FCFtInputTranspose = _model->F_->createTranspose(fc1Name+".FCFtInput.Transpose", FCFtInput , {1,0});


    auto *Ft = _model->F_->createSigmoid(sigmoid1Name, _model->F_->createAdd(add1Name, FCFtInputTranspose,FCFtTranspose));

    auto fc3Name = nameBase + ".fc3." + std::to_string(t)+".";
    auto fc4Name = nameBase + ".fc4." + std::to_string(t)+".";
    auto add2Name = nameBase + ".add2." + std::to_string(t)+".";
    auto sigmoid2Name = nameBase + ".sigmoid2." + std::to_string(t)+".";

    Node *FCItInput = _model->F_->createFullyConnected(fc3Name+ ".Input" , inputs[t] ,this->input.Wi, this->input.Bi);
    Node *FCIt = _model->F_->createFullyConnected(fc4Name+ ".Input" , Ht_Transpose , this->input.Wh, this->input.Bh);
    auto *FCItTranspose = _model->F_->createTranspose(fc3Name+".FCIt.Transpose", FCIt , {1,0});
    auto *FCItInputTranspose = _model->F_->createTranspose(fc3Name+".FCItInput.Transpose", FCItInput , {1,0});

    auto *It = _model->F_->createSigmoid(sigmoid2Name, _model->F_->createAdd(add2Name, FCItInputTranspose, FCItTranspose));

    auto fc5Name = nameBase + ".fc5." + std::to_string(t)+".";
    auto fc6Name = nameBase + ".fc6." + std::to_string(t)+".";
    auto add3Name = nameBase + ".add3." + std::to_string(t)+".";
    auto sigmoid3Name = nameBase + ".sigmoid3." + std::to_string(t)+".";

    Node *FCOtInput = _model->F_->createFullyConnected(fc5Name+ ".Input" , inputs[t] ,this->output.Wi, this->output.Bi);
    Node *FCOt = _model->F_->createFullyConnected(fc6Name+ ".Input" , Ht_Transpose , this->output.Wh, this->output.Bh);
    auto *FCOtTranspose = _model->F_->createTranspose(fc6Name+".FCOt.Transpose", FCOt , {1,0});
    auto *FCOtInputTranspose = _model->F_->createTranspose(fc5Name+".FCOtInput.Transpose", FCOtInput , {1,0});

    auto *Ot = _model->F_->createSigmoid(sigmoid3Name, _model->F_->createAdd(add3Name, FCOtInputTranspose,FCOtTranspose));

    auto fc7Name = nameBase + ".fc7." + std::to_string(t)+".";
    auto fc8Name = nameBase + ".fc8." + std::to_string(t)+".";
    auto add4Name = nameBase + ".add4." + std::to_string(t)+".";
    auto tanh1Name = nameBase + ".tanh1." + std::to_string(t)+".";

    Node *FCGtInput = _model->F_->createFullyConnected(fc7Name+ ".Input" , inputs[t] ,this->cell.Wi, this->cell.Bi);
    Node *FCGt = _model->F_->createFullyConnected(fc8Name+ ".Input" , Ht_Transpose , this->cell.Wh, this->cell.Bh);
    auto *FCGtInputTranspose = _model->F_->createTranspose(fc7Name+".FCGtInput.Transpose", FCGtInput , {1,0});
    auto *FCGtTranspose = _model->F_->createTranspose(fc8Name+".FCOt.Transpose", FCGt , {1,0});

    auto *Gt = _model->F_->createTanh(tanh1Name, _model->F_->createAdd(add4Name, FCGtInputTranspose,FCGtTranspose));

    auto mul1Name = nameBase + ".mul1." + std::to_string(t)+".";
    auto mul2Name = nameBase + ".mul2." + std::to_string(t)+".";
    Node *temp1 = _model->F_->createMul(mul1Name + "mul1.for.Ct." + std::to_string(t)+".", Ft, Ct);
    Node *temp2 = _model->F_->createMul(mul2Name + "mul2.for.Ct." + std::to_string(t)+".", It, Gt);
    Ct = _model->F_->createAdd(nameBase + ".C." + std::to_string(t)+".",temp1, temp2);
    auto htName = nameBase + ".H." + std::to_string(t)+".";
    auto tanh2Name = nameBase + ".tanh2." + std::to_string(t)+".";
    Ht = _model->F_->createMul(htName, Ot, _model->F_->createTanh(tanh2Name, Ct));
    Ht_Transpose = _model->F_->createTranspose(nameBase +".lstm.out.transpose."+ std::to_string(t)+"."
    		, Ht , {1,0});
    std::vector<NodeValue> temp;
    temp.push_back(Ht_Transpose);
    temp.push_back(Ht);
    temp.push_back(Ct);
    outputs.push_back(temp);
  }
};
