/*
 * model.cpp
 *
 *  Created on: May 5, 2019
 *      Author: ezimmer9
 */


#include "model.h"
#include "common.cpp"
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

// namespace

void Model::createInferPytorchBiLSTM(PlaceholderBindings &bindings,
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
//  Placeholder *HInit =
//      getParent()->createPlaceholder(ElemKind::FloatTy, {batchSize,hiddenSize},
//                                     "initial_hidden_state", false);

  Placeholder *HInit =
      F_->getParent()->createPlaceholder(ElemKind::FloatTy, {hiddenSize,batchSize},
                                     "initial_hidden_state", false);
  bindings.allocate(HInit)->zero();
  Node *Ht = HInit;

//  Placeholder *CInit = F_->getParenF_->getParent()->aceholder(
//      ElemKind::FloatTy, {batchSize,hiddenSize}, "initial_cell_state", false);

  Placeholder *CInit = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize,batchSize}, "initial_cell_state", false);
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

  // forget gate
  Placeholder *Wif = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize,inputSize}, nameBase + ".Wif", false);
  Placeholder *Whf = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, hiddenSize}, nameBase + ".Whf", false);
  Placeholder *Bif = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize,1}, nameBase + ".Bif", false);
  Placeholder *Bhf = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize,1}, nameBase + ".Bhf", false);
  bindings.allocate(Wif);
  bindings.allocate(Whf);
  bindings.allocate(Bif);
  bindings.allocate(Bhf);

  // input gate
  Placeholder *Wii = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize,inputSize}, nameBase + ".Wii", false);
  Placeholder *Whi = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, hiddenSize}, nameBase + ".Whi", false);
  Placeholder *Bii = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize ,1}, nameBase + ".Bii", false);
  Placeholder *Bhi = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, 1}, nameBase + ".Bhi", false);
  bindings.allocate(Wii);
  bindings.allocate(Whi);
  bindings.allocate(Bii);
  bindings.allocate(Bhi);

  // output gate
  Placeholder *Wio = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize,inputSize}, nameBase + ".Wio", false);
  Placeholder *Who = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, hiddenSize}, nameBase + ".Who", false);
  Placeholder *Bio = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, 1}, nameBase + ".Bio", false);
  Placeholder *Bho = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, 1}, nameBase + ".Bho", false);
  bindings.allocate(Wio);
  bindings.allocate(Who);
  bindings.allocate(Bio);
  bindings.allocate(Bho);

  // cell state
  Placeholder *Wig = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize,inputSize}, nameBase + ".Wig", false);
  Placeholder *Whg = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, hiddenSize}, nameBase + ".Whg", false);
  Placeholder *Big = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, 1}, nameBase + ".Big", false);
  Placeholder *Bhg = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, 1}, nameBase + ".Bhg", false);
  bindings.allocate(Wig);
  bindings.allocate(Whg);
  bindings.allocate(Big);
  bindings.allocate(Bhg);

  std::vector<Node *> outputNodes;
  for (unsigned t = 0; t < timeSteps; t++) {
	auto *InputTranspose = F_->createTranspose("lstm.input.transpose", inputs[t],{1,0});

	// forget gate
    auto fc1Name = nameBase + ".fc1." + std::to_string(t);
    auto fc2Name = nameBase + ".fc2." + std::to_string(t);
    auto add1Name = nameBase + ".add1." + std::to_string(t);
    auto sigmoid1Name = nameBase + ".sigmoid1." + std::to_string(t);

    auto *MatMulFt = F_->createMatMul(fc1Name, NodeValue(Whf),NodeValue(Ht));
    auto *MatMulFtinput = F_->createMatMul(fc2Name, NodeValue(Wif),NodeValue(InputTranspose));
    auto *MatMulAdd1 = F_->createAdd(add1Name+fc1Name, MatMulFtinput, Bif);
    auto *MatMulAdd2 = F_->createAdd(add1Name+fc2Name, MatMulFt, Bhf);
    auto *Ft = F_->createSigmoid(
    		sigmoid1Name, F_->createAdd(add1Name, MatMulAdd1,MatMulAdd2));

    // input gate
    auto fc3Name = nameBase + ".fc3." + std::to_string(t);
    auto fc4Name = nameBase + ".fc4." + std::to_string(t);
    auto add2Name = nameBase + ".add2." + std::to_string(t);
    auto sigmoid2Name = nameBase + ".sigmoid2." + std::to_string(t);

    auto *MatMulIt = F_->createMatMul(fc3Name, NodeValue(Whi),NodeValue(Ht));
    auto *MatMulItinput = F_->createMatMul(fc4Name, NodeValue(Wii),NodeValue(InputTranspose));
    auto *It = F_->createSigmoid(
        sigmoid2Name,
		F_->createAdd(add2Name, F_->createAdd(add2Name+fc3Name, MatMulItinput, Bii),
				F_->createAdd(add2Name+fc4Name, MatMulIt, Bhi)));

    // output gate
    auto fc5Name = nameBase + ".fc5." + std::to_string(t);
    auto fc6Name = nameBase + ".fc6." + std::to_string(t);
    auto add3Name = nameBase + ".add3." + std::to_string(t);
    auto sigmoid3Name = nameBase + ".sigmoid3." + std::to_string(t);

    auto *MatMulOt = F_->createMatMul(fc5Name, NodeValue(Who),NodeValue(Ht));
    auto *MatMulOtinput = F_->createMatMul(fc6Name, NodeValue(Wio),NodeValue(InputTranspose));
    auto *Ot = F_->createSigmoid(
        sigmoid3Name,
		F_->createAdd(add3Name, F_->createAdd(add3Name+fc5Name, MatMulOtinput, Bio),
				F_->createAdd(add3Name+fc6Name, MatMulOt, Bho)));

    // cell gate
    auto fc7Name = nameBase + ".fc7." + std::to_string(t);
    auto fc8Name = nameBase + ".fc8." + std::to_string(t);
    auto add4Name = nameBase + ".add4." + std::to_string(t);
    auto tanh1Name = nameBase + ".tanh1." + std::to_string(t);


    auto *MatMulGt = F_->createMatMul(fc7Name, NodeValue(Whg),NodeValue(Ht));
    auto *MatMulGtinput = F_->createMatMul(fc8Name, NodeValue(Wig),NodeValue(InputTranspose));
    auto *Gt = F_->createTanh(
        tanh1Name,
		F_->createAdd(add4Name, F_->createAdd(add4Name+fc7Name, MatMulGtinput, Big),
				F_->createAdd(add4Name+fc8Name, MatMulGt, Bhg)));


    auto mul1Name = nameBase + ".mul1." + std::to_string(t);
    auto mul2Name = nameBase + ".mul2." + std::to_string(t);
    Ct = F_->createAdd(nameBase + ".C." + std::to_string(t),
    		F_->createMul(mul1Name, Ft, Ct), F_->createMul(mul2Name, It, Gt));

    auto htName = nameBase + ".H." + std::to_string(t);
    auto tanh2Name = nameBase + ".tanh2." + std::to_string(t);
    Ht = F_->createMul(htName, Ot, F_->createTanh(tanh2Name, Ct));
    auto *Ht_Transpose = F_->createTranspose("lstm.out.transpose", Ht , {1,0});
    outputs.push_back(Ht_Transpose);
  }
};

void Model::createInferPytorchLSTM(PlaceholderBindings &bindings,
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
      F_->getParent()->createPlaceholder(ElemKind::FloatTy, {hiddenSize,batchSize},
                                     "initial_hidden_state", false);
  bindings.allocate(HInit)->zero();
  Node *Ht = HInit;

  Placeholder *CInit = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, batchSize}, "initial_cell_state", false);
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

  // forget gate
  Placeholder *Wif = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {inputSize,hiddenSize}, nameBase + ".Wif", false);
  Placeholder *Whf = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, hiddenSize}, nameBase + ".Whf", false);
  Placeholder *Bif = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".Bif", false);
  Placeholder *Bhf = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".Bhf", false);
  bindings.allocate(Wif);
  bindings.allocate(Whf);
  bindings.allocate(Bif);
  bindings.allocate(Bhf);

  // input gate
  Placeholder *Wii = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {inputSize, hiddenSize}, nameBase + ".Wii", false);
  Placeholder *Whi = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, hiddenSize}, nameBase + ".Whi", false);
  Placeholder *Bii = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".Bii", false);
  Placeholder *Bhi = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".Bhi", false);
  bindings.allocate(Wii);
  bindings.allocate(Whi);
  bindings.allocate(Bii);
  bindings.allocate(Bhi);

  // output gate
  Placeholder *Wio = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {inputSize, hiddenSize}, nameBase + ".Wio", false);
  Placeholder *Who = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, hiddenSize}, nameBase + ".Who", false);
  Placeholder *Bio = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".Bio", false);
  Placeholder *Bho = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".Bho", false);
  bindings.allocate(Wio);
  bindings.allocate(Who);
  bindings.allocate(Bio);
  bindings.allocate(Bho);

  // cell state
  Placeholder *Wig = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {inputSize, hiddenSize}, nameBase + ".Wig", false);
  Placeholder *Whg = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize, hiddenSize}, nameBase + ".Whg", false);
  Placeholder *Big = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".Big", false);
  Placeholder *Bhg = F_->getParent()->createPlaceholder(
      ElemKind::FloatTy, {hiddenSize}, nameBase + ".Bhg", false);
  bindings.allocate(Wig);
  bindings.allocate(Whg);
  bindings.allocate(Big);
  bindings.allocate(Bhg);
  //  ****************** Ht Transpose ***************
  	auto *Ht_Transpose = F_->createTranspose("Ht.transpose" , Ht , {1,0});
  //  ***********************************************

  std::vector<Node *> outputNodes;
  for (unsigned t = 0; t < timeSteps; t++) {

    auto fc1Name = nameBase + ".fc1." + std::to_string(t);
    auto fc2Name = nameBase + ".fc2." + std::to_string(t);
    auto add1Name = nameBase + ".add1." + std::to_string(t);
    auto sigmoid1Name = nameBase + ".sigmoid1." + std::to_string(t);

    auto *FCFtInput = F_->createFullyConnected(fc1Name , inputs[t] ,Wif, Bif);
    auto *FCFtInputTranspose = F_->createTranspose("FCFtInput.Transpose", FCFtInput , {1,0});

    auto *FCFt = F_->createFullyConnected(fc2Name , Ht_Transpose , Whf, Bhf);
    auto *FCFtTranspose = F_->createTranspose("FCFt.Transpose", FCFt , {1,0});

    auto *Ft = F_->createSigmoid(
        sigmoid1Name,
		F_->createAdd(add1Name, FCFtInputTranspose,FCFtTranspose));

    auto fc3Name = nameBase + ".fc3." + std::to_string(t);
    auto fc4Name = nameBase + ".fc4." + std::to_string(t);
    auto add2Name = nameBase + ".add2." + std::to_string(t);
    auto sigmoid2Name = nameBase + ".sigmoid2." + std::to_string(t);

    auto *FCItInput = F_->createFullyConnected(fc3Name , inputs[t] ,Wii, Bii);
    auto *FCItInputTranspose = F_->createTranspose("FCItInput.Transpose", FCItInput , {1,0});

    auto *FCIt = F_->createFullyConnected(fc4Name , Ht_Transpose , Whi, Bhi);
    auto *FCItTranspose = F_->createTranspose("FCIt.Transpose", FCIt , {1,0});

    auto *It = F_->createSigmoid(
        sigmoid2Name,
		F_->createAdd(add2Name, FCItInputTranspose, FCItTranspose));

    auto fc5Name = nameBase + ".fc5." + std::to_string(t);
    auto fc6Name = nameBase + ".fc6." + std::to_string(t);
    auto add3Name = nameBase + ".add3." + std::to_string(t);
    auto sigmoid3Name = nameBase + ".sigmoid3." + std::to_string(t);

    auto *FCOtInput = F_->createFullyConnected(fc5Name , inputs[t] ,Wio, Bio);
    auto *FCOtInputTranspose = F_->createTranspose("FCOtInput.Transpose", FCOtInput , {1,0});

    auto *FCOt = F_->createFullyConnected(fc6Name , Ht_Transpose , Who, Bho);
    auto *FCOtTranspose = F_->createTranspose("FCOt.Transpose", FCOt , {1,0});

    auto *Ot = F_->createSigmoid(
        sigmoid3Name,
		F_->createAdd(add3Name, FCOtInputTranspose,FCOtTranspose));

    auto fc7Name = nameBase + ".fc7." + std::to_string(t);
    auto fc8Name = nameBase + ".fc8." + std::to_string(t);
    auto add4Name = nameBase + ".add4." + std::to_string(t);
    auto tanh1Name = nameBase + ".tanh1." + std::to_string(t);

    auto *FCGtInput = F_->createFullyConnected(fc7Name , inputs[t] ,Wig, Big);
    auto *FCGtInputTranspose = F_->createTranspose("FCGtInput.Transpose", FCGtInput , {1,0});

    auto *FCGt = F_->createFullyConnected(fc8Name , Ht_Transpose , Whg, Bhg);
    auto *FCGtTranspose = F_->createTranspose("FCOt.Transpose", FCGt , {1,0});

    auto *Gt = F_->createTanh(
    tanh1Name,
	F_->createAdd(add4Name, FCGtInputTranspose,FCGtTranspose));

    auto mul1Name = nameBase + ".mul1." + std::to_string(t);
    auto mul2Name = nameBase + ".mul2." + std::to_string(t);
    Ct = F_->createAdd(nameBase + ".C." + std::to_string(t),
    		F_->createMul(mul1Name, Ft, Ct), F_->createMul(mul2Name, It, Gt));

    auto htName = nameBase + ".H." + std::to_string(t);
    auto tanh2Name = nameBase + ".tanh2." + std::to_string(t);
    Ht = F_->createMul(htName, Ot, F_->createTanh(tanh2Name, Ct));
    Ht_Transpose = F_->createTranspose(nameBase +"lstm.out.transpose", Ht , {1,0});
    outputs.push_back(Ht_Transpose);
  }
};

void Model::loadLanguages(){
	std::printf("*** loadLanguages ***\n\n");
	gr_.loadVocabularyFromFile("en2gr/fr_vocabulary.txt");
	en_.loadVocabularyFromFile("en2gr/en_vocabulary.txt");
	embedding_gr_ = loadEmbedding("fr", gr_.index2word_.size());
	embedding_en_ = loadEmbedding("en", en_.index2word_.size());
}

void Model::loadTokens(){
	std::printf("*** loadTokens ***\n\n");
	tok_.loadVocabularyFromFile("../examples/en2gr-model/vocab.bpe.32000");
	embedding_tok_ = loadEmbedding("encoder.embedder.weight", tok_.index2word_.size()+4);
}

void debug_size_print(Node *candident){
	std::cout << " " << candident->getDebugDesc() << std::endl << std::endl;
}

void Model::loadEncoder(){
	std::printf("*** loadEncoder ***\n\n");

	auto &mod = EE_.getModule();

	input_ = mod.createPlaceholder(ElemKind::Int64ITy, {batchSize_, MAX_LENGTH}, "encoder.inputsentence", false);
	(*context.getPlaceholderBindings()).allocate(input_);

	Node *inputEmbedded =
	   F_->createGather("encoder.embedding", embedding_tok_, input_);

	std::vector<NodeValue> hidenOutputs0;
	std::vector<NodeValue> oppHidenOutputs0;
	std::vector<NodeValue> hidenOutputs1;
	std::vector<NodeValue> hidenOutputs2;
	std::vector<NodeValue> hidenOutputs3;
	std::vector<NodeValue> hiddenOut2;
	std::vector<NodeValue> encOut;

	std::vector<NodeValue> enc_seq;
	std::vector<SliceNode *> enc_seq__;
	std::vector<NodeValue> opposite_seq;

	F_->createSplit("encoder.split", inputEmbedded , MAX_LENGTH, 1 ,{}, enc_seq__);

	for (unsigned word_inx=0 ; word_inx < MAX_LENGTH; word_inx++){
		Node *reshape = F_->createReshape("encoder.reshape." + std::to_string(word_inx),
				enc_seq__[word_inx],{batchSize_, EMBEDDING_SIZE});
		NodeValue reshapeNV(reshape);
		enc_seq.push_back(reshapeNV);

		Node *oppReshape = F_->createReshape("opp.encoder.reshape." + std::to_string(word_inx),
				enc_seq__[MAX_LENGTH-1 - word_inx],{batchSize_, EMBEDDING_SIZE});
		NodeValue oppReshapeNV(oppReshape);
		opposite_seq.push_back(oppReshapeNV);
	}
	enc_seq__.clear();

	// Bi-Directional LSTM.
	createInferPytorchLSTM(*context.getPlaceholderBindings(), "encoder.lstm0", enc_seq ,
			batchSize_, EMBEDDING_SIZE , HIDDEN_SIZE, hidenOutputs0);

	createInferPytorchBiLSTM(*context.getPlaceholderBindings(), "encoder.opp.lstm", opposite_seq ,
			batchSize_, EMBEDDING_SIZE , HIDDEN_SIZE, oppHidenOutputs0);

	std::vector<NodeValue> lstm0Concat;
	for (uint i=0 ; i < hidenOutputs0.size() ; i++){
		Node *concat = F_->createConcat("encoder."+std::to_string(i)+".concat",
				{hidenOutputs0[i], oppHidenOutputs0[MAX_LENGTH-1-i]},1);
		NodeValue concatNV(concat);
		lstm0Concat.push_back(concatNV);
	}

	// ---------------- end Bi-Directional LSTM --------------

	// lstm 1
	createInferPytorchLSTM(*context.getPlaceholderBindings(),"encoder.lstm1",lstm0Concat ,
			batchSize_, HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs1);
	// lstm 2
	createInferPytorchLSTM(*context.getPlaceholderBindings(),"encoder.lstm2",hidenOutputs1 ,
			batchSize_, HIDDEN_SIZE  , HIDDEN_SIZE, hidenOutputs2);

	// add 0
	assert(hidenOutputs2.size() == hidenOutputs1.size() && "LSTM outputs doesn't align");
	for (uint i = 0 ; i < hidenOutputs2.size() ; i++){
		Node *add1 = F_->createAdd("encoder."+std::to_string(i)+".residual.",hidenOutputs2[i], hidenOutputs1[i]);
		NodeValue add1NV(add1);
		hiddenOut2.push_back(add1NV);
	}

	// lstm 3
	createInferPytorchLSTM(*context.getPlaceholderBindings(),"encoder.lstm3",hiddenOut2 ,
			batchSize_, HIDDEN_SIZE  , HIDDEN_SIZE, hidenOutputs3);

	// add 1
	assert(hidenOutputs3.size() == hiddenOut2.size() && "LSTM outputs doesn't align");
	for (uint i = 0 ; i < hidenOutputs2.size() ; i++){
		Node *add2 = F_->createAdd("encoder."+std::to_string(i)+".residual1.",hiddenOut2[i], hidenOutputs3[i]);
		NodeValue add2NV(add2);
		encOut.push_back(add2NV);
	}

	hidenOutputs0.clear();
	hidenOutputs1.clear();
	hidenOutputs2.clear();
	hidenOutputs3.clear();
	hiddenOut2.clear();

	Node *output = F_->createConcat("encoder.output", encOut, 0);
	// TODO: maybe no need here the reshape concat 3rd arg is dimension
	encoderHiddenOutput_ = F_->createReshape("encoder.output.reshape", output,
			{batchSize_, MAX_LENGTH , HIDDEN_SIZE});

	//  ******** example how to change the lstm weights *******
	//  *******************************************************
//	auto *test = mod.getPlaceholderByName("encoder_inputsentence");
//	//  convert all placeholders to constants except test
//	::convertPlaceholdersToConstants(F_ , *context.getPlaceholderBindings() , {test});
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

NodeValue Model::loadAttention(Node *AttentionQuery){
	std::printf("*** loadAttention ***\n\n");

	auto &mod = EE_.getModule();
	auto *Wa = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE , HIDDEN_SIZE},
			"attention.1.Wfc1_keys" , false);
	loadMatrixFromFile("en2gr/decoder.att_rnn.attn.linear_k.weight.bin", *(*context.getPlaceholderBindings()).allocate(Wa));

	auto *Bwa = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE},
				"attention.1.Bfc1_keys" , false);
	//no bias here - so allocate as zero
	(*context.getPlaceholderBindings()).allocate(Bwa)->zero();

	auto *Ua = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE , HIDDEN_SIZE},
			"attention.2.Wfc2_query" , false);
	loadMatrixFromFile("en2gr/decoder.att_rnn.attn.linear_q.weight.bin", *(*context.getPlaceholderBindings()).allocate(Ua));


	auto *Bua = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE},
			"attention.2.Bfc2_query" , false);
	//no bias here - so allocate as zero
	(*context.getPlaceholderBindings()).allocate(Bua)->zero();

	Placeholder *Vt = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE}, "attention.Vt", false);
	Node *NVt = Vt;
	loadMatrixFromFile("en2gr/decoder.att_rnn.attn.linear_att.bin", *(*context.getPlaceholderBindings()).allocate(Vt));

	Placeholder *NormBias = mod.createPlaceholder(ElemKind::FloatTy, {HIDDEN_SIZE}, "norm_bias", false);
	loadMatrixFromFile("en2gr/decoder.att_rnn.attn.normalize_bias.bin", *(*context.getPlaceholderBindings()).allocate(NormBias));

	Placeholder *NormScalar = mod.createPlaceholder(ElemKind::FloatTy, {1}, "norm_scalar", false);
	loadMatrixFromFile("en2gr/decoder.att_rnn.attn.normalize_scalar.bin", *(*context.getPlaceholderBindings()).allocate(NormScalar));

	Placeholder *SM = mod.createPlaceholder(ElemKind::Int64ITy, {batchSize_, MAX_LENGTH , 1}, "attention.softmax.ph", false);

	std::vector<NodeValue> Qexpand;
	const size_t querySize = 1;
	const size_t keySize = MAX_LENGTH;
	//TODO: not need to do every iteration
	/* ************************************************************* */
	/*                    Keys Fully Connected                       */
	/* ************************************************************* */

	// this is instead of fullyconnected that support only 2 dimentions.
	Node *BroadKeyW = F_->createBroadcast("attention.BroadKeyW",
			Wa , {batchSize_, HIDDEN_SIZE , HIDDEN_SIZE} , 1);
	auto *EncoderOutTranspoae = F_->createTranspose("attention.encoder.transpose",
			encoderHiddenOutput_ , {0 ,2 ,1});
	Node *BroadKeys = F_->createBatchMatMul("attention.broadkeys",
			BroadKeyW , EncoderOutTranspoae);
	auto *MatMulKeyTranspose = F_->createTranspose("attention.matmul.transpose",
			BroadKeys , {0 , 2 ,1});
	Node *BroadForAdd = F_->createBroadcast("attention.BroadForAdd",
			Bwa , {batchSize_, MAX_LENGTH , HIDDEN_SIZE} , 2);
	Node *AddKeys = F_->createAdd("attention.addkeys", MatMulKeyTranspose, BroadForAdd);

	//Node *KeyFC = F_->createFullyConnected("attention.keys.fc",);
	//debug_size_print(AddKeys);
	/* ****************************END****************************** */


	/* ************************************************************* */
	/*                    Query Fully Connected                       */
	/* ************************************************************* */
	Node *QueryExpand = F_->createExpandDims("attention.expandquery",
			AttentionQuery, {1});
	//debug_size_print(QueryExpand);
	auto *QueryTranspose = F_->createTranspose("attention.query.transpose",
			QueryExpand , {0,2,1});

	Node *BroadQureyW = F_->createBroadcast("attention.BroadQurey",
			Ua , {batchSize_, HIDDEN_SIZE , HIDDEN_SIZE} , 1);

	// this is instead of fullyconected that support only 2 dimentions.
	Node *MatMulQuery = F_->createBatchMatMul("attention.Mat.Mul.query",
			BroadQureyW ,QueryTranspose);
	auto *MatMulQueryTranspose = F_->createTranspose("attention.matmul.transpose",
			MatMulQuery , {0 , 2 ,1});

	Node *BroadForQureAdd = F_->createBroadcast("attention.BroadForQureAdd",
			Bua , {batchSize_, 1 , HIDDEN_SIZE} , 2);
	Node *AddQuery = F_->createAdd("attention.addquery", MatMulQueryTranspose, BroadForQureAdd);
	//debug_size_print(AddQuery);
	auto vect = [](Node *AddQuery){
		std::vector<NodeValue> vec;
		for (uint i=0 ; i < MAX_LENGTH ; i++){
			vec.push_back(AddQuery);
		}
		return vec;
	};
	Node *copyQuery = F_->createConcat("attention.concatquery",vect(AddQuery),1);
	//debug_size_print(copyQuery);
	/* ****************************END****************************** */


	/* ************************************************************* */
	/*                           Calc Score                          */
	/* ************************************************************* */

	// keys unsqueeze (b , 1 , keySize , hidden)
	Node *KeysUnsqueeze = F_->createExpandDims("attention.calc.score.expandim",
			AddKeys , {1});
	Node *KeyExpand = F_->createBroadcast("attention.calc.score.KeyBrod",
			KeysUnsqueeze , {batchSize_, querySize , keySize , HIDDEN_SIZE}, 0);
	//debug_size_print(KeyExpand);

	// query usqueeze + expand (b , querySize , 1 hidden)
	Node *QueryUnsqueese = F_->createBroadcast("attention.calc.score.queryBrod",
			copyQuery , {batchSize_, querySize , keySize , HIDDEN_SIZE}, 1);
	//debug_size_print(QueryUnsqueese);

	Node *CalcScoreAdd = F_->createAdd ("attention.add.inside", QueryUnsqueese, KeyExpand);
	//debug_size_print(CalcScoreAdd);

	// --------------------------- normalize ------------------------
	Node *BroadForNormBiasAdd = F_->createBroadcast("attention.BroadForQureAdd",
			NormBias , {batchSize_, querySize, keySize , HIDDEN_SIZE} , 3);
	Node *SumQK = F_->createAdd("attention.norm.batchadd", CalcScoreAdd,BroadForNormBiasAdd);
	Node *ReshSumQK = F_->createReshape("attention.norm.reshap",
			SumQK,{batchSize_*querySize , keySize , HIDDEN_SIZE});
	//debug_size_print(ReshSumQK);

	Node *PowSqure = F_->createPow("attention.pow" , NVt , 2);
	//debug_size_print(PowSqure);
	Node *Root = F_->createPow("attention.root" , F_->createBatchedReduceAdd("attention.reduce.add" ,
			PowSqure , {0}), 0.5);
	//debug_size_print(Root);
	Node *BroadcastRoot = F_->createBroadcast("attention.root.broadcast", Root,
			{HIDDEN_SIZE}, 0);
	//debug_size_print(BroadcastRoot);
	Node *LinearAtte = F_->createDiv("attention.linear.atte.div", NVt , BroadcastRoot);
	Node *BNormScalar = F_->createBroadcast("attention.linear.atten.broad",
			NormScalar , {HIDDEN_SIZE} , 0);

	LinearAtte = F_->createMul("attention.linear.atte.mul" , LinearAtte , BNormScalar);
	//debug_size_print(LinearAtte);

	Node *BlinearAtteExpand = F_->createExpandDims("attention.linear.atte.expanddim" ,
			LinearAtte , {0});
	Node *BlinearAtte = F_->createBroadcast("attention.linear.atte.broadcast" ,
			BlinearAtteExpand , {batchSize_*querySize , HIDDEN_SIZE} , 0);
	Node *BlinearAtteExpand2 = F_->createExpandDims("attention.linear.atte.expanddim2" ,
			BlinearAtte , {2});
	// --------------------------------------------------------------

	Node *THinside = F_->createTanh("attention.TanH.inside" , ReshSumQK);
	//debug_size_print(THinside);
	Node *Amul = F_->createBatchMatMul("attention.matmul"
			,THinside,BlinearAtteExpand2);
	//debug_size_print(Amul);
	/* ****************************END****************************** */

	Node *Asoftmax = F_->createSoftMax("attention.softmax" , Amul , SM);
	//debug_size_print(Asoftmax);

	Node *SoftmaxReshape = F_->createReshape("attention.softmax.reshape",
			Asoftmax , {batchSize_, keySize});
	//debug_size_print(SoftmaxReshape);
	auto *SoftmaxExpand = F_->createExpandDims("attention.softmax.expanddims",
			SoftmaxReshape , {1});
	//debug_size_print(SoftmaxExpand);
	Node *Bmm = F_->createBatchMatMul("attention.bmm", SoftmaxExpand , encoderHiddenOutput_);
	//debug_size_print(Bmm);
	Node *bmmReshape = F_->createReshape("attention.bmm.reshape", Bmm ,
			{batchSize_*querySize, HIDDEN_SIZE});
	//debug_size_print(bmmReshape);
	NodeValue bmmReshapeNV(bmmReshape);
	Qexpand.clear();

	return bmmReshapeNV;
}

void Model::loadDecoder(){
	std::printf("*** loadDecoder ***\n\n");
	auto &mod = EE_.getModule();

	std::vector<Node *> dec_seq;
	std::vector<NodeValue> dec_seq_embedding;
	std::vector<NodeValue> attentionOut;

	for (uint i=0 ; i < MAX_LENGTH ; i++){
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
	loadMatrixAndTransposeFromFile("en2gr/decoder.classifier.classifier.weight.bin", *(*context.getPlaceholderBindings()).allocate(classifier_w));

	Placeholder *classifier_b = mod.createPlaceholder(ElemKind::FloatTy,
				{tok_.index2word_.size()+4}, "decoder.classifier_w", false);
	loadMatrixFromFile("en2gr/decoder.classifier.classifier.bias.bin", *(*context.getPlaceholderBindings()).allocate(classifier_b));

	Placeholder *S = mod.createPlaceholder(ElemKind::Int64ITy, {batchSize_, 1}, "S", false);

	//TODO: init the 1st dec_seq with </s> embedding
	// the Initial vector fot the decoder will be:
	// { embedding(</s>), 0 , 0 ,0 ...}

	std::vector<NodeValue> hidenOutputs0;
	std::vector<Node *> hiddenOut0;
	std::vector<NodeValue> hidenOutputs1;
	std::vector<NodeValue> hidenOutputs2;
	std::vector<NodeValue> hidenOutputs3;

	std::vector<Node *> output_vec;

	for (uint word_inx=0 ; word_inx < MAX_LENGTH; word_inx++){
		std::printf("***************** word_ind %u***************\n", word_inx);


		Node *decoderEmbedded =
		   F_->createGather("decoder.embedding", embedding_tok_, dec_seq[word_inx]);
		//debug_size_print(decoderEmbedded);
		NodeValue decoderEmbeddedNV(decoderEmbedded);

		dec_seq_embedding[word_inx] = decoderEmbeddedNV;

		// lstm 0
		createInferPytorchLSTM(*context.getPlaceholderBindings(), "decoder.lstm.0."+std::to_string(word_inx), dec_seq_embedding ,
				batchSize_, HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs0);

		for (uint i = 0 ; i < hidenOutputs0.size() ; ++i){
			Node *hid = hidenOutputs0[i].getNode();
			hiddenOut0.push_back(hid);
		}

		std::vector<NodeValue> currentOut;
		// attention
		NodeValue attentionOutput = loadAttention(hiddenOut0[word_inx]);
		attentionOut[word_inx] = attentionOutput;
		for (uint i=0 ; i < attentionOut.size() ; i++){
			//concat 0
			currentOut.push_back(F_->createConcat("decoder."+std::to_string(i)+".concat",
					{hidenOutputs0[i], attentionOut[i]},1));
			//debug_size_print(currentOut[i]);
		}

		// lstm 1
		createInferPytorchLSTM(*context.getPlaceholderBindings(), "decoder.lstm.1."+std::to_string(word_inx), currentOut ,
				batchSize_, HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs1);

		currentOut.clear();
		for (uint i=0 ; i < hidenOutputs1.size() ; i++){
			// concat 1  - concat lstm1 with attentionout
			currentOut.push_back(F_->createConcat("decoder1."+std::to_string(i)+".concat",
					{hidenOutputs1[i], attentionOut[i]},1));
		}

		//lstm 2
		createInferPytorchLSTM(*context.getPlaceholderBindings(), "decoder.lstm.2."+std::to_string(word_inx), currentOut ,
				batchSize_, HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs2);

		// add residual 0 - lstm2 + attention-out
		std::vector<NodeValue> addResidual1;
		for (uint i=0 ; i < hidenOutputs2.size() ; i++){
			addResidual1.push_back(F_->createAdd("add.residual1."+std::to_string(i),
					hidenOutputs1[i] , hidenOutputs2[i]));
		}

		currentOut.clear();
		for (uint i=0 ; i < addResidual1.size() ; i++){
			// concat 2 - concat addResidual1 with attentionout
			currentOut.push_back(F_->createConcat("decoder2."+std::to_string(i)+".concat",
					{addResidual1[i], attentionOut[i]},1));
		}

		// lstm 3
		createInferPytorchLSTM(*context.getPlaceholderBindings(), "decoder.lstm.3."+std::to_string(word_inx), currentOut , batchSize_,
				 HIDDEN_SIZE , HIDDEN_SIZE, hidenOutputs3);


		// add residual 1 - LSTM2 + LSTM3
		std::vector<Node *> addResidual2;
		for (uint i=0 ; i < hidenOutputs3.size() ; i++){
			addResidual2.push_back(F_->createAdd("add.residual2."+std::to_string(i),
					addResidual1[i] , hidenOutputs3[i]));
		}

		hidenOutputs0.clear();
		hidenOutputs1.clear();
		hidenOutputs2.clear();
		hidenOutputs3.clear();
		hiddenOut0.clear();
		addResidual1.clear();
		// TODO: think how to avoid the last allocation for dec_seq

		// classifier
		auto *ClasiffierFC = F_->createFullyConnected("decoder.classifier.fc",
				addResidual2[word_inx] , classifier_w , classifier_b);
		//debug_size_print(ClasiffierFC);
		addResidual2.clear();

		// softmax
		Node *DecSM = F_->createSoftMax("decoder.softmax", ClasiffierFC, S);
		//debug_size_print(DecSM);
		// topk
		auto *topK = F_->createTopK("decoder.topK", DecSM, 1);
		//debug_size_print(topK);
	    Node *lastWordIdx =
	        F_->createReshape("decoder.reshape", topK->getIndices(), {batchSize_});
		//debug_size_print(lastWordIdx);

		if (word_inx+1 < MAX_LENGTH){
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
	//debug_size_print(output);
	Node *reshape = F_->createReshape("decoder.output.reshape", output,
	                                  {MAX_LENGTH, batchSize_});
	auto *save = F_->createSave("decoder.output", reshape);
	output_ = save->getPlaceholder();
	(*context.getPlaceholderBindings()).allocate(output_);
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

	    assert(words.size() <= (MAX_LENGTH-2) && "sentence is too long.");

	    input.getHandle<int64_t>().at({j, 0}) = BOS ;
	    std::cout << "BOS: " << ": " << input.getHandle<int64_t>().at({j, 0}) << "\n";
	    for (size_t i = 0; i < words.size(); i++) {
	      auto iter = tok_.word2index_.find(words[i]);
	      if (iter == tok_.word2index_.end()){
	    	  std::cout << "Unknown word, can't find it on the vocab!\n\n";
	    	  return;
	      }
	      std::cout << iter->first << ": " << iter->second +4 << "\n";
	      input.getHandle<int64_t>().at({j, i+1}) = (iter->second)+4;
	    }
	    input.getHandle<int64_t>().at({j, words.size()+1}) = EOS ;
	    std::cout << "EOS: " << ": " << input.getHandle<int64_t>().at({j, words.size()+1}) << "\n";
	  }
	updateInputPlaceholders(*(context.getPlaceholderBindings()), {input_}, {&input});
	EE_.run(context);

	auto OH = context.getPlaceholderBindings()->get(output_)->getHandle<int64_t>();
	for (unsigned j = 0; j < batch.size(); j++) {
		for (unsigned i = 0; i < MAX_LENGTH; i++) {
			int64_t wordIdx = (int64_t)OH.at({i, j});
			if (wordIdx == 3)
				break;

			if (i)
				std::cout << ' ';
				std::cout << wordIdx << ": " << tok_.index2word_[wordIdx-4];
		}
		std::cout << "\n\n";
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
	  loadMatrixFromFile("en2gr/" + langPrefix.str() + "_embedding.bin",*test);
	  return result;
}

void Model::loadEncoderWieghts(){
	auto mod = F_->getParent();
    std::vector<Constant *> ConstVecH, ConstVecX;
	std::vector<Constant *> ConstVecOppH, ConstVecOppX;
    for (uint j = 0 ; j < ENCODER_LSTMS_NUM ; ++j)
    {
		for (uint i = 0 ; i < LSTM_LEVELS ; ++i)
		{
			std::string const_name_h = "encoder_lstm"+ std::to_string(j) +"_Wh" + files_indices[i]+"1";
			std::string const_name_x = "encoder_lstm"+ std::to_string(j) +"_Wi" + files_indices[i]+"1";
			Constant *ConstH = mod->getConstantByName(const_name_h);
			Constant *ConstX = mod->getConstantByName(const_name_x);
			ConstVecH.push_back(ConstH);
			ConstVecX.push_back(ConstX);
		}
		loadMatrixAndSplitAndTransposeFromFile(
	   	//loadMatrixAndSplitFromFile(
	   			"en2gr/encoder.rnn_layers."+ std::to_string(j) +".weight_hh_l0.bin" ,
				ConstVecH , LSTM_LEVELS);
	   	ConstVecH.clear();

	   	loadMatrixAndSplitAndTransposeFromFile(
	   	//loadMatrixAndSplitFromFile(
	   			"en2gr/encoder.rnn_layers."+ std::to_string(j) +".weight_ih_l0.bin" ,
				ConstVecX , LSTM_LEVELS);
	   	ConstVecX.clear();
    }

}

void Model::loadEncoderBiases(){
	auto mod = F_->getParent();
    std::vector<Constant *> ConstVecHb, ConstVecXb;
	std::vector<Constant *> ConstVecOppHb, ConstVecOppXb;
    for (uint j = 0 ; j < DECODER_LSTMS_NUM ; ++j)
    {
		for (uint i = 0 ; i < LSTM_LEVELS ; ++i)
		{
			std::string const_name_h_b = "encoder_lstm"+ std::to_string(j) +"_Bh" + files_indices[i]+"1";
			std::string const_name_x_b = "encoder_lstm"+ std::to_string(j) +"_Bi" + files_indices[i]+"1";
			Constant *ConstHb = mod->getConstantByName(const_name_h_b);
			Constant *ConstXb = mod->getConstantByName(const_name_x_b);
			ConstVecHb.push_back(ConstHb);
			ConstVecXb.push_back(ConstXb);
		}
	   	loadMatrixAndSplitFromFile(
	   			"en2gr/encoder.rnn_layers."+ std::to_string(j) +".bias_hh_l0.bin" ,
				ConstVecHb , LSTM_LEVELS);
	   	ConstVecHb.clear();
	   	loadMatrixAndSplitFromFile(
	   			"en2gr/encoder.rnn_layers."+ std::to_string(j) +".bias_ih_l0.bin" ,
				ConstVecXb , LSTM_LEVELS);
	   	ConstVecXb.clear();
    }

}

void Model::loadEncoderReverse(){
	auto mod = F_->getParent();
	std::vector<Constant *> ConstVecOppH, ConstVecOppX;
	std::vector<Constant *> ConstVecOppHb, ConstVecOppXb;
	for (uint i =0 ; i < LSTM_LEVELS ; i++){
		std::string const_name_opp_h = "encoder_opp_lstm_Wh" + files_indices[i] +"1";
		std::string const_name_opp_x = "encoder_opp_lstm_Wi" + files_indices[i] +"1";
		Constant *ConstOppH = mod->getConstantByName(const_name_opp_h);
		Constant *ConstOppX = mod->getConstantByName(const_name_opp_x);
		ConstVecOppH.push_back(ConstOppH);
		ConstVecOppX.push_back(ConstOppX);
	}
	//loadMatrixAndSplitAndTransposeFromFile(
	loadMatrixAndSplitFromFile(
			"en2gr/encoder.rnn_layers.0.weight_hh_l0_reverse.bin" ,
			ConstVecOppH , LSTM_LEVELS);
	ConstVecOppH.clear();
	//loadMatrixAndSplitAndTransposeFromFile(
	loadMatrixAndSplitFromFile(
			"en2gr/encoder.rnn_layers.0.weight_ih_l0_reverse.bin" ,
			ConstVecOppX , LSTM_LEVELS);
	ConstVecOppX.clear();

	for (uint i =0 ; i < LSTM_LEVELS ; i++){
		std::string const_name_opp_h_b = "encoder_opp_lstm_Bh" + files_indices[i] +"1";
		std::string const_name_opp_x_b = "encoder_opp_lstm_Bi" + files_indices[i] +"1";
		Constant *ConstOppHb = mod->getConstantByName(const_name_opp_h_b);
		Constant *ConstOppXb = mod->getConstantByName(const_name_opp_x_b);
		ConstVecOppHb.push_back(ConstOppHb);
		ConstVecOppXb.push_back(ConstOppXb);

	}
   	loadMatrixAndSplitFromFile(
   			"en2gr/encoder.rnn_layers.0.bias_hh_l0_reverse.bin" ,
			ConstVecOppHb , LSTM_LEVELS);
   	ConstVecOppHb.clear();
   	loadMatrixAndSplitFromFile(
   			"en2gr/encoder.rnn_layers.0.bias_ih_l0_reverse.bin" ,
			ConstVecOppXb , LSTM_LEVELS);
	ConstVecOppXb.clear();

}

void Model::loadDecoderWieghts(){
	auto mod = F_->getParent();
    std::vector<Constant *> ConstVecH, ConstVecX;
	std::vector<Constant *> ConstVecOppH, ConstVecOppX;
	for (uint k = 0 ; k < MAX_LENGTH ; ++k)
	{
		for (uint j = 0 ; j < DECODER_LSTMS_NUM ; ++j)
		{
			for (uint i = 0 ; i < LSTM_LEVELS ; ++i)
			{
				std::string const_name_h = "decoder_lstm_"+ std::to_string(j) +"_" + std::to_string(k) +
						"_Wh" + files_indices[i]+"1";
				std::string const_name_x = "decoder_lstm_"+ std::to_string(j) +"_" + std::to_string(k) +
						"_Wi" + files_indices[i]+"1";
				//std::cout << const_name_h << "  " << const_name_x << "\n";
				Constant *ConstH = mod->getConstantByName(const_name_h);
				Constant *ConstX = mod->getConstantByName(const_name_x);
				ConstVecH.push_back(ConstH);
				ConstVecX.push_back(ConstX);
			}
			if (j == 0){
				loadMatrixAndSplitAndTransposeFromFile(
				//loadMatrixAndSplitFromFile(
						"en2gr/decoder.att_rnn.rnn.weight_hh_l0.bin" ,
						ConstVecH , LSTM_LEVELS);
				ConstVecH.clear();
				loadMatrixAndSplitAndTransposeFromFile(
				//loadMatrixAndSplitFromFile(
						"en2gr/decoder.att_rnn.rnn.weight_ih_l0.bin" ,
						ConstVecX , LSTM_LEVELS);
				ConstVecX.clear();
			}
			else{
				loadMatrixAndSplitAndTransposeFromFile(
				//loadMatrixAndSplitFromFile(
						"en2gr/decoder.rnn_layers."+ std::to_string(j-1) +".weight_hh_l0.bin" ,
						ConstVecH , LSTM_LEVELS);
				ConstVecH.clear();
				loadMatrixAndSplitAndTransposeFromFile(
				//loadMatrixAndSplitFromFile(
						"en2gr/decoder.rnn_layers."+ std::to_string(j-1) +".weight_ih_l0.bin" ,
						ConstVecX , LSTM_LEVELS);
				ConstVecX.clear();
			}
		}
	}
}

void Model::loadDecoderBiases(){
	auto mod = F_->getParent();
    std::vector<Constant *> ConstVecH, ConstVecX;
	std::vector<Constant *> ConstVecOppH, ConstVecOppX;
	for (uint k = 0 ; k < MAX_LENGTH ; ++k)
	{
		for (uint j = 0 ; j < ENCODER_LSTMS_NUM ; ++j)
		{
			for (uint i = 0 ; i < LSTM_LEVELS ; ++i)
			{
				std::string const_name_h = "decoder_lstm_"+ std::to_string(j) +"_" + std::to_string(k) +
						"_Bh" + files_indices[i]+"1";
				std::string const_name_x = "decoder_lstm_"+ std::to_string(j) +"_" + std::to_string(k) +
						"_Bi" + files_indices[i]+"1";
				//std::cout << const_name_h << "  " << const_name_x << "\n";
				Constant *ConstH = mod->getConstantByName(const_name_h);
				Constant *ConstX = mod->getConstantByName(const_name_x);
				ConstVecH.push_back(ConstH);
				ConstVecX.push_back(ConstX);
			}
			if (j == 0){
				loadMatrixAndSplitFromFile(
						"en2gr/decoder.att_rnn.rnn.bias_hh_l0.bin" ,
						ConstVecH , LSTM_LEVELS);
				ConstVecH.clear();
				loadMatrixAndSplitFromFile(
						"en2gr/decoder.att_rnn.rnn.bias_ih_l0.bin" ,
						ConstVecX , LSTM_LEVELS);
				ConstVecX.clear();
			}
			else{
				loadMatrixAndSplitFromFile(
						"en2gr/decoder.rnn_layers."+ std::to_string(j-1) +".bias_hh_l0.bin" ,
						ConstVecH , LSTM_LEVELS);
				ConstVecH.clear();
				loadMatrixAndSplitFromFile(
						"en2gr/decoder.rnn_layers."+ std::to_string(j-1) +".bias_ih_l0.bin" ,
						ConstVecX , LSTM_LEVELS);
				ConstVecX.clear();
			}
		}
	}

}

void Model::compile() {
	std::printf("*** compile ***\n\n");

    std::vector<Placeholder *> placeHolderVec;
    placeHolderVec.push_back(input_);
    placeHolderVec.push_back(output_);
    ::glow::convertPlaceholdersToConstants(F_, *context.getPlaceholderBindings(), placeHolderVec);
//    ConstList consts = F_->findConstants();
//    	for (auto con : consts){
//    		std::cout << "constant: " << con->getName().str ()<< "\n";
//   	}
    loadEncoderWieghts();
    loadEncoderBiases();
    loadEncoderReverse();
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
      if (ExecutionBackend == "CPU"){
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
