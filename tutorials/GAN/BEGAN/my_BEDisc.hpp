#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.hpp"

template<typename DTYPE> class my_BEDisc : public NeuralNetwork<DTYPE> {
private:
public:
    my_BEDisc(Operator<float> *x){
        Alloc(x);
    }

    virtual ~my_BEDisc() {
    }

    int Alloc(Operator<float> *x){
        this->SetInput(x);

        Operator<float> *out = x;

        // input 28x28 output 14x14
        out = new ReShape<float>(out, 1, 28, 28, "Flat2Img");

        out = new ConvolutionLayer2D<float>(out, 1, 64, 4, 4, 2, 2, 1, 1, "D_Conv1");
        out = new Relu<float>(out, "D_Relu1");

        out = new ReShape<float>(out, 1, 1, 64 * 14 * 14, "Img2Flat");

        // Encode
        out = new Linear<float>(out, 64 * 14 * 14, 32);
        out = new BatchNormalizeLayer<float>(out, TRUE, "D_BN1");
        out = new Relu<float>(out, "D_Relu2");

        // Decode
        out = new Linear<float>(out, 32, 64 * 14 * 14);
        out = new BatchNormalizeLayer<float>(out, TRUE, "D_BN2");
        out = new Relu<float>(out, "D_Relu3");

        out = new ReShape<float>(out, 64, 14, 14, "Img2Flat");

        // input 14x14 output 28x28
        out = new TransposedConvolutionLayer2D<float>(out, 64, 1, 4, 4, 2, 2, 1, FALSE, "G_TranspoedConv1");
        out = new Sigmoid<float>(out, "D_Sigmoid3");

        out = new MSEOP<float>(out, x, "MSEOP");

        this->AnalyzeGraph(out);
    }
};
