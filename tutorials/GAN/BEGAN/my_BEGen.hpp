#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.hpp"

template<typename DTYPE> class my_BEGen : public NeuralNetwork<DTYPE> {
private:
public:
    my_BEGen(Operator<float> *z){
        Alloc(z);
    }

    virtual ~my_BEGen() {
    }

    int Alloc(Operator<float> *z){
        this->SetInput(z);

        Operator<float> *out = z;

        out = new Linear<float>(out, 100, 1024, TRUE, "G_L1");
        out = new BatchNormalizeLayer<float>(out, TRUE, "G_BN1");
        out = new Relu<float>(out, "G_Relu1");

        out = new Linear<float>(out, 1024, 128 * 7 * 7, TRUE, "G_L2");
        out = new BatchNormalizeLayer<float>(out, TRUE, "G_BN2");
        out = new Relu<float>(out, "G_Relu2");

        out = new ReShape<float>(out, 128, 7, 7, "G_Reshape1");

        // input 7x7 output 14x14
        out = new TransposedConvolutionLayer2D<float>(out, 128, 64, 4, 4, 2, 2, 1, FALSE, "G_TranspoedConv1");
        out = new BatchNormalizeLayer<float>(out, TRUE, "G_BN3");
        out = new Relu<float>(out, "G_Relu3");

        out = new TransposedConvolutionLayer2D<float>(out, 64, 1, 4, 4, 2, 2, 1, FALSE, "G_TranspoedConv2");
        out = new Tanh<float>(out, "G_Tanh");
        out = new ReShape<float>(out, 1, 1, 28*28, "Img2Flat");

        this->AnalyzeGraph(out);
    }
};
