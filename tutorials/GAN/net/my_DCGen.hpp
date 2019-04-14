#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.hpp"

using namespace std;

template<typename DTYPE> class my_DCGen : public NeuralNetwork<DTYPE> {
private:
public:
    my_DCGen(Operator<float> *z){
        Alloc(z);
    }

    virtual ~my_DCGen() {
    }

    int Alloc(Operator<float> *z){
        this->SetInput(z);

        Operator<float> *out = z;
        const int D = 128;
        

        // // for 64x64 
        // // ======================= layer 1 ======================
        // out = new TransposedConvolutionLayer2D<float>(out, 100, D*8, 4, 4, 1, 1, 0, 0, "G_TranspoedConv1");
        // out = new BatchNormalizeLayer<float>(out, FALSE, "G_BN1");
        // out = new Relu<float>(out, "G_R1");
        // // ======================= layer 2 ======================
        // out = new TransposedConvolutionLayer2D<float>(out, D*8, D*4, 4, 4, 2, 2, 1, 1, "G_TransposedConv2");
        // out = new BatchNormalizeLayer<float>(out, FALSE, "G_BN2");
        // out = new Relu<float>(out, "G_R2");
        // // ======================= layer 3 ======================
        // out = new TransposedConvolutionLayer2D<float>(out, D*4, D*2, 4, 4, 2, 2, 1, 1, "G_TransposedConv3");
        // out = new BatchNormalizeLayer<float>(out, FALSE, "G_BN3");
        // out = new Relu<float>(out, "G_R3");

        // // ======================= layer 4 ======================
        // out = new TransposedConvolutionLayer2D<float>(out, D*2, D, 4, 4, 2, 2, 1, 1, "G_TransposedConv4");
        // out = new BatchNormalizeLayer<float>(out, FALSE, "G_BN4");
        // out = new Relu<float>(out, "G_R4");

        // // ======================= layer 5 ====================
        // out = new TransposedConvolutionLayer2D<float>(out, D, 1, 4, 4, 2, 2, 1, 1, "G_TransposedConv5");
        // out = new Tanh<float>(out, "G_Tanh1");



        out = new Linear<float>(out, 100, 7 * 7 * 128);
        out = new ReShape<float>(out, 128, 7, 7, "G_Reshape1");
        out = new BatchNormalizeLayer<float>(out, TRUE, "G_BN1");
        out = new Relu<float>(out, "G_Relu1");

        out = new TransposedConvolutionLayer2D<float>(out, 128, 64, 4, 4, 2, 2, 1, FALSE, "G_TranspoedConv2");
        out = new BatchNormalizeLayer<float>(out, TRUE, "G_BN2");
        out = new Relu<float>(out, "G_Relu2");

        out = new TransposedConvolutionLayer2D<float>(out, 64, 1, 4, 4, 2, 2, 1, FALSE, "G_TransposedConv3");
        out = new BatchNormalizeLayer<float>(out, TRUE, "G_BN3");

        out = new Sigmoid<float>(out, "G_Tanh");
        out = new ReShape<float>(out, 1, 1, 28*28, "Img2Flat");

        this->AnalyzeGraph(out);
    }
};

