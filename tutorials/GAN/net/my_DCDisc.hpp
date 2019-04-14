#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.hpp"
using namespace std;

template<typename DTYPE> class my_DCDisc : public NeuralNetwork<DTYPE> {
private:
public:
    my_DCDisc(Operator<float> *x){
        Alloc(x);
    }

    virtual ~my_DCDisc() {
    }

    int Alloc(Operator<float> *x){
        this->SetInput(x);
                
        Operator<float> *out = x;

        // // for 64 x 64
        // // ======================= layer 1 ======================
        // out = new ConvolutionLayer2D<float>(out, 1, D, 4, 4, 2, 2, 1, 1, "D_Conv1");
        // out = new Relu<float>(out, "D_R1");

        // // ======================= layer 2 ======================
        // out = new ConvolutionLayer2D<float>(out, D, D*2, 4, 4, 2, 2, 1, 1, "D_Conv1");
        // out = new BatchNormalizeLayer<float>(out, FALSE, "G_BN1");
        // out = new Relu<float>(out, "D_R2");

        // // ======================= layer 3 ======================
        // out = new ConvolutionLayer2D<float>(out, D*2, D*4, 4, 4, 2, 2, 1, 1, "D_Conv1");
        // out = new BatchNormalizeLayer<float>(out, FALSE, "G_BN1");
        // out = new Relu<float>(out, "D_R3");

        // // ======================= layer 4 ======================
        // out = new ConvolutionLayer2D<float>(out, D*4, D*8, 4, 4, 2, 2, 1, 1, "D_Conv1");
        // out = new BatchNormalizeLayer<float>(out, FALSE, "G_BN1");
        // out = new Relu<float>(out, "D_R4");

        // // ======================= layer 5 ======================
        // out = new ConvolutionLayer2D<float>(out, D*8, 1, 4, 4, 1, 1, 0, 0, "D_Conv1");

        // out = new Sigmoid<float>(out, "D_Sigmo");

        out = new ReShape<float>(out, 1, 28, 28, "Flat2Img");

        out = new ConvolutionLayer2D<float>(out, 1, 64, 4, 4, 2, 2, 1, 1, "D_Conv1");
        out = new BatchNormalizeLayer<float>(out, TRUE, "D_BN1");
        out = new Relu<float>(out, "D_Relu1");

        out = new ConvolutionLayer2D<float>(out, 64, 128, 4, 4, 2, 2, 1, 1, "D_Conv2");
        out = new BatchNormalizeLayer<float>(out, TRUE, "D_BN2");
        out = new Relu<float>(out, "D_Relu2");

        out = new ReShape<float>(out, 1, 1, 128 * 7 * 7, "D_ReShape3");
        out = new Linear<float>(out, 128 * 7 * 7, 1);
        out = new Sigmoid<float>(out, "D_Sigmoid3");

        this->AnalyzeGraph(out);
    }
};