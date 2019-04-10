#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.hpp"

template<typename DTYPE> class my_Generator : public NeuralNetwork<DTYPE> {
private:
public:
    my_Generator(Operator<float> *z){
        Alloc(z);
    }

    virtual ~my_Generator() {
    }

    int Alloc(Operator<float> *z){
        this->SetInput(z);

        Operator<float> *out = z;

        // ======================= layer 1 ======================
        out = new Linear<float>(out, 100, 256, TRUE, "G_L1");
        // out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "G_BN1")
        out = new Relu<float>(out, "G_Tanh1");

       // ======================= layer 2 ======================
        out = new Linear<float>(out, 256, 512, TRUE, "G_L2");
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "G_BN2");
        out = new Relu<float>(out, "G_Tanh2");

        // ======================= layer 3 ======================
        out = new Linear<float>(out, 512, 784, TRUE, "G_L3");
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "G_BN3");
        out = new Relu<float>(out, "G_Tanh3");

        this->AnalyzeGraph(out);
    }
};
