#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.hpp"

template<typename DTYPE> class my_Discriminator : public NeuralNetwork<DTYPE> {
private:
public:
    my_Discriminator(Tensorholder<float> *x){
        Alloc(x);
    }

    virtual ~my_Discriminator() {
    }

    int Alloc(Tensorholder<float> *x){
        SetInput(x);
        
        Operator<float> *out = x;
        
        // ======================= layer 1 ======================
        out = new Linear<float>(out, 784, 512, TRUE, "D_L1");
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "D_BN1")
        out = new Relu<float>(out, "D_Tanh1");

       // ======================= layer 2 ======================
        out = new Linear<float>(out, 512, 256, TRUE, "D_L2");
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "D_BN2")
        out = new Relu<float>(out, "D_Tanh2");

        // ======================= layer 3 ======================
        out = new Linear<float>(out, 256, 1, TRUE, "D_L3");
        out = new BatchNormalizeLayer<DTYPE>(out, TRUE, "D_BN3")
        out = new Relu<float>(out, "D_Tanh3");
        
        AnalyzeGraph(out);
    }
}
