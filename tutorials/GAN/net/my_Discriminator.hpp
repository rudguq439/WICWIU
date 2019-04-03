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
        // SetInput(x);
        //
        // Operator<float> *out = NULL;
        //
        // AnalyzeGraph(out);
    }
}
