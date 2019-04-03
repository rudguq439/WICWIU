#include <iostream>
#include <string>

#include "../../../WICWIU_src/NeuralNetwork.hpp"

template<typename DTYPE> class my_Generator : public NeuralNetwork<DTYPE> {
private:
public:
    my_Generator(Tensorholder<float> *z){
        Alloc(z);
    }

    virtual ~my_Generator() {
    }

    int Alloc(Tensorholder<float> *z){
        // SetInput(z);
        // 
        // Operator<float> * out = NULL;
        //
        // AnalayzeGraph(out);
    }
}
