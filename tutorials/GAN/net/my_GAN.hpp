#include <iostream>
#include <string>

#include "../../../WICWIU_src/GAN.hpp"

template<typename DTYPE> class my_GAN : public GAN<DTYPE> {
private:
public:
    my_GAN(Tensorholder<float> *z, Tensorholder<float> *x, Tensorholder<float> *label){
        Alloc(z, x, label);
    }

    virtual ~my_GAN() {
    }

    int Alloc(Tensorholder<float> *z, Tensorholder<float> *x, Tensorholder<float> *label){
        this->setInput(z, x, label);

        this->GetGenerator() = new my_Generator<float>(z);
        this->GetDiscriminator() = new my_Discriminator<float>(this->GetGenerator());
        this->AnalyzeGraph(this->GetDiscriminator());

        this->SetRealInput(x);
        this->SetLabel(label);

        // ======================= Select LossFunction ===================
        this->SetGANLossFunctions(new VanilaGeneratorLoss<float>(this->GetDiscriminator(), this->GetLabel()), new VanilaDiscriminatorLoss<float>(this->GetDiscriminator(), this->GetLabel()));

        // ======================= Select Optimizer ===================
        this->SetGANOptimizers(new RMSPropOptimizer<float>(this->GetGenerator()->GetParameter(), 0.01, 0.9, MAXIMIZE), new RMSPropOptimizer<float>(this->GetDiscriminator()->GetParameter(), 0.01, 0.9, MINIMIZE));
    }
};
