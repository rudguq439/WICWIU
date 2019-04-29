#include <iostream>
#include <string>

#include "../../../WICWIU_src/GAN.hpp"
#include "my_BEGen.hpp"
#include "my_BEDisc.hpp"

template<typename DTYPE> class my_BEGAN : public GAN<DTYPE> {
private:
public:
    my_BEGAN(Tensorholder<float> *z, Tensorholder<float> *x, Tensorholder<float> *label){
        Alloc(z, x, label);
    }

    virtual ~my_BEGAN() {

    }

    int Alloc(Tensorholder<float> *z, Tensorholder<float> *x, Tensorholder<float> *label){
        this->SetInput(3, z, x, label);

        // this->GetGenerator() = new my_Generator<float>(z);
        // this->GetDiscriminator() = new my_Discriminator<float>(this->GetGenerator());
        // this->AnalyzeGraph(this->GetDiscriminator());

        // this->SetGenerator(new my_Generator<float>(z));
        // this->SetSwitchInput(new SwitchInput<float>(this->GetGenerator(), x));
        // this->SetDiscriminator(new my_Discriminator<float>(this->GetSwitchInput()));
        // this->AnalyzeGraph(this->GetDiscriminator());
        //
        // this->SetLabel(label);
        //
        // // ======================= Select LossFunction ===================
        // this->SetGANLossFunctions(new VanillaGeneratorLoss<float>(this->GetDiscriminator(), this->GetLabel(), "VanillaGeneratorLoss"), new VanillaDiscriminatorLoss<float>(this->GetDiscriminator(), this->GetLabel(), "VanillaDiscriminatorLoss"));
        //
        // // ======================= Select Optimizer ===================
        // // this->SetGANOptimizers(new GradientDescentOptimizer<float>(this->GetGenerator()->GetParameter(), 0.000001, MINIMIZE), new GradientDescentOptimizer<float>(this->GetDiscriminator()->GetParameter(), 0.000001, MAXIMIZE));
        // // this->SetGANOptimizers(new RMSPropOptimizer<float>(this->GetGenerator()->GetParameter(), 0.0001, 0.9, 1e-08, FALSE, MINIMIZE), new RMSPropOptimizer<float>(this->GetDiscriminator()->GetParameter(), 0.0001, 0.9, 1e-08, FALSE, MAXIMIZE));
        // this->SetGANOptimizers(new AdamOptimizer<float>(this->GetGenerator()->GetParameter(), 0.0002, 0.5, 0.999, 1e-08, MAXIMIZE), new AdamOptimizer<float>(this->GetDiscriminator()->GetParameter(), 0.0002, 0.5, 0.999, 1e-08, MINIMIZE));

        this->SetGenerator(new my_BEGen<float>(z));
        this->SetSwitchInput(new SwitchInput<float>(this->GetGenerator(), x));
        this->SetDiscriminator(new my_BEDisc<float>(this->GetSwitchInput()));
        this->AnalyzeGraph(this->GetDiscriminator());

        this->SetLabel(label);

        // ======================= Select LossFunction ===================
        this->SetGANLossFunctions(new WGANGeneratorLoss<float>(this->GetDiscriminator(), this->GetLabel(), "WGANGeneratorLoss"), new WGANDiscriminatorLoss<float>(this->GetDiscriminator(), this->GetLabel(), "WGANDiscriminatorLoss"));
        // this->SetGANLossFunctions(new VanillaGeneratorLoss<float>(this->GetDiscriminator(), this->GetLabel(), "WGANGeneratorLoss"), new VanillaDiscriminatorLoss<float>(this->GetDiscriminator(), this->GetLabel(), "WGANDiscriminatorLoss"));

        // ======================= Select Optimizer ===================
        // this->SetGANOptimizers(new GradientDescentOptimizer<float>(this->GetGenerator()->GetParameter(), 0.000001, MINIMIZE), new GradientDescentOptimizer<float>(this->GetDiscriminator()->GetParameter(), 0.000001, MAXIMIZE));
        // this->SetGANOptimizers(new RMSPropOptimizer<float>(this->GetGenerator()->GetParameter(), 0.0001, 0.9, 1e-08, FALSE, MINIMIZE), new RMSPropOptimizer<float>(this->GetDiscriminator()->GetParameter(), 0.0001, 0.9, 1e-08, FALSE, MAXIMIZE));
        this->SetGANOptimizers(new AdamOptimizer<float>(this->GetGenerator()->GetParameter(), 0.001, 0.5, 0.999, 1e-08, MINIMIZE), new AdamOptimizer<float>(this->GetDiscriminator()->GetParameter(), 0.0002, 0.5, 0.999, 1e-08, MINIMIZE));
    }
};
