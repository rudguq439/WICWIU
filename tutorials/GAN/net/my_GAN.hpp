#include <iostream>
#include <string>

#include "../../../WICWIU_src/GAN.hpp"
#include "my_DCDisc.hpp"
#include "my_DCGen.hpp"

template<typename DTYPE> class my_GAN : public GAN<DTYPE> {
private:
public:
    my_GAN(Tensorholder<float> *z, Tensorholder<float> *x, Tensorholder<float> *label){
        Alloc(z, x, label);
    }

    virtual ~my_GAN() {
    }

    int Alloc(Tensorholder<float> *z, Tensorholder<float> *x, Tensorholder<float> *label){
        this->SetInput(3, z, x, label);

        // this->GetGenerator() = new my_Generator<float>(z);
        // this->GetDiscriminator() = new my_Discriminator<float>(this->GetGenerator());
        // this->AnalyzeGraph(this->GetDiscriminator());

        // Vanlia GAN
        // this->SetGenerator(new my_Generator<float>(z));
        // this->SetDiscriminator(new my_Discriminator<float>(this->GetGenerator()));

        // DCGAN
        this->SetGenerator(new my_DCGen<float>(z)); 
        this->SetDiscriminator(new my_DCDisc<float>(this->GetGenerator()));


        this->AnalyzeGraph(this->GetDiscriminator());

        this->SetRealData(x);
        this->SetLabel(label);

        // ======================= Select LossFunction ===================
        this->SetGANLossFunctions(new VanilaGeneratorLoss<float>(this->GetDiscriminator(), this->GetLabel(), "VanilaGeneratorLoss"), new VanilaDiscriminatorLoss<float>(this->GetDiscriminator(), this->GetLabel(), "VanilaDiscriminatorLoss"));

        // ======================= Select Optimizer ===================
        // this->SetGANOptimizers(new GradientDescentOptimizer<float>(this->GetGenerator()->GetParameter(), 0.000001, MINIMIZE), new GradientDescentOptimizer<float>(this->GetDiscriminator()->GetParameter(), 0.000001, MAXIMIZE));
        // this->SetGANOptimizers(new RMSPropOptimizer<float>(this->GetGenerator()->GetParameter(), 0.0001, 0.9, 1e-08, FALSE, MINIMIZE), new RMSPropOptimizer<float>(this->GetDiscriminator()->GetParameter(), 0.0001, 0.9, 1e-08, FALSE, MAXIMIZE));
        this->SetGANOptimizers(new AdamOptimizer<float>(this->GetGenerator()->GetParameter(), 0.0001, 0.5, 0.999, 1e-08, MINIMIZE), new AdamOptimizer<float>(this->GetDiscriminator()->GetParameter(), 0.00005, 0.5, 0.999, 1e-08, MINIMIZE));
    }
};

