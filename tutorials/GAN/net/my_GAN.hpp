#include <iostream>
#include <string>

#include "../../../WICWIU_src/GAN.hpp"

template<typename DTYPE> class my_GAN : public GAN<DTYPE> {
private:
public:
    my_GAN(Tensorholder<float> *z, Tensorholder<float> *x, Tensorholder<float> *label){
        Alloc(z, x, label);
    }

    virtual ~my_Gan() {
    }

    int Alloc(Tensorholder<float> *z, Tensorholder<float> *x, Tensorholder<float> *label){
        setInput(z, x, label);

        m_pGenerator = new my_Generator<float>(z);
        m_pDiscriminator = new my_Discriminator<float>(m_pGenerator);
        AnalyzeGraph(m_pDiscriminator);

        this->SetRealInput(x);
        this->SetLabel(label);

        // ======================= Select LossFunction ===================
        SetGANLossFunctions(new VanilaGeneratorLoss<float>(m_pDiscriminator, this->GetLabel()), new VanilaDiscriminatorLoss<float>(m_pDiscriminator, this->GetLabel()));

        // ======================= Select Optimizer ===================
        SetGANOptimizers(new RMSPropOptimizer<float>(m_pGenerator->GetParameter(), 0.01, 0.9, MAXIMIZE), new RMSPropOptimizer<float>(m_pDiscriminator->GetParameter(), 0.01, 0.9, MINIMIZE));
    }
}
