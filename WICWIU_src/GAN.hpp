#include "LossFunction/VanilaGeneratorLoss.hpp"
#include "LossFunction/VanilaDiscriminatorLoss.hpp"

#define REALLABEL 1
#define FAKELABEL -1

template<typename DTYPE> class GAN : public NeuralNetwork<DTYPE> {
private:
    NeuralNetwork<DTYPE> *m_pGenerator;
    NeuralNetwork<DTYPE> *m_pDiscriminator;

    Tensorholder<DTYPE> *m_pRealData;
    Tensorholder<DTYPE> *m_pLabel;

    LossFunction<DTYPE> *m_aGeneratorLossFunction;
    LossFunction<DTYPE> *m_aDiscriminatorLossFunction;

private:
    virtual void Delete();
    int AllocLabel(int plabelValue);

public:
    GAN();
    virtual ~GAN();

    NeuralNetwork<DTYPE>*               SetGenerator(NeuralNetwork<DTYPE> *pGen);
    NeuralNetwork<DTYPE>*               SetDiscriminator(NeuralNetwork<DTYPE> *pDiscLoss);

    Tensorholder<DTYPE>*                SetRealData(Tensorholder<DTYPE> *pRealData);
    Tensorholder<DTYPE>*                SetLabel(Tensorholder<DTYPE> *pLabel);

    void                                SetGANLossFunctions(LossFunction<DTYPE> *pGenLoss, LossFunction<DTYPE> *pDiscLoss);
    LossFunction<DTYPE>*                SetGeneratorLossFunction(LossFunction<DTYPE> *pGenLoss);
    LossFunction<DTYPE>*                SetDiscriminatorLossFunction(LossFunction<DTYPE> *pDiscLoss);

    void                                SetGANOptimizers(Optimizer<DTYPE> *pGenOpt, Optimizer<DTYPE> *pDiscOpt);
    Optimizer<DTYPE>*                   SetGeneratorOptimizer(Optimizer<DTYPE> *pGenOpt);
    Optimizer<DTYPE>*                   SetDiscriminatorOptimizer(Optimizer<DTYPE> *pDiscOpt);


    NeuralNetwork<DTYPE>*               GetGenerator();
    NeuralNetwork<DTYPE>*               GetDiscriminator();

    Tensorholder<DTYPE>*                GetRealData();
    Tensorholder<DTYPE>*                GetLabel();

    LossFunction<DTYPE>*                GetGeneratorLossFunction();
    LossFunction<DTYPE>*                GetDiscriminatorLossFunction();

    Optimizer<DTYPE>*                   GetGeneratorOptimizer();
    Optimizer<DTYPE>*                   GetDiscriminatorOptimizer();


    int                                 TrainGenerator();
    int                                 TrainDiscriminator();

    int                                 Test();

    int                                 TrainGeneratorOnCPU();
    int                                 TrainDiscriminatorOnCPU();

    int                                 TestOnCPU();

#ifdef __CUDNN__
    int                                 TrainGeneratorOnGPU();
    int                                 TrainDiscriminatorOnGPU();

    int                                 TestOnGPU();
#endif  // if __CUDNN__

    int                                 ResetGeneratorLossFunctionResult();
    int                                 ResetGeneratorLossFunctionGradient();

    int                                 ResetDiscriminatorLossFunctionResult();
    int                                 ResetDiscriminatorLossFunctionGradient();
};


template<typename DTYPE> void GAN<DTYPE>::Delete() {
    #ifdef __DEBUG__
    std::cout << "GAN<DTYPE>::Delete()" << '\n';
    #endif  // __DEBUG__`

    // if (m_pRealInput) {
    //     delete m_pRealInput;
    //     m_pRealInput = NULL;
    // }
    //
    // if (m_pLabel) {
    //     delete m_pLabel;
    //     m_pLabel = NULL;
    // }

    if (m_aGeneratorLossFunction) {
        delete m_aGeneratorLossFunction;
        m_aGeneratorLossFunction = NULL;
    }

    if (m_aDiscriminatorLossFunction) {
        delete m_aDiscriminatorLossFunction;
        m_aDiscriminatorLossFunction = NULL;
    }

    #ifdef __CUDNN__
        this->DeleteOnGPU();
    #endif  // if __CUDNN__
}

template<typename DTYPE> int GAN<DTYPE>::AllocLabel(int plabelValue){
    #ifdef __DEBUG__
    std::cout << "GAN<DTYPE>::AllocLabel(int plabel)" << '\n';
    #endif  // __DEBUG__
    m_pLabel->FeedTensor(Tensor<DTYPE>::Constants(m_pLabel->GetResult()->GetShape(), plabelValue, 1));

    return true;
}

template<typename DTYPE> GAN<DTYPE>::GAN() : NeuralNetwork<DTYPE>() {
    #ifdef __DEBUG__
    std::cout << "GAN<DTYPE>::GAN()" << '\n';
    #endif  // __DEBUG__

    m_pGenerator = NULL;
    m_pDiscriminator = NULL;

    m_pRealInput = NULL;
    m_pLabel = NULL;

    m_aGeneratorLossFunction = NULL;
    m_aDiscriminatorLossFunction = NULL;

}

template<typename DTYPE> GAN<DTYPE>::~GAN(){
    #ifdef __DEBUG__
    std::cout << "GAN<DTYPE>::~GAN()" << '\n';
    #endif  // __DEBUG__

    this->Delete();
}

// Setter
template<typename DTYPE> NeuralNetwork<DTYPE>* GAN<DTYPE>::SetGenerator(Generator<DTYPE> *pGen){
    m_pGenerator = pGen;
    return m_pGenerator;
}

template<typename DTYPE> NeuralNetwork<DTYPE>* GAN<DTYPE>::SetDiscriminator(LossFunction<DTYPE> *pDisc){
    m_pDiscriminator = pDisc;
    return m_pDiscriminator;
}

template<typename DTYPE> Tensorholder<DTYPE>* GAN<DTYPE>::SetRealData(Tensorholder<DTYPE> *pRealData){
    m_aRealData = pRealData;
    return m_aRealData;
}

template<typename DTYPE> Tensorholder<DTYPE>* GAN<DTYPE>::SetLabel(Tensorholder<DTYPE> *pLabel){
    m_pLabel = pLabel;
    return m_pLabel;
}

template<typename DTYPE> void GAN<DTYPE>::SetGANLossFunctions(LossFunction<DTYPE> *pGenLoss, LossFunction<DTYPE> *pDiscLoss){
    SetGeneratorLossFunction(pGenLoss);
    SetDiscriminatorLossFunction(pDiscLoss);
}

template<typename DTYPE> LossFunction<DTYPE>* GAN<DTYPE>::SetGeneratorLossFunction(LossFunction<DTYPE> *pGenLoss){
    m_aGeneratorLossFunction = pGenLoss;
    return m_aGeneratorLossFunction;
}

template<typename DTYPE> LossFunction<DTYPE>* GAN<DTYPE>::SetDiscriminatorLossFunction(LossFunction<DTYPE> *pDiscLoss){
    m_aDiscriminatorLossFunction = pDiscLoss;
    return m_aDiscriminatorLossFunction;
}

template<typename DTYPE> void GAN<DTYPE>::SetGANOptimizers(Optimizer<DTYPE> *pGenOpt, Optimizer<DTYPE> *pDiscOpt){
    SetGeneratorOptimizer(pGenOpt);
    SetDiscriminatorOptimizer(pDiscOpt);
}

template<typename DTYPE> Optimizer<DTYPE>* GAN<DTYPE>::SetGeneratorOptimizer(Optimizer<DTYPE> *pGenOpt){
    return m_pGenerator->SetOptimizer(pGenOpt);
}

template<typename DTYPE> Optimizer<DTYPE>* GAN<DTYPE>::SetDiscriminatorOptimizer(Optimizer<DTYPE> *pDiscOpt){
    return m_pDiscriminator->SetOptimizer(pDiscOpt)
}

// Getter
template<typename DTYPE> NeuralNetwork<DTYPE>* GAN<DTYPE>::GetGenerator(){
    return m_pGenerator;
}

template<typename DTYPE> NeuralNetwork<DTYPE>* GAN<DTYPE>::GetDiscriminator(){
    return m_pDiscriminator;
}

template<typename DTYPE> Tensorholder<DTYPE>* GAN<DTYPE>::GetRealData(){
    return m_pRealData;
}

template<typename DTYPE> Tensorholder<DTYPE>* GAN<DTYPE>::GetLabel(){
    return m_pLabel;
}

template<typename DTYPE> LossFunction<DTYPE>* GAN<DTYPE>::GetGeneratorLossFunction(){
    return m_aGeneratorLossFunction;
}

template<typename DTYPE> LossFunction<DTYPE>* GAN<DTYPE>::GetDiscriminatorLossFunction(){
    return m_aDiscriminatorLossFunction;
}

template<typename DTYPE> Optimizer<DTYPE>* GAN<DTYPE>::GetGeneratorOptimizer(){
    return m_pGenerator->GetOptimizer();
}

template<typename DTYPE> Optimizer<DTYPE>* GAN<DTYPE>::GetDiscriminatorOptimizer(){
    return m_pDiscriminator->GetOptimizer();
}

template<typename DTYPE> int GAN<DTYPE>::TrainGenerator(){
    if(this->GetDevice() = CPU) {
        TrainGeneratorOnCPU();
    } else if(this->GetDevice() = GPU) {
        TrainGeneratorOnGPU();
    } else return FALSE;
}

template<typename DTYPE> int GAN<DTYPE>::TrainDiscriminator(){
    if(this->GetDevice() = CPU) {
        TrainDiscriminatorOnCPU();
    } else if(this->GetDevice() = GPU) {
        TrainDiscriminatorOnGPU();
    } else return FALSE;
}

template<typename DTYPE> int GAN<DTYPE>::Test() {
  if(this->GetDevice() = CPU) {
      TestGeneratorOnCPU();
  } else if(this->GetDevice() = GPU) {
      TestGeneratorOnGPU();
  } else return FALSE:
}

template<typename DTYPE> int GAN<DTYPE>::TrainGeneratorOnCPU(){
    this->ResetResult();
    this->ResetGradient();
    this->ResetGeneratorLossFunctionResult();
    this->ResetGeneratorLossFunctionGradient();

    this->AllocLabel(REALLABEL);
    this->ForwardPropagate();
    m_aGeneratorLossFunction->ForwardPropagate();
    m_aGeneratorLossFunction->BackPropagate();
    this->BackPropagate();

    m_pGenerator->UpdateParameter();

    return TRUE;
}

template<typename DTYPE> int GAN<DTYPE>::TrainDiscriminatorOnCPU(){
    this->ResetResult();
    m_pDiscriminator->ResetGradient();
    this->ResetDiscriminatorLossFunctionResult();
    this->ResetDiscriminatorLossFunctionGradient();

    this->AllocLabel(REALLABEL);
    m_pGenerator->GetOutputContainer()->SetResult(m_pRealData->GetResult());
    m_pDiscriminator->ForwardPropagate();
    m_aDiscriminatorLossFunction->ForwardPropagate();
    m_aDiscriminatorLossFunction->BackPropagate();

    this->AllocLabel(FAKELABEL);
    this->ForwardPropagate();
    m_aDiscriminatorLossFunction->ForwardPropagate();
    m_aDiscriminatorLossFunction->BackPropagate();

    m_pDiscriminator->BackPropagate();

    m_pDiscriminator->UpdateParameter();

    return TRUE;
}

template<typename DTYPE> int GAN<DTYPE>::TestGeneratorOnCPU(){
    m_pGenerator->ResetResult();
    m_pGenerator->ForwardPropagate();

    return TRUE;
}


#ifdef __CUDNN__

template<typename DTYPE> int GAN<DTYPE>::TrainGeneratorOnGPU(){
    #ifdef __CUDNN__
        this->ResetResult();
        this->ResetGradient();
        this->ResetGeneratorLossFunctionResult();
        this->ResetGeneratorLossFunctionGradient();

        this->AllocLabel(REALLABEL);
        this->ForwardPropagateOnGPU();
        m_aGeneratorLossFunction->ForwardPropagateOnGPU();
        m_aGeneratorLossFunction->BackPropagateOnGPU();
        this->BackPropagateOnGPU();

        m_pGenerator->UpdateParameterOnGPU();

    #else  // __CUDNN__
        std::cout << "There is no GPU option!" << '\n';
        exit(-1);
    #endif  // __CUDNN__
}

template<typename DTYPE> int GAN<DTYPE>::TrainDiscriminatorOnGPU(){
    #ifdef __CUDNN__
        this->ResetResult();
        m_pDiscriminator->ResetGradient();
        this->ResetDiscriminatorLossFunctionResult();
        this->ResetDiscriminatorLossFunctionGradient();

        this->AllocLabel(REALLABEL);
        m_pGenerator->GetOutputContainer()->SetResult(this->GetRealInput());
        m_pDiscriminator->ForwardPropagateOnGPU();
        m_aDiscriminatorLossFunction->ForwardPropagateOnGPU();
        m_aDiscriminatorLossFunction->BackPropagateOnGPU();

        this->AllocLabel(FAKELABEL);
        this->ForwardPropagateOnGPU();
        m_aDiscriminatorLossFunction->ForwardPropagateOnGPU();
        m_aDiscriminatorLossFunction->BackPropagateOnGPU();

        m_pDiscriminator->BackPropagateOnGPU();

        m_pDiscriminator->UpdateParameterOnGPU();

    #else  // __CUDNN__
        std::cout << "There is no GPU option!" << '\n';
        exit(-1);
    #endif  // __CUDNN__
}

template<typename DTYPE> int GAN<DTYPE>::TestGeneratorOnGPU(){
    #ifdef __CUDNN__
        this->ResetResult();
        this->ForwardPropagateOnGPU();
    #else  // __CUDNN__
        std::cout << "There is no GPU option!" << '\n';
        exit(-1);
    #endif  // __CUDNN__

        return TRUE;
}
#endif

template<typename DTYPE> int GAN<DTYPE>::ResetGeneratorLossFunctionResult(){
    m_aGeneratorLossFunction->ResetResult();
    return TRUE;
}

template<typename DTYPE> int GAN<DTYPE>::ResetGeneratorLossFunctionGradient(){
    m_aGeneratorLossFunction->ResetGradient();
    return TRUE;
}

template<typename DTYPE> int GAN<DTYPE>::ResetDiscriminatorLossFunctionResult(){
    m_aDiscriminatorLossFunction->ResetResult();
    return TRUE;
}

template<typename DTYPE> int GAN<DTYPE>::ResetDiscriminatorLossFunctionGradient(){
    m_aDiscriminatorLossFunction->ResetGradient();
    return TRUE;
}
