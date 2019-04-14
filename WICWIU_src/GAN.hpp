#include "LossFunction/VanilaGeneratorLoss.hpp"
#include "LossFunction/VanilaDiscriminatorLoss.hpp"
#include "NeuralNetwork.hpp"

#define REALLABEL 1.f
#define FAKELABEL 0.f

template<typename DTYPE> class GAN : public NeuralNetwork<DTYPE> {
private:
    NeuralNetwork<DTYPE> *m_pGenerator;
    NeuralNetwork<DTYPE> *m_pDiscriminator;

    Tensorholder<DTYPE> *m_pRealData;
    Tensorholder<DTYPE> *m_pLabel;

    LossFunction<DTYPE> *m_pGeneratorLossFunction;
    LossFunction<DTYPE> *m_pDiscriminatorLossFunction;

private:
    int AllocLabel(DTYPE plabelValue);
    int AllocRealDataOnGenerator();

#ifdef __CUDNN__
    int AllocLabelOnGPU(DTYPE plabelValue);
    int AllocRealDataOnGeneratorOnGPU();
#endif

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

    int                                 TrainGeneratorOnGPU();
    int                                 TrainDiscriminatorOnGPU();

    int                                 TestOnGPU();

    int                                 ResetParameterGradient();

    int                                 ResetGeneratorLossFunctionResult();
    int                                 ResetGeneratorLossFunctionGradient();

    int                                 ResetDiscriminatorLossFunctionResult();
    int                                 ResetDiscriminatorLossFunctionGradient();

#ifdef __CUDNN__
    void                                SetDeviceGPUOnGAN(unsigned int idOfDevice);
#endif  // __CUDNN__
};

template<typename DTYPE> int GAN<DTYPE>::AllocLabel(DTYPE plabelValue){
    #ifdef __DEBUG__
    std::cout << "GAN<DTYPE>::AllocLabel(int plabel)" << '\n';
    #endif  // __DEBUG__

    int m_timesize = m_pLabel->GetResult()->GetDim(0);
    int m_batchsize = m_pLabel->GetResult()->GetDim(1);
    int m_channelsize = m_pLabel->GetResult()->GetDim(2);
    int m_rowsize = m_pLabel->GetResult()->GetDim(3);
    int m_colsize = m_pLabel->GetResult()->GetDim(4);

    m_pLabel->FeedTensor(Tensor<DTYPE>::Constants(m_timesize, m_batchsize, m_channelsize, m_rowsize, m_colsize, plabelValue));

    return true;
}

template<typename DTYPE> int GAN<DTYPE>::AllocRealDataOnGenerator(){
    Tensor<DTYPE> * temp = new Tensor<DTYPE>(m_pRealData->GetResult());
    m_pGenerator->GetResult()->SetResult(temp);

    return true;
}

template<typename DTYPE> int GAN<DTYPE>::AllocLabelOnGPU(DTYPE plabelValue){
    #ifdef __DEBUG__
    std::cout << "GAN<DTYPE>::AllocLabel(int plabel)" << '\n';
    #endif  // __DEBUG__

    int m_timesize = m_pLabel->GetResult()->GetDim(0);
    int m_batchsize = m_pLabel->GetResult()->GetDim(1);
    int m_channelsize = m_pLabel->GetResult()->GetDim(2);
    int m_rowsize = m_pLabel->GetResult()->GetDim(3);
    int m_colsize = m_pLabel->GetResult()->GetDim(4);

    m_pLabel->FeedTensor(Tensor<DTYPE>::Constants(m_timesize, m_batchsize, m_channelsize, m_rowsize, m_colsize, plabelValue));
    m_pLabel->GetTensor()->SetDeviceGPU(this->GetDeviceID());

    return true;
}

template<typename DTYPE> int GAN<DTYPE>::AllocRealDataOnGeneratorOnGPU(){
    Tensor<DTYPE> * temp = new Tensor<DTYPE>(m_pRealData->GetResult());
    temp->SetDeviceGPU(this->GetDeviceID());
    m_pGenerator->GetResult()->SetResult(temp);

    return true;
}

template<typename DTYPE> GAN<DTYPE>::GAN() : NeuralNetwork<DTYPE>() {
    #ifdef __DEBUG__
    std::cout << "GAN<DTYPE>::GAN()" << '\n';
    #endif  // __DEBUG__

    m_pGenerator = NULL;
    m_pDiscriminator = NULL;

    m_pRealData = NULL;
    m_pLabel = NULL;

    m_pGeneratorLossFunction = NULL;
    m_pDiscriminatorLossFunction = NULL;

}

template<typename DTYPE> GAN<DTYPE>::~GAN(){
    #ifdef __DEBUG__
    std::cout << "GAN<DTYPE>::~GAN()" << '\n';
    #endif  // __DEBUG__
}

// Setter
template<typename DTYPE> NeuralNetwork<DTYPE>* GAN<DTYPE>::SetGenerator(NeuralNetwork<DTYPE> *pGen){
    m_pGenerator = pGen;
    return m_pGenerator;
}

template<typename DTYPE> NeuralNetwork<DTYPE>* GAN<DTYPE>::SetDiscriminator(NeuralNetwork<DTYPE> *pDisc){
    m_pDiscriminator = pDisc;
    return m_pDiscriminator;
}

template<typename DTYPE> Tensorholder<DTYPE>* GAN<DTYPE>::SetRealData(Tensorholder<DTYPE> *pRealData){
    m_pRealData = pRealData;
    return m_pRealData;
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
    m_pGeneratorLossFunction = pGenLoss;
    return m_pGenerator->SetLossFunction(pGenLoss);
}

template<typename DTYPE> LossFunction<DTYPE>* GAN<DTYPE>::SetDiscriminatorLossFunction(LossFunction<DTYPE> *pDiscLoss){
    m_pDiscriminatorLossFunction = pDiscLoss;
    return m_pDiscriminator->SetLossFunction(pDiscLoss);
}

template<typename DTYPE> void GAN<DTYPE>::SetGANOptimizers(Optimizer<DTYPE> *pGenOpt, Optimizer<DTYPE> *pDiscOpt){
    SetGeneratorOptimizer(pGenOpt);
    SetDiscriminatorOptimizer(pDiscOpt);
}

template<typename DTYPE> Optimizer<DTYPE>* GAN<DTYPE>::SetGeneratorOptimizer(Optimizer<DTYPE> *pGenOpt){
    return m_pGenerator->SetOptimizer(pGenOpt);
}

template<typename DTYPE> Optimizer<DTYPE>* GAN<DTYPE>::SetDiscriminatorOptimizer(Optimizer<DTYPE> *pDiscOpt){
    return m_pDiscriminator->SetOptimizer(pDiscOpt);
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
    return m_pGeneratorLossFunction;
}

template<typename DTYPE> LossFunction<DTYPE>* GAN<DTYPE>::GetDiscriminatorLossFunction(){
    return m_pDiscriminatorLossFunction;
}

template<typename DTYPE> Optimizer<DTYPE>* GAN<DTYPE>::GetGeneratorOptimizer(){
    return m_pGenerator->GetOptimizer();
}

template<typename DTYPE> Optimizer<DTYPE>* GAN<DTYPE>::GetDiscriminatorOptimizer(){
    return m_pDiscriminator->GetOptimizer();
}

template<typename DTYPE> int GAN<DTYPE>::TrainGenerator(){
    if(this->GetDevice() == CPU) {
        this->TrainGeneratorOnCPU();
    } else if(this->GetDevice() == GPU) {
        this->TrainGeneratorOnGPU();
    } else return FALSE;
}

template<typename DTYPE> int GAN<DTYPE>::TrainDiscriminator(){
    if(this->GetDevice() == CPU) {
        this->TrainDiscriminatorOnCPU();
    } else if(this->GetDevice() == GPU) {
        this->TrainDiscriminatorOnGPU();
    } else return FALSE;
}

template<typename DTYPE> int GAN<DTYPE>::Test() {
  if(this->GetDevice() == CPU) {
      this->TestOnCPU();
  } else if(this->GetDevice() == GPU) {
      this->TestOnGPU();
  } else return FALSE;
}

template<typename DTYPE> int GAN<DTYPE>::TrainGeneratorOnCPU(){
    this->ResetResult();
    this->ResetGradient();
    this->ResetGeneratorLossFunctionResult();
    this->ResetGeneratorLossFunctionGradient();

    this->ForwardPropagate();
    m_pGeneratorLossFunction->ForwardPropagate();
    m_pGeneratorLossFunction->BackPropagate();

    this->GetGeneratorOptimizer()->UpdateParameter();

    return TRUE;
}

template<typename DTYPE> int GAN<DTYPE>::TrainDiscriminatorOnCPU(){
    this->ResetResult();
    m_pDiscriminator->ResetGradient();

    this->ResetDiscriminatorLossFunctionResult();
    this->ResetDiscriminatorLossFunctionGradient();

    this->AllocLabel(REALLABEL);
    this->AllocRealDataOnGenerator();
    m_pDiscriminator->ForwardPropagate();
    // for(int ba = 0; ba < 64; ba++){
    //     std::cout << "Dis Dis Forward ba = : " << ba << " ===== " << (*m_pDiscriminator->GetResult()->GetResult())[Index5D(m_pDiscriminator->GetResult()->GetResult()->GetShape(), 0, ba, 0, 0, 0)] << '\n';
    // }
    // std::cout << '\n';

    m_pDiscriminatorLossFunction->ForwardPropagate();
    m_pDiscriminatorLossFunction->BackPropagate();

    this->AllocLabel(FAKELABEL);
    this->ForwardPropagate();
    // for(int ba = 0; ba < 64; ba++){
    //     std::cout << "Dis Gen Forward ba = : " << ba << " ===== " << (*m_pDiscriminator->GetResult()->GetResult())[Index5D(m_pDiscriminator->GetResult()->GetResult()->GetShape(), 0, ba, 0, 0, 0)] << '\n';
    // }
    // std::cout << '\n';

    m_pDiscriminatorLossFunction->ForwardPropagate();
    m_pDiscriminatorLossFunction->BackPropagate();

    m_pDiscriminator->BackPropagate();

    this->GetDiscriminatorOptimizer()->UpdateParameter();

    return TRUE;
}

template<typename DTYPE> int GAN<DTYPE>::TestOnCPU(){
    m_pGenerator->ResetResult();
    m_pGenerator->ForwardPropagate();

    return TRUE;
}


template<typename DTYPE> int GAN<DTYPE>::TrainGeneratorOnGPU(){
    #ifdef __CUDNN__
        this->ResetResult();
        this->ResetGradient();
        this->ResetGeneratorLossFunctionResult();
        this->ResetGeneratorLossFunctionGradient();

        this->ForwardPropagateOnGPU();
        std::cout << "Generator Fake Data Forward : " << (*m_pDiscriminator->GetResult()->GetResult())[Index5D(m_pDiscriminator->GetResult()->GetResult()->GetShape(), 0, 0, 0, 0, 0)];
        m_pGeneratorLossFunction->ForwardPropagateOnGPU();
        m_pGeneratorLossFunction->BackPropagateOnGPU();
        this->BackPropagateOnGPU();

        this->GetGeneratorOptimizer()->UpdateParameterOnGPU();

    #else  // __CUDNN__
        std::cout << "There is no GPU option!" << '\n';
        exit(-1);
    #endif  // __CUDNN__

    return TRUE;
}

template<typename DTYPE> int GAN<DTYPE>::TrainDiscriminatorOnGPU(){
    #ifdef __CUDNN__
        this->ResetResult();
        m_pDiscriminator->ResetGradient();
        this->ResetDiscriminatorLossFunctionResult();
        this->ResetDiscriminatorLossFunctionGradient();

        this->AllocLabelOnGPU(REALLABEL);
        this->AllocRealDataOnGeneratorOnGPU();
        m_pDiscriminator->ForwardPropagateOnGPU();
        std::cout << "Discriminator True Data Forward : " << (*m_pDiscriminator->GetResult()->GetResult())[Index5D(m_pDiscriminator->GetResult()->GetResult()->GetShape(), 0, 0, 0, 0, 0)];
        m_pDiscriminatorLossFunction->ForwardPropagateOnGPU();
        m_pDiscriminatorLossFunction->BackPropagateOnGPU();
        m_pDiscriminator->BackPropagateOnGPU();

        this->ResetResult();
        m_pDiscriminator->ResetGradient();
        this->ResetDiscriminatorLossFunctionResult();
        this->ResetDiscriminatorLossFunctionGradient();

        this->AllocLabelOnGPU(FAKELABEL);
        this->ForwardPropagateOnGPU();
        std::cout << "Discriminator Fake Data Forward : " << (*m_pDiscriminator->GetResult()->GetResult())[Index5D(m_pDiscriminator->GetResult()->GetResult()->GetShape(), 0, 0, 0, 0, 0)];
        m_pDiscriminatorLossFunction->ForwardPropagateOnGPU();
        m_pDiscriminatorLossFunction->BackPropagateOnGPU();

        m_pDiscriminator->BackPropagateOnGPU();

        this->GetDiscriminatorOptimizer()->UpdateParameterOnGPU();

    #else  // __CUDNN__
        std::cout << "There is no GPU option!" << '\n';
        exit(-1);
    #endif  // __CUDNN__

    return TRUE;
}

template<typename DTYPE> int GAN<DTYPE>::TestOnGPU(){
    #ifdef __CUDNN__
        this->ResetResult();
        this->ForwardPropagateOnGPU();
    #else  // __CUDNN__
        std::cout << "There is no GPU option!" << '\n';
        exit(-1);
    #endif  // __CUDNN__

        return TRUE;
}

// Need to override
template<typename DTYPE> int GAN<DTYPE>::ResetParameterGradient(){
    this->GetGeneratorOptimizer()->ResetParameterGradient();
    this->GetDiscriminatorOptimizer()->ResetParameterGradient();
    return TRUE;
}

template<typename DTYPE> int GAN<DTYPE>::ResetGeneratorLossFunctionResult(){
    m_pGeneratorLossFunction->ResetResult();
    return TRUE;
}

template<typename DTYPE> int GAN<DTYPE>::ResetGeneratorLossFunctionGradient(){
    m_pGeneratorLossFunction->ResetGradient();
    return TRUE;
}

template<typename DTYPE> int GAN<DTYPE>::ResetDiscriminatorLossFunctionResult(){
    m_pDiscriminatorLossFunction->ResetResult();
    return TRUE;
}

template<typename DTYPE> int GAN<DTYPE>::ResetDiscriminatorLossFunctionGradient(){
    m_pDiscriminatorLossFunction->ResetGradient();
    return TRUE;
}

#ifdef __CUDNN__

template<typename DTYPE> void GAN<DTYPE>::SetDeviceGPUOnGAN(unsigned int idOfDevice) {
    this->SetDeviceGPU(idOfDevice);
    m_pGenerator->SetDeviceGPU(idOfDevice);
    m_pDiscriminator->SetDeviceGPU(idOfDevice);
}

#endif  // __CUDNN__
