#ifndef GENERATORLOSS_H_
#define GENERATORLOSS_H_    value

#include "../LossFunction.hpp"

template<typename DTYPE>
class GeneratorLoss : public LossFunction<DTYPE>{
public:
    GeneratorLoss(Operator<DTYPE> *pOperator, std::string pName) : LossFunction<DTYPE>(pOperator, pLabel, pName){
        this->Alloc(pOperator);
    }

    virtual ~GeneratorLoss(){}

    virtual int Alloc(Operator<DTYPE> *pOperator){

        Operator<DTYPE> *pInput = pOperator;

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();

    
        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, 1, 1, 1));

        return TRUE;
    }

    Tensor<DTYPE>* ForwardPropagate(int pTime = 0) {
        Tensor<DTYPE> *input  = this->GetTensor();
        Tensor<DTYPE> *label  = this->GetLabel()->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int batchsize = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize = input->GetRowSize();
        int colsize = input->GetColSize();

        int capacity = channelsize * rowsize * colsize;

        int ti = pTime;


        int start = 0;
        int end   = 0;

        DTYPE tmp = 0.f;

        for (int ba = 0; ba < batchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            for (int i = start; i < end; i++) {
                (*result)[i] += - log((*input)[i]);
            }
        }
        return result;
    }


    Tensor<DTYPE>* BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *input       = this->GetTensor();
        Tensor<DTYPE> *input_delta = this->GetOperator()->GetDelta();

        int batchsize = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();
        int capacity    = channelsize * rowsize * colsize;

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            for (int i = start; i < end; i++) {
                (*input_delta)[i] += 1.0 / (*input)[i]
            }
        }

        return NULL;
    }

#ifdef __CUDNN__

    Tensor<DTYPE>* ForwardPropagateOnGPU(int pTime = 0) {
        this->ForwardPropagate();
        return NULL;ㅡxxe
    }

    Tensor<DTYPE>* BackPropagateOnGPU(int pTime = 0) {
        this->BackPropagate();
        return NULL;
    }

#endif
};

#endif
