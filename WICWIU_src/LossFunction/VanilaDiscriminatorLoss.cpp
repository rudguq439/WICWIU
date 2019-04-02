#ifndef DISCRIMINATORLOSS_H_
#define DISCRIMINATORLOSS_H_    value

#include "../LossFunction.hpp"

template<typename DTYPE>
class DiscriminatorLoss : public LossFunction<DTYPE>{
public:
    DiscriminatorLoss(Operator<DTYPE> *pOperator, Operator<DTYPE> *pLabel, std::string pName) : LossFunction<DTYPE>(pOperator, pLabel, pName){
        #ifdef __DEBUG__
        std::cout << "DiscriminatorLoss::DiscriminatorLoss(Operator<DTYPE> *, MetaParameter *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pOperator);
    }

    virtual ~DiscriminatorLoss(){
        #ifdef __DEBUG__
        std::cout << "DiscriminatorLoss::~DiscriminatorLoss()" << '\n';
        #endif  // __DEBUG__
    }

    virtual int Alloc(Operator<DTYPE> *pOperator){
        #ifdef __DEBUG__
        std::cout << "DiscriminatorLoss::Alloc(Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        Operator<DTYPE> *pInput = pOperator;

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, 1, 1, 1));

        return TRUE;
    }

    void Delete(){}

    Tensor<DTYPE>* ForwardPropagate(int pTime = 0){
        Tensor<DTYPE> *input = this->GetTensor();
        Tensor<DTYPE> *label = this->GetLabel()->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int batchsize   = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();
        int capacity = channelsize * rowsize * colsize;
        int ti = pTime;


        int start = 0;
        int end   = 0;

        // generator를 넣은 D의 계산
        for (int ba = 0; ba < batchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            for (int i = start; i < end; i++) {
                // -mean( log(D(x)) - log(D(G(z))) )
                // For each, -log(D(x)) + log(D(G(z)))
                // Label = +1 --> Real input for D, so +1*logD(x)
                // Label = -1 --> Fake input for D, so -1*logD(G(z))
                // Add to result. So result = logD(x) - logD(G(z))
                (*result)[i] += -1 * (*label)[i] * log((*input)[i]);
            }
            
        }
        return result;
    }


    Tensor<DTYPE>* BackPropagate(int pTime = 0) {
        Tensor<DTYPE> *input       = this->GetTensor();
        Tensor<DTYPE> *label       = this->GetLabel()->GetResult();
        Tensor<DTYPE> *input_delta = this->GetOperator()->GetDelta();

        int batchsize = input->GetBatchSize();
        int colsize   = input->GetColSize();
        int rowsize   = input->GetRowSize();
        int colsize   = input->GetColSize();
        int capacity  = channelsize * rowsize * colsize;

        int start = 0;
        int end   = 0;

        int ti = pTime;

        for (int ba = 0; ba < batchsize; ba++) {
            start = (ti * batchsize + ba) * capacity;
            end   = start + capacity;

            for (int i = start; i < end; i++) {
                // ( - 1.0 / D(x) ) + ( 1.0 / D(G(z)) )
                // Label = +1 --> Real input for D, so +1 * logD(x)
                // Label = -1 --> Fake input for D, so -1 * logD(G(z))
                // Add to result. - 넣은 이유는, 위의 식을 따라가기 위함
                (*input_delta)[i] += ( (*label)[i] * -1.0 / (*input)[i] );
            }

        }

    }

#ifdef __CUDNN__

    Tensor<DTYPE>* ForwardPropagateOnGPU(int pTime = 0) {
        this->ForwardPropagate();
        return NULL;
    }


    Tensor<DTYPE>* BackPropagateOnGPU(int pTime = 0) {
        this->BackPropagate();
        return NULL;
    }

#endif
};

#endif
