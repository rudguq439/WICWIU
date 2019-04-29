#ifndef MSEOP_H_
#define MSEOP_H_    value

#include "../Operator.hpp"

template<typename DTYPE>
class MSEOP : public Operator<DTYPE>{
public:
    MSEOP(Operator<DTYPE> *pInput, Operator<DTYPE> *pLabel, std::string pName) : Operator<DTYPE>(pInput, pLabel, pName) {
        #ifdef __DEBUG__
        std::cout << "MSE::MSE(Operator<DTYPE> *, MetaParameter *, std::string)" << '\n';
        #endif  // __DEBUG__
        this->Alloc(pInput);
    }

    /*!
    @brief MSE(Mean Squared Error) LossFunction 클래스 소멸자
    @return 없음
    */
    virtual ~MSEOP() {
        #ifdef __DEBUG__
        std::cout << "MSE::~MSE()" << '\n';
        #endif  // __DEBUG__
    }

    virtual int Alloc(Operator<DTYPE> *pInput) {
        #ifdef __DEBUG__
        std::cout << "MSE::Alloc(Operator<DTYPE> *, Operator<DTYPE> *)" << '\n';
        #endif  // __DEBUG__

        int timesize    = pInput->GetResult()->GetTimeSize();
        int batchsize   = pInput->GetResult()->GetBatchSize();
        int channelsize = pInput->GetResult()->GetChannelSize();
        int rowsize     = 1;
        int colsize     = 1;

        this->SetResult(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));
        this->SetDelta(new Tensor<DTYPE>(timesize, batchsize, channelsize, rowsize, colsize));

        return TRUE;
    }

    int ForwardPropagate(int pTime = 0) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *input  = (*input_contatiner)[0]->GetResult();
        Tensor<DTYPE> *label  = (*input_contatiner)[1]->GetResult();
        Tensor<DTYPE> *result = this->GetResult();

        int batchsize   = input->GetBatchSize();
        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();
        int capacity    = channelsize * rowsize * colsize;

        int ti = pTime;

        for (int ba = 0, i = 0; ba < batchsize; ba++) {
            i = ti * batchsize + ba;

            for (int j = 0, index = 0; j < capacity; j++) {
                index         = i * capacity + j;
                (*result)[i] += Error((*input)[index], (*label)[index]);
            }
        }

        return TRUE;
    }

    /*!
    @brief Addall의 BackPropagate 매소드.
    @details Container에 저장한 pLeftInput, pRightInput의 Gradient값에 계산한 Gradient값을 각각 더한다.
    @param pTime 연산 할 Tensor가 위치한 Time값. default는 0으로 사용한다.
    @return 성공 시 TRUE.
    */
    int BackPropagate(int pTime = 0) {
        Container<Operator<DTYPE> *> *input_contatiner = this->GetInputContainer();

        Tensor<DTYPE> *input       = (*input_contatiner)[0]->GetResult();
        Tensor<DTYPE> *label       = (*input_contatiner)[1]->GetResult();
        Tensor<DTYPE> *input_delta = (*input_contatiner)[0]->GetGradient();
        Tensor<DTYPE> *this_delta  = this->GetGradient();

        int batchsize = input->GetBatchSize();

        int channelsize = input->GetChannelSize();
        int rowsize     = input->GetRowSize();
        int colsize     = input->GetColSize();
        int capacity    = channelsize * rowsize * colsize;

        int ti = pTime;

        for (int ba = 0, i = 0; ba < batchsize; ba++) {
            i = ti * batchsize + ba;

            for (int j = 0, index = 0; j < capacity; j++) {
                index                  = i * capacity + j;
                (*input_delta)[index] += ((*input)[index] - (*label)[index]) * (*this_delta)[i];
            }
        }

        return TRUE;
    }

#ifdef __CUDNN__

    /*!
    @brief GPU 동작 모드에서의 MSE(Mean Squared Error) LossFunction의 순전파를 수행하는 메소드
    @param pTime 더미 변수
    @return NULL
    @ref Tensor<DTYPE>MSE::ForwardPropagate(int pTime = 0)
    */
    int ForwardPropagateOnGPU(int pTime = 0) {
        this->ForwardPropagate();
        return TRUE;
    }

    /*!
    @brief GPU 동작 모드에서의 MSE(Mean Squared Error) LossFunction의 역전파를 수행하는 메소드
    @param pTime 더미 변수
    @return NULL
    @ref Tensor<DTYPE>MSE::BackPropagate(int pTime = 0)
    */
    int BackPropagateOnGPU(int pTime = 0) {
        this->BackPropagate();
        return TRUE;
    }

#endif  // __CUDNN__


    inline DTYPE Error(DTYPE pred, DTYPE ans) {
        return (pred - ans) * (pred - ans) / 2;
    }
};

#endif  // MSE_H_
