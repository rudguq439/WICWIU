#include "net/my_Generator.hpp"
#include "net/my_Discriminator.hpp"
#include "net/my_GAN.hpp"
#include "MNIST_Reader.hpp"
#include "../../WICWIU_src/Operator/NoiseGenerator/GaussianNoiseGenerator.hpp"
#include <time.h>

#define BATCH                 100
#define EPOCH                 10
#define LOOP_FOR_TRAIN        (60000 / BATCH)
#define LOOP_FOR_TEST         (10000 / BATCH)
#define LOOP_FOR_TRAIN_DISC   5
#define GPUID                 1

int main(int argc, char const *argv[]) {
    clock_t startTime, endTime;
    double  nProcessExcuteTime;
    char filename[]      = "GAN_params";

    // create input, label data placeholder -> Tensorholder
    Tensorholder<float> *z     = new Tensorholder<float>(1, BATCH, 1, 1, 100, "z");
    Tensorholder<float> *x     = new Tensorholder<float>(1, BATCH, 1, 1, 784, "x");
    Tensorholder<float> *label = new Tensorholder<float>(1, BATCH, 1, 1, 1, "label");

    // create NoiseGenrator
    GaussianNoiseGenerator<float> *Gnoise = new GaussianNoiseGenerator<float>(1, BATCH, 1, 1, 100, 0, 1);


    // ======================= Select net ===================
    GAN<float> *net  = new my_GAN<float>(z, x, label);

    // ======================= Prepare Data ===================
    MNISTDataSet<float> *dataset = CreateMNISTDataSet<float>();

#ifdef __CUDNN__
    net->SetDeviceGPU(GPUID);
#endif  // __CUDNN__

    net->PrintGraphInformation();

    float bestGenLoss  = 0.f;
    float bestDiscLoss = 0.f;
    int   epoch        = 0;

    // @ When load parameters
    net->Load(filename);

    std::cout << "bestGenLoss : " << bestGenLoss << '\n';
    std::cout << "bestDiscLoss : " << bestDiscLoss << '\n';
    std::cout << "epoch : " << epoch << '\n';

    //Start make Noise
    Gnoise->StartProduce();

    for (int i = epoch + 1; i < EPOCH; i++) {
        std::cout << "EPOCH : " << i << '\n';

        if ((i + 1) % 50 == 0) {
            std::cout << "Change learning rate!" << '\n';
            float lr = net->GetOptimizer()->GetLearningRate();
            net->GetOptimizer()->SetLearningRate(lr * 0.1);
        }

        // ======================= Train =======================
        float genLoss  = 0.f;
        float discLoss = 0.f;

        net->SetModeTrain();

        startTime = clock();

        for (int j = 0; j < LOOP_FOR_TRAIN; j++) {
            dataset->CreateTrainDataPair(BATCH);

            Tensor<float> *x_t = dataset->GetTrainFeedImage();
            Tensor<float> *z_t = Gnoise->GetNoiseFromBuffer();


#ifdef __CUDNN__
            x_t->SetDeviceGPU(GPUID);
            z_t->SetDeviceGPU(GPUID);
#endif  // __CUDNN__

            net->ResetParameterGradient();
            for(int k = 0; k < LOOP_FOR_TRAIN_DISC; k++){
                net->FeedInputTensor(2, z_t, x_t);
                net->TrainDiscriminator();
            }
            net->FeedInputTensor(1, z_t);
            net->TrainGenerator();

            genLoss  = net->GetGeneratorLossFunction()->GetResult()[0];
            discLoss = net->GetDiscriminatorLossFunction()->GetResult()[0];

            printf("\rTrain complete percentage is %d / %d -> Generator Loss : %f, Discriminator Loss : %f",
                   j + 1,
                   LOOP_FOR_TRAIN,
                   genLoss,
                   discLoss);
             fflush(stdout);

            // ** Legacy of other main.cpp **
            // train_accuracy += net->GetAccuracy();
            // train_avg_loss += net->GetLoss();
            //
            // printf("\rTrain complete percentage is %d / %d -> loss : %f, acc : %f"  /*(ExcuteTime : %f)*/,
            //        j + 1, LOOP_FOR_TRAIN,
            //        train_avg_loss / (j + 1),
            //        train_accuracy / (j + 1)
            //        /*nProcessExcuteTime*/);
            // fflush(stdout);
        }

        endTime            = clock();
        nProcessExcuteTime = ((double)(endTime - startTime)) / CLOCKS_PER_SEC;
        printf("\n(excution time per epoch : %f)\n\n", nProcessExcuteTime);

        // ======================= Test(Save Generated Image)======================
        float testGenLoss  = 0.f;
        float testDiscLoss = 0.f;

        net->SetModeInference();

        for (int j = 0; j < (int)LOOP_FOR_TEST; j++) {
            Tensor<float> *z_t = Gnoise->GetNoiseFromBuffer();

#ifdef __CUDNN__
            z_t->SetDeviceGPU(GPUID);
#endif  // __CUDNN__

            net->FeedInputTensor(1, z_t);
            net->Test();

            testGenLoss  = net->GetGeneratorLossFunction()->GetResult()[0];
            testDiscLoss = net->GetDiscriminatorLossFunction()->GetResult()[0];

            printf("\rTest complete percentage is %d / %d -> loss : %f, acc : %f",
                   j + 1,
                   LOOP_FOR_TEST,
                   testGenLoss,
                   testDiscLoss);
            fflush(stdout);
        }
        std::cout << "\n\n";

        // Global Optimal C(G) = -log4
        if ( abs(- 1.0 * log(4) - bestGenLoss) > abs(- 1.0 * log(4) - testGenLoss) ) {
            net->Save(filename);
        }
    }

    //Stop making Noise
    Gnoise->StopProduce();
    delete Gnoise;

    delete dataset;
    delete net;

    return 0;
}

// string filePath  = "MNIST.jpg";
// const char *cstr = filePath.c_str();
// Tensor2Image<float>(x_t, cstr, 3, 28, 28)
