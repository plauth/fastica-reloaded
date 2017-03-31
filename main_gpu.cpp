#include <iostream>
#include <boost/program_options.hpp>
#include <sndfile.hh>
#include <eigen3/Eigen/Dense>
#include "clutil.h"
#include "clmatrix.h"
#include "clreduceaddkernel.h"
#include "clcenterkernel.h"

using namespace Eigen;
using namespace std;
using namespace boost::program_options;

inline int nextPow2(int x) {
    --x;

    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;

    return ++x;
}

static std::vector<SndfileHandle> retrieveSndfileHandlesFromArgs(variables_map args) {
    std::vector<SndfileHandle> sndfileHandles;
    
    for(auto fileName : args["infile"].as<std::vector<std::string>>()) {
        SndfileHandle sndfileHandle = SndfileHandle (fileName.c_str()) ;
        
        if(args.count("verbose")) {
            printf ("Opened file '%s'\n", fileName.c_str());
            printf ("\tSample rate : %d\n", sndfileHandle.samplerate ()) ;
            printf ("\tChannels    : %d\n", sndfileHandle.channels ()) ;
        }
        
        sndfileHandles.push_back(sndfileHandle);
    }
    
    return sndfileHandles;
}

/**
 * Verifies that all SndfileHandles are equal in format, lenght, samplerate and the number of channels.
 * @param sndfileHandles a std::vector of SndfileHandle objects.
 */
static void checkAllFilesAreEqual(std::vector<SndfileHandle> sndfileHandles) {
    SndfileHandle* comparisonHandle = nullptr;
    for(auto sndfileHandle : sndfileHandles) {
        if(comparisonHandle == nullptr) {
            comparisonHandle = &sndfileHandle;
        } else {
            assert(comparisonHandle->frames() == sndfileHandle.frames());
            assert(comparisonHandle->format() == sndfileHandle.format());
            assert(comparisonHandle->channels() == sndfileHandle.channels());
            assert(comparisonHandle->samplerate() == sndfileHandle.samplerate());
            
            comparisonHandle = &sndfileHandle;
        }
    }
}

static std::shared_ptr<CLMatrix> readSndfiles(std::vector<SndfileHandle> sndfileHandles, cl::Context context, cl::CommandQueue queue) {
    unsigned long numTracks = sndfileHandles.size();
    long long numFrames = sndfileHandles.back().frames();
    long long numFramesPadded = nextPow2(numFrames);
    std::shared_ptr<CLMatrix> audioMatrix(new CLMatrix(context, queue, numTracks, numFramesPadded));
    //audioMatrix->setZero();

    for(int row = 0; row < numTracks; row++) {
        sndfileHandles[row].readf(audioMatrix->getData() + row*numFramesPadded, numFrames);
    }

    return audioMatrix;
}

static void writeSignalMatrix(MatrixXf outputSignalMatrix, std::vector<SndfileHandle> sndfileHandles) {
    for(int row = 0; row < outputSignalMatrix.rows(); row++){
        SndfileHandle templateSndfileHandle = sndfileHandles.at(row);
        long long numFrames = sndfileHandles.back().frames();
        
        std::stringstream outputFilnameStream;
        outputFilnameStream << "output_" << row << ".aif";
        SndfileHandle writeSndfileHandle = SndfileHandle(outputFilnameStream.str(), SFM_WRITE, templateSndfileHandle.format(),
                                                         templateSndfileHandle.channels(), templateSndfileHandle.samplerate());
        
        float * audioTrack = (float *) malloc(numFrames * sizeof(float));
        for(long long col = 0; col < numFrames; col++) {
            audioTrack[col] = outputSignalMatrix(row,col);
        }
        writeSndfileHandle.writef(audioTrack, numFrames);
        free(audioTrack);
    }
}

static MatrixXf contrast(MatrixXf previousUnmixingCandidate, MatrixXf signalMatrix) {
    // g(x) = tanh(x)
    MatrixXf g = previousUnmixingCandidate * signalMatrix;
    g = g.array().tanh();
    
    // g'(x) = 1-tanh(x)^2
    MatrixXf gDash = g.array().pow(2);
    gDash = MatrixXf::Ones(signalMatrix.rows(), signalMatrix.cols()) - gDash;
    
    MatrixXf newUnmixingMatrixCandidate = g * signalMatrix.transpose();
    newUnmixingMatrixCandidate /= signalMatrix.cols();
    
    MatrixXf gDashMeans = MatrixXf::Zero(signalMatrix.rows(),signalMatrix.rows());
    MatrixXf gDashMeansTemp = gDash.rowwise().mean();
    
    for(int i = 0; i < signalMatrix.rows(); i++){
        gDashMeans(i,i) = gDashMeansTemp.coeffRef(i);
    }
    
    return newUnmixingMatrixCandidate - (gDashMeans * previousUnmixingCandidate);
    
}

static MatrixXf orthogonalizeUnmixingMatrix(MatrixXf unmixingMatrix) {
    EigenSolver<MatrixXf> eigenSolver(unmixingMatrix * unmixingMatrix.transpose());
    MatrixXf eigenvalues = eigenSolver.eigenvalues().real();
    MatrixXf eigenvectors = eigenSolver.eigenvectors().real();

    MatrixXf eigenvaluesDiagonalized = MatrixXf::Zero(unmixingMatrix.rows(),unmixingMatrix.cols());
    
    for(int i = 0; i < unmixingMatrix.rows(); i++){
        eigenvaluesDiagonalized(i,i) = 1.0f / sqrtf(eigenvalues.coeffRef(i));
    }
    
    return (eigenSolver.eigenvectors().real() * eigenvaluesDiagonalized * eigenSolver.eigenvectors().real().transpose()) * unmixingMatrix;
}

static float getConvergenceRate(MatrixXf newUnmixingMatrixCandidate, MatrixXf previousUnmixingMatrixCandidate) {
    MatrixXf differenceMatrix = newUnmixingMatrixCandidate * previousUnmixingMatrixCandidate.transpose();
    
    float minimum = 1.0f;
    for(int i = 0; i < differenceMatrix.rows(); i++){
        if(minimum > fabs(differenceMatrix(i,i)))
            minimum = fabs(differenceMatrix(i,i));
    }
    
    return 1.0f-minimum;
}



int main(int argc, char *argv[]) {

    try
    {
        options_description opts{"Options"};
        opts.add_options()
        ("help,h", "Help screen")
        ("verbose,v", "Verbose mode")
        ("infile,i", value<std::vector<std::string>>()->multitoken()->
         zero_tokens()->composing(), "Input File");

        
        variables_map args;
        store(parse_command_line(argc, argv, opts), args);
        notify(args);
        
        if (args.count("help") || !args.count("infile")) {
            std::cout << opts << endl;
            exit(0);
        }

        cl_int err;
        cl::vector< cl::Platform > platformList;
        cl::Platform::get(&platformList);
        checkErr(platformList.size()!=0 ? CL_SUCCESS : -1, "cl::Platform::get");
        std::cerr << "Platform number is: " << platformList.size() << std::endl;
        std::string platformVendor;
        platformList[0].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
        std::cerr << "Platform is by: " << platformVendor << "\n";
        cl_context_properties cprops[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[0])(), 0};
        //platformList[0]()
        cl::Context context(CL_DEVICE_TYPE_GPU, cprops, NULL, NULL, &err);
        checkErr(err, "Context::Context()");
        cl::vector<cl::Device> devices;
        devices = context.getInfo<CL_CONTEXT_DEVICES>();
        checkErr(devices.size() > 0 ? CL_SUCCESS : -1, "devices.size() > 0");
        std::string deviceName;
        devices[0].getInfo((cl_device_info)CL_DEVICE_NAME,&deviceName);
        std::cerr << "Device: " << deviceName << std::endl;

        cl::CommandQueue mappingQueue(context, devices[0], 0, &err);
        checkErr(err, "CommandQueue::CommandQueue()");


        if(args.count("verbose"))
            std::cout << "Number of threads used by Eigen: " << Eigen::nbThreads() << endl;
        
        auto sndfileHandles = retrieveSndfileHandlesFromArgs(args);
        checkAllFilesAreEqual(sndfileHandles);
        std::shared_ptr<CLMatrix> inputSignalClMatrix = readSndfiles(sndfileHandles, context, mappingQueue);
    
        // Stage 1: preprocessing
        // Step 1.1: centering
        inputSignalClMatrix = CLCenterKernel::subtract(context, mappingQueue, inputSignalClMatrix, CLReduceAddKernel::getMeans(context, mappingQueue, inputSignalClMatrix));
        checkErr(mappingQueue.finish(), "clFinish()");
        MatrixXf inputSignalMatrix = inputSignalClMatrix->toMatrixXf();

        
        // Step 1.2: whitening
        MatrixXf covarianceMatrix = (inputSignalMatrix * inputSignalMatrix.transpose()) / float(inputSignalMatrix.cols()-1);
        
        EigenSolver<MatrixXf> eigenSolver(covarianceMatrix);
        
        MatrixXf eigenvalues = eigenSolver.eigenvalues().real();
        MatrixXf eigenvaluesSqrt = MatrixXf::Zero(inputSignalMatrix.rows(),inputSignalMatrix.rows());
        MatrixXf eigenvaluesReciprocalSqrt = MatrixXf::Zero(inputSignalMatrix.rows(),inputSignalMatrix.rows());
        
        for(int i = 0; i < inputSignalMatrix.rows(); i++){
            float eigenvalueSqrt = sqrtf(eigenvalues.coeffRef(i));
            eigenvaluesSqrt(i,i) = eigenvalueSqrt;
            eigenvaluesReciprocalSqrt(i,i) = 1.0f / eigenvalueSqrt;
        }
        
        MatrixXf eigenvectors = eigenSolver.eigenvectors().real();
        
        MatrixXf whiteningMatrix = eigenvaluesReciprocalSqrt * eigenvectors.transpose();
        MatrixXf dewhiteningMatrix = eigenvectors * eigenvaluesSqrt;
        
        MatrixXf whitenedSignalMatrix = whiteningMatrix * inputSignalMatrix;
        
        // Step 2: FastICA
        MatrixXf previousUnmixingMatrixCandidate = MatrixXf::Identity(whitenedSignalMatrix.rows(), whitenedSignalMatrix.rows());
        MatrixXf newUnmixingMatrixCandidate = MatrixXf::Zero(whitenedSignalMatrix.rows(), whitenedSignalMatrix.rows());
        
        for(int iteration = 0; iteration < 100; iteration++) {
            // Step 2.1: contrast function
             newUnmixingMatrixCandidate = contrast(previousUnmixingMatrixCandidate, whitenedSignalMatrix);
            
            // Step 2.2: Orthogonalize new guess
            newUnmixingMatrixCandidate = orthogonalizeUnmixingMatrix(newUnmixingMatrixCandidate);
            
            float convergence = getConvergenceRate(newUnmixingMatrixCandidate, previousUnmixingMatrixCandidate);
            if(args.count("verbose"))
                std::cout << "iteration: " << iteration << ", convergence: " << convergence << endl;
            
            if((convergence) < 0.000001f && iteration > 0)
                break;
            
            previousUnmixingMatrixCandidate = newUnmixingMatrixCandidate;
        }
        
        // Step 3: Postprocessing
        MatrixXf mixingMatrix = dewhiteningMatrix * newUnmixingMatrixCandidate.transpose();
        MatrixXf unmixingMatrix = newUnmixingMatrixCandidate * whiteningMatrix;
        
        MatrixXf seperatedSignalsMatrix = unmixingMatrix * inputSignalMatrix;
        
        MatrixXf maxima = seperatedSignalsMatrix.rowwise().maxCoeff();
        MatrixXf minima = seperatedSignalsMatrix.rowwise().minCoeff().array().abs().matrix();
        
        for(int i = 0; i < seperatedSignalsMatrix.rows(); i++) {
            float maximum = maxima.coeffRef(i) > minima.coeffRef(i) ? maxima.coeffRef(i) : minima.coeffRef(i);
            seperatedSignalsMatrix.row(i) /= maximum;
        }
        
        writeSignalMatrix(seperatedSignalsMatrix, sndfileHandles);
    }
    catch (const error &ex)
    {
        std::cerr << ex.what() << '\n';
    }

    
}
