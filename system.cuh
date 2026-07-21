#include "matrix_operations.cuh"
#include "controller.cuh"

namespace System{

    constexpr int n = Config::n;
    constexpr int m = Config::m;
    constexpr int p = Config::p;
    constexpr int r = Config::r;

    int iterations = 10;

    static const float A_init[n*n] = {0.9987f, 0.0037f, -0.0426f, -0.4901f, 0.0f,
                                       -0.0112f, 0.8916f, 3.2229f, -0.0021f, 0.0f,
                                       0.0002f, -0.0067f, 0.8510f, 0.0f, 0.0f,
                                       0.0f, -0.0002f, 0.0463f, 1.0f, 0.0f,
                                       -0.0008f, 0.0476f, -0.0049f, -3.7499f, 1.0f};

    static const float B_init[n*m] = {0.2295f, 0.0243f,
                                       -0.0013f, -2.7057f,
                                       2.9023e-05f, -1.1274f,
                                       5.5736e-07f, -0.0290f,
                                       -8.6231e-05f, -0.0149f};

    static const float C_init[p*n] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                       0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                                       0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                       0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                                       0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

    static const float H_init[r*n] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                       0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

    // initial states, inputs, setpoint
    static const float X_init[n] = {75.0f, 0.0f, 0.0f, 0.0f, 500.0f};
    static const float U_init[m] = {0.25f, -0.3f};
    static const float R_set_init[r*1] = {25.0f, 30.0f};

    // Pointers (will be allocated at runtime)
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* H = nullptr;
    float* X = nullptr;
    float* U = nullptr;
    float* Y = nullptr;
    float* Z = nullptr;
    float* Zpr = nullptr;
    float* E = nullptr;
    float* f = nullptr;
    float* R_set = nullptr;

    //Placeholders
    float* mul_AX = nullptr;
    float* mul_BU = nullptr;
    float* mul_CX = nullptr;
    float* mul_HX = nullptr;

    //CUDA
    cudaGraph_t graph;
    cudaStream_t stream;

    //Forward declaration
    inline void postProcess();

    //Step simulation
    void step(cudaStream_t stream) {
        // AX
        matrixMultiply(A, X, mul_AX, n, n, 1, stream);
        // BU
        matrixMultiply(B, U, mul_BU, n, m, 1, stream);
        // CX
        matrixMultiply(C, X, mul_CX, p, n, 1, stream);
        // X = AX + BU
        matrixAdd(mul_AX, mul_BU, X, n, 1, stream);
        //Assign outputs
        matrixAssign(mul_CX, Y, p, 1, stream);
        // Z = HY
        matrixMultiply(H, Y, Z, r, p, 1, stream);
        // Zpr = FX
        matrixMultiply(Controller::F, X, Zpr, r, n, 1, stream);
        // E = R_Set - Zpr
        matrixSubtract(R_set, Zpr, E, r, 1, stream);
        // f = mul_GQ * E
        matrixMultiply(Controller::mul_GQ, E, f, r, r, 1, stream);
        // U = K_eMPC * E
        matrixMultiply(Controller::K_eMPC, E, U, m, r, 1, stream);   
    }

    inline void synchronize() {
        cudaStreamSynchronize(stream);
    }

    //Build graph
    inline void buildGraph() {
        cudaStreamCreate(&stream);

        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        for (int i = 0; i < iterations; ++i) {
            step(stream);
        }
        cudaStreamEndCapture(stream, &graph);
    }

    //Execute
    inline void executeGraph() {
        cudaGraphExec_t graphExec;
        cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);        
        cudaEventRecord(start, stream);

        //Main execution (non-blocking), calling it again for another graph will achieve concurrency
        cudaGraphLaunch(graphExec, stream);

        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop); 
        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        printf("step time = %f ms\n", ms);
    }
    inline void run(){
        executeGraph();
    }

    //Allocate
    inline void allocateAll() {
        // system matrices and vectors (no initialization)
        Config::allocate(A, n, n);
        Config::allocate(B, n, m);
        Config::allocate(C, p, n);
        Config::allocate(H, r, n);

        // state and I/O
        Config::allocate(X, n, 1);
        Config::allocate(U, m, 1);
        Config::allocate(Y, p, 1);
        Config::allocate(Z, r, 1);
        Config::allocate(Zpr, r, 1);
        Config::allocate(E, r, 1);
        Config::allocate(f, r, 1);
        Config::allocate(R_set, r, 1);

        // placeholders
        Config::allocate(mul_AX, n, 1);
        Config::allocate(mul_BU, n, 1);
        Config::allocate(mul_CX, p, 1);
        Config::allocate(mul_HX, r, 1);
    }

    // Initialize
    inline void initializeAll() {
        //Allocated memory (copy inits)
        std::memcpy(A, A_init, n * n * sizeof(float));
        std::memcpy(B, B_init, n * m * sizeof(float));
        std::memcpy(C, C_init, p * n * sizeof(float));
        std::memcpy(H, H_init, r * n * sizeof(float));
        std::memcpy(X, X_init, n * sizeof(float));
        std::memcpy(U, U_init, m * sizeof(float));
        std::memcpy(R_set, R_set_init, r * sizeof(float));
        //Zero temporaries
        std::memset(Y, 0, p * sizeof(float));
        std::memset(Z, 0, r * sizeof(float));
        std::memset(Zpr, 0, r * sizeof(float));
        std::memset(E, 0, r * sizeof(float));
        std::memset(f, 0, r * sizeof(float));
        std::memset(mul_AX, 0, n * sizeof(float));
        std::memset(mul_BU, 0, n * sizeof(float));
        std::memset(mul_CX, 0, p * sizeof(float));
        std::memset(mul_HX, 0, r * sizeof(float));
    }

    //Build graph
    inline void build() {
        buildGraph();
    }

    // Free
    inline void freeAll() {
        Config::free(A);
        Config::free(B);
        Config::free(C);
        Config::free(H);
        Config::free(X);
        Config::free(U);
        Config::free(Y);
        Config::free(Z);
        Config::free(E);
        Config::free(f);
        Config::free(R_set);
        Config::free(mul_AX);
        Config::free(mul_BU);
        Config::free(mul_CX);
        Config::free(mul_HX);
        //Graph and stream
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream); 
    }

    //Pre-calculation
    inline void preCalculate() {
        matrixMultiply(Controller::G_T, Controller::Q, Controller::mul_GQ, r, m, r, 0);
    }

    //Post-processing
    inline void postProcess() {
        std::cout << "C::mul_GQ: ";
        for (int i = 0; i < r * m; ++i) {
            std::cout << Controller::mul_GQ[i] << " ";
        }
        std::cout << "S::E: ";
        for (int i = 0; i < r; ++i) {
            std::cout << E[i] << " ";
        }
        std::cout << "S::F: ";
        for (int i = 0; i < r; ++i) {
            std::cout << f[i] << " ";
        }
        std::cout << "S::U: ";
        for (int i = 0; i < m; ++i) {
            std::cout << U[i] << " ";
        }
        std::cout << std::endl;
    }

}