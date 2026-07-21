#ifndef CONTROLLER_CUH
#define CONTROLLER_CUH

#include "config.h"

namespace Controller{

    constexpr int n = Config::n;
    constexpr int m = Config::m;
    constexpr int p = Config::p;
    constexpr int r = Config::r;

    static const float K_eMPC_init[m*r] = {0.08606f, -0.05880f, 0.04419f, 0.08617f};

    static const float F_init[r*n] = {32.95182f, 0.42580f, -5.69892f, -56.67244f, -0.00005f, 
                                -0.02013f, 0.64302f, -0.45613f, -55.40961f, 7.38901f};

    static const float G_T_init[r*m] = {7.34467f, -0.00346f, 5.67438f, 0.03309f};

    static const float Q_init[r*r] = {1.0f, 0.0f, 0.0f, 49.0f};

    static const float R_init[m*m] = {10.0f, 0.0f, 0.0f, 15.0f};

    static const float M_con_init[(4*m)*m] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        -1.0f, 0.0f,
        0.0f, -1.0f,
        1.0f, 0.0f,
        0.0f, 1.0f,
        -1.0f, 0.0f,
        0.0f, -1.0f
    };

    static const float g_con_init[(4*m)*1] = {
        0.4f, 0.7f, 0.4f, -0.0f, 0.4f, 0.01040f, -0.4f, -0.01040f
    };

    // Pointers (will be allocated at runtime)
    float* K_eMPC = nullptr;
    float* F = nullptr;
    float* G_T = nullptr;
    float* Q = nullptr;
    float* R = nullptr;
    float* M_con = nullptr;
    float* g_con = nullptr;

    //Placeholders
    float* mul_GQ = nullptr;
    
    //Allocate
    inline void allocateAll() {
        //matrices
        Config::allocate(K_eMPC, m, r);
        Config::allocate(F, r, n);
        Config::allocate(G_T, r, m);
        Config::allocate(Q, r, r);
        Config::allocate(R, m, m);
        Config::allocate(M_con, 4 * m, m);
        Config::allocate(g_con, 4 * m, 1);
        //placeholders
        Config::allocate(mul_GQ, r, n);
    }

    //Initialize
    inline void initializeAll() {
        //Allocated memory (copy inits)
        std::memcpy(K_eMPC, K_eMPC_init, m * r * sizeof(float));
        std::memcpy(F, F_init, r * n * sizeof(float));
        std::memcpy(G_T, G_T_init, r * m * sizeof(float));
        std::memcpy(Q, Q_init, r * r * sizeof(float));
        std::memcpy(R, R_init, m * m * sizeof(float));
        std::memcpy(M_con, M_con_init, (4 * m) * m * sizeof(float));
        std::memcpy(g_con, g_con_init, (4 * m) * sizeof(float));
        //Zero temporaries
        std::memset(mul_GQ, 0, r * n * sizeof(float));
    }

    //Free
    inline void freeAll() {
        //matrices
        Config::free(K_eMPC);
        Config::free(F);
        Config::free(G_T);
        Config::free(Q);
        Config::free(R);
        Config::free(M_con);
        Config::free(g_con);
        Config::free(mul_GQ);
    }
    
}
#endif // CONTROLLER_CUH
