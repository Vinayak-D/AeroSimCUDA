#include "system.cuh"
#include "controller.cuh"

// Compile with:
// nvcc main.cu matrix_operations.cu -o main

int main(){

    //Allocate memory
    System::allocateAll();
    Controller::allocateAll();

    //Initialize
    System::initializeAll();
    Controller::initializeAll();
    System::preCalculate();

    //Build
    System::build();

    //Execute
    System::run();

    //Synchronize
    System::synchronize();

    //Post-process
    System::postProcess();

    //Free memory
    System::freeAll();
    Controller::freeAll();

    std::cout << "---------------------------------- Done!" << std::endl;

    return 0;
}