#include <iostream>
//#include <SDL.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace std;

int main() {

    //Selecting Manageable GPUs

    int nDevices;
    printf("[!] Querying GPU Details \n");
    printf("=================================================================================== \n");
        const int kb = 1024;
        const int mb = kb * kb;
     
        wcout << "CUDA version:   v" << CUDART_VERSION << endl;
        
        int devCount;
        cudaGetDeviceCount(&devCount);
        wcout << "CUDA Devices: " << endl << endl;

        for (int i = 0; i < devCount; ++i)
        {
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, i);
            wcout << i << ": " << props.name << ": " << props.major << "." << props.minor << endl;
            wcout << "  Global memory:   " << props.totalGlobalMem / mb << "MB" << endl;
            wcout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "KB" << endl;
            wcout << "  Constant memory: " << props.totalConstMem / kb << "KB" << endl;
            wcout << "  Block registers: " << props.regsPerBlock << endl << endl;

            wcout << "  Warp size:         " << props.warpSize << endl;
            wcout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
            wcout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << " ]" << endl;
            wcout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1] << ", " << props.maxGridSize[2] << " ]" << endl;
            wcout << endl;
        } 
            
        printf("=================================================================================== \n");
 
        //Delay Will do
 
        int selectedGPU; //ENC / DEC GPU 
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, devCount);
        if (devCount == 0) {
            wcout << "[!]"<< " GPU " << devCount <<" " << props.name << " will be under use for all processes \n";
            devCount = selectedGPU;
        }
        else
        {
            do {
                wcout << "[!] Multiple GPUs Detected, Please Enter the GPU to be used \n";
                wcin >> selectedGPU;
                if (selectedGPU > devCount) {
                    wcout << "[!] Invalid GPU ID \n";
                   //Delay
                    int n; 
                    for (n = 0; n < 10; n++)
                    {
                        printf("\n\n\n\n\n\n\n\n\n\n");
                    }

                }
            }
            while (selectedGPU <= devCount);
        } 
        int n; //CLS the bad boi way
        for (n = 0; n < 10; n++)
        {
            printf("\n\n\n\n\n\n\n\n\n\n");
        }
        return 0;
        
}
