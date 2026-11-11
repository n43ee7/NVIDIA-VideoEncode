#include <iostream>
//#include <SDL.h>
#include <cuda.h>
#include <stdio.h>
#include <filesystem>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "Dependencies/NvCodec/NvDecoder/NvDecoder.h"
#include "../Utils/NvCodecUtils.h"
#include "../Utils/FFmpegDemuxer.h"

using namespace std;
simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();


void ConvertSemiplanarToPlanar(uint8_t *pHostFrame, int nWidth, int nHeight, int nBitDepth) {
    if (nBitDepth == 8) {
        // nv12->iyuv
        YuvConverter<uint8_t> converter8(nWidth, nHeight);
        converter8.UVInterleavedToPlanar(pHostFrame);
    }
    else {
        // p016->yuv420p16
        YuvConverter<uint16_t> converter16(nWidth, nHeight);
        converter16.UVInterleavedToPlanar((uint16_t *)pHostFrame);
    }
}

void DecodeMediaFile(CUcontext cuContext, const char *szInFilePath, const char *szOutFilePath, bool bOutPlanar,
    const Rect &cropRect, const Dim &resizeDim)
{
    std::ofstream fpOut(szOutFilePath, std::ios::out | std::ios::binary);
    if (!fpOut)
    {
        std::ostringstream err;
        err << "Unable to open output file: " << szOutFilePath << std::endl;
        throw std::invalid_argument(err.str());
    }

    FFmpegDemuxer demuxer(szInFilePath);
    NvDecoder dec(cuContext, demuxer.GetWidth(), demuxer.GetHeight(), false, FFmpeg2NvCodecId(demuxer.GetVideoCodec()), NULL, false, false, &cropRect, &resizeDim);

    int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
    uint8_t *pVideo = NULL, **ppFrame;
    bool bDecodeOutSemiPlanar = false;
    do {
        demuxer.Demux(&pVideo, &nVideoBytes);
        dec.Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);
        if (!nFrame && nFrameReturned)
            LOG(INFO) << dec.GetVideoInfo();

        bDecodeOutSemiPlanar = (dec.GetOutputFormat() == cudaVideoSurfaceFormat_NV12) || (dec.GetOutputFormat() == cudaVideoSurfaceFormat_P016);

        for (int i = 0; i < nFrameReturned; i++) {
            if (bOutPlanar && bDecodeOutSemiPlanar) {
                ConvertSemiplanarToPlanar(ppFrame[i], dec.GetWidth(), dec.GetHeight(), dec.GetBitDepth());
            }
            fpOut.write(reinterpret_cast<char*>(ppFrame[i]), dec.GetFrameSize());
        }
        nFrame += nFrameReturned;
    } while (nVideoBytes);

    std::vector <std::string> aszDecodeOutFormat = { "NV12", "P016", "YUV444", "YUV444P16" };
    if (bOutPlanar) {
        aszDecodeOutFormat[0] = "iyuv";   aszDecodeOutFormat[1] = "yuv420p16";
    }
    std::cout << "Total frame decoded: " << nFrame << std::endl
        << "Saved in file " << szOutFilePath << " in "
        << aszDecodeOutFormat[dec.GetOutputFormat()]
        << " format" << std::endl;
    fpOut.close();
}

int main() {

    //Displaying Manageable GPUs

    printf("[!] Querying GPU Details \n");
    Sleep(1000);
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
        Sleep(5000);
        
        int selectedGPU; //ENC-DEC GPU 

        if (devCount == 1) {
            wcout << "[!]"<< " GPU " << devCount - 1<< " will be under use for all processes \n";
            selectedGPU = devCount - 1;
            Sleep(1500);
        }
        else
        {
            do 
            {
                cerr << "[!] Multiple GPUs Detected, Please Enter the GPU to be used \n";
                wcin >> selectedGPU;
                if (selectedGPU > devCount - 1) {
                    cerr << "[!] Invalid GPU ID \n";
                    cerr << " \n";
                }
            }
            while (selectedGPU <= devCount - 1);
            wcout << "[!]" << " GPU " << devCount - 1 << " will be under use for all processes \n";
            //Adding Different Encode and Decode GPUS
        } 
        int n; //CLS the bad boi way
        for (n = 0; n < 10; n++)
        {
            printf("\n\n\n\n\n\n\n\n\n\n");
        }
        char szInFilePath[256] = "", szOutFilePath[256] = "";
        bool bOutPlanar = false;
        int iGpu = 0;
        Rect cropRect = {};
        Dim resizeDim = {};
        try
        {
            //ParseCommandLine(argc, argv, szInFilePath, szOutFilePath, bOutPlanar, iGpu, cropRect, resizeDim);
            
            //Loading Modules
            cout << "Location for File to Decode";
            cin >> szInFilePath;
            cout << "Location for Output";
            cin >> szOutFilePath;
            cout << "bOutPlanar";
            int a;
            cin >> a;
            if (a == 1) {
                bOutPlanar = true;
            }
            iGpu = selectedGPU;

            // ENd loading modules
            
            
            
            
            
            

            if (!*szOutFilePath) {
                sprintf(szOutFilePath, bOutPlanar ? "out.planar" : "out.native");
            }

            ck(cuInit(0));
            CUdevice cuDevice = 0;
            ck(cuDeviceGet(&cuDevice, iGpu));
            char szDeviceName[80];
            ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
            std::cout << "GPU in use: " << szDeviceName << std::endl;
            CUcontext cuContext = NULL;
            ck(cuCtxCreate(&cuContext, 0, cuDevice));

            std::cout << "Decode with demuxing." << std::endl;
            DecodeMediaFile(cuContext, szInFilePath, szOutFilePath, bOutPlanar, cropRect, resizeDim);
        }
        catch (const std::exception& ex)
        {
            std::cout << ex.what();
            exit(1);
        }
        return 0;
        
}

//From Nvidia Examples
void ShowDecoderCapability() {
    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    std::cout << "Decoder Capability" << std::endl << std::endl;
    const char *aszCodecName[] = { "JPEG", "MPEG1", "MPEG2", "MPEG4", "H264", "HEVC", "HEVC", "HEVC", "HEVC", "HEVC", "HEVC", "VC1", "VP8", "VP9", "VP9", "VP9" };
    const char *aszChromaFormat[] = { "4:0:0", "4:2:0", "4:2:2", "4:4:4" };
    cudaVideoCodec aeCodec[] = { cudaVideoCodec_JPEG, cudaVideoCodec_MPEG1, cudaVideoCodec_MPEG2, cudaVideoCodec_MPEG4, cudaVideoCodec_H264, cudaVideoCodec_HEVC,
        cudaVideoCodec_HEVC, cudaVideoCodec_HEVC, cudaVideoCodec_HEVC, cudaVideoCodec_HEVC, cudaVideoCodec_HEVC, cudaVideoCodec_VC1, cudaVideoCodec_VP8,
        cudaVideoCodec_VP9, cudaVideoCodec_VP9, cudaVideoCodec_VP9 };
    int anBitDepthMinus8[] = { 0, 0, 0, 0, 0, 0, 2, 4, 0, 2, 4, 0, 0, 0, 2, 4 };
    cudaVideoChromaFormat aeChromaFormat[] = { cudaVideoChromaFormat_420, cudaVideoChromaFormat_420, cudaVideoChromaFormat_420, cudaVideoChromaFormat_420,
        cudaVideoChromaFormat_420, cudaVideoChromaFormat_420, cudaVideoChromaFormat_420, cudaVideoChromaFormat_420, cudaVideoChromaFormat_444, cudaVideoChromaFormat_444,
        cudaVideoChromaFormat_444, cudaVideoChromaFormat_420, cudaVideoChromaFormat_420, cudaVideoChromaFormat_420, cudaVideoChromaFormat_420, cudaVideoChromaFormat_420 };
    for (int iGpu = 0; iGpu < nGpu; iGpu++) {
        CUdevice cuDevice = 0;
        ck(cuDeviceGet(&cuDevice, iGpu));
        char szDeviceName[80];
        ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
        CUcontext cuContext = NULL;
        ck(cuCtxCreate(&cuContext, 0, cuDevice));

        std::cout << "GPU " << iGpu << " - " << szDeviceName << std::endl << std::endl;
        for (int i = 0; i < sizeof(aeCodec) / sizeof(aeCodec[0]); i++) {
            CUVIDDECODECAPS decodeCaps = {};
            decodeCaps.eCodecType = aeCodec[i];
            decodeCaps.eChromaFormat = aeChromaFormat[i];
            decodeCaps.nBitDepthMinus8 = anBitDepthMinus8[i];

            cuvidGetDecoderCaps(&decodeCaps);
            std::cout << "Codec" << "  " << aszCodecName[i] << '\t' <<
                "BitDepth" << "  " << decodeCaps.nBitDepthMinus8 + 8 << '\t' <<
                "ChromaFormat" << "  " << aszChromaFormat[decodeCaps.eChromaFormat] << '\t' <<
                "Supported" << "  " << (int)decodeCaps.bIsSupported << '\t' <<
                "MaxWidth" << "  " << decodeCaps.nMaxWidth << '\t' <<
                "MaxHeight" << "  " << decodeCaps.nMaxHeight << '\t' <<
                "MaxMBCount" << "  " << decodeCaps.nMaxMBCount << '\t' <<
                "MinWidth" << "  " << decodeCaps.nMinWidth << '\t' <<
                "MinHeight" << "  " << decodeCaps.nMinHeight << std::endl;
        }

        std::cout << std::endl;

        ck(cuCtxDestroy(cuContext));
    }
}