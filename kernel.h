#pragma once

#define CUDA_VERSION 11060
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "utils.h"
#include "TransferFunction.h"
#include "Octree.h"
#include "BinaryLoader.h"

namespace myCUDAspace{
	void setupRayDirectionCUDA(glm::vec3* screen_pixel_ray_dir, AppData* appdata);
	
    //do not use, only for test purpose
    void renderLoop(glm::vec4* host_screen, AppData* host_appdata, Octree* host_octree, TransferFunction* host_transfer_function, NiftiFile* host_nf);

    //do not use, only for test purpose
    void renderLoop2(glm::vec4* host_screen, AppData* host_appdata, Octree* host_octree, TransferFunction* host_transfer_function, NiftiFile* host_nf);

    cudaError_t allocateDeviceMemory(
        glm::vec4* host_screen, AppData* host_appdata, Octree* host_octree, TransferFunction* host_transfer_function, NiftiFile* host_nf,
        glm::vec4* device_screen, AppData* device_application_data, Octree* device_octree, Node* device_octree_array,
        TransferFunction* device_transfer_function, mat_interval* device_material_intervals, NiftiFile* device_nf,
        glm::vec3* device_primary_rays, glm::vec4* device_sample_colors, glm::mat4* device_modelAux);

    // device_* parameters are outputs of this function, and hold the pointer to the GPU allocated locations
    cudaError_t allocateDeviceMemory2(
        glm::vec4* host_screen, AppData* host_appdata, Octree* host_octree, TransferFunction* host_transfer_function, NiftiFile* host_nf,
        glm::vec4** device_screen, AppData** device_application_data, Octree** device_octree, Node** device_octree_array,
        TransferFunction** device_transfer_function, mat_interval** device_material_intervals, NiftiFile** device_nf, float** device_nf_volume,
        glm::vec3** device_primary_rays, glm::vec4** device_sample_colors, glm::mat4** device_modelAux);

    // free the allocated GPU memory locations
    cudaError_t deallocateDeviceMemory(glm::vec4* device_screen, AppData* device_application_data, Octree* device_octree, Node* device_octree_array,
        TransferFunction* device_transfer_function, mat_interval* device_material_intervals, NiftiFile* device_nf, float* device_nf_volume,
        glm::vec3* device_primary_rays, glm::vec4* device_sample_colors, glm::mat4* device_modelAux);

    //updates device memory
    cudaError_t updatePrimaryRayDirection(AppData* host_appdata, AppData* device_application_data, glm::vec3* device_primary_rays);

    //updates device memory
    cudaError_t updateCameraLocation(AppData* host_appdata, AppData* device_appdata);

    cudaError_t getSampleColors(
        AppData* host_appdata,
        NiftiFile* host_nf,
        AppData* device_application_data,
        Octree* device_octree,
        TransferFunction* device_transfer_function,
        glm::vec3* device_primary_rays,
        glm::vec4* device_sample_colors, //output
        glm::mat4* device_modelAux,
        int screen_size
    );

    cudaError_t getSampleColorsFromNF(
        glm::vec4* device_sample_colors, //output
        AppData* host_appdata,
        NiftiFile* host_nf,
        AppData* device_appdata,
        NiftiFile* device_nf,
        TransferFunction* device_transfer_function,
        int screen_size
    );

    cudaError_t blendSampleColors(AppData* host_appdata,
        AppData* device_application_data,
        glm::vec4* device_screen,
        glm::vec4* device_sample_colors,
        int screen_size);
};