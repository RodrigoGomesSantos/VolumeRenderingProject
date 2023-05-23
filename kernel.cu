

#include <stdio.h>
#include <iostream>
#include <time.h>

#include "kernel.h" 

#include <string>


//forward declaration
__global__ void rayDirectionKernel(glm::vec3* dev_res, AppData* dev_appdata);
void  myCUDAspace::setupRayDirectionCUDA(glm::vec3* screen_pixel_ray_dir, AppData* appdata);

///////////
///////////
///////////

__global__ void rayDirectionKernel(glm::vec3* dev_res, AppData* dev_appdata) {

    int idx =  blockIdx.x * blockDim.x + threadIdx.x;
    int idy =  blockIdx.y * blockDim.y + threadIdx.y;
    
    int id = idx * dev_appdata->SCR_HEIGHT + idy;

    int size = dev_appdata->SCR_WIDTH * dev_appdata->SCR_HEIGHT;

    if(id < size) {
        if (dev_appdata->conic)
            dev_res[id] = glm::normalize(dev_appdata->top_left_corner +
                (idx * dev_appdata->real_screen_width / dev_appdata->SCR_WIDTH) * (dev_appdata->cameraRight) +
                (idy * dev_appdata->real_screen_height / dev_appdata->SCR_HEIGHT) * (-dev_appdata->cameraUp)
                - dev_appdata->cameraPos);
        else
            dev_res[id] = dev_appdata->cameraFront;
    }
}

__global__ void calculateSampleColor(glm::vec4* device_sample_colors, AppData* device_appdata,
    Octree* device_octree, TransferFunction* device_transfer_function,
    glm::vec3* device_primary_rays,  glm::mat4* modelAux, int max_intensity , int total_sample_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    int id = idx * device_appdata->SCR_HEIGHT * device_appdata->samples_per_ray + idy * device_appdata->samples_per_ray + idz;

    if (id < total_sample_size) {

        glm::vec3 auxPos = glm::vec3(0.0f);
        if (device_appdata->conic)
            auxPos  = device_appdata->cameraPos + (idz * device_appdata->sample_distance + device_appdata->front_clip_plane) * device_primary_rays[idx * device_appdata->SCR_HEIGHT + idy];
        else
            auxPos = device_appdata->top_left_corner +
                idx * device_appdata->real_screen_width / device_appdata->SCR_WIDTH * device_appdata->cameraRight +
                idy * device_appdata->real_screen_height / device_appdata->SCR_HEIGHT * (-device_appdata->cameraUp) +
                (idz * device_appdata->sample_distance + device_appdata->front_clip_plane) * device_primary_rays[idx * device_appdata->SCR_HEIGHT + idy];
        /*
        float intensity = device_octree->device_getIntensity(*modelAux * glm::vec4(auxPos, 1.0f));
        float normalized_intensity = intensity / max_intensity;
        */
        float normalized_intensity = device_octree->device_getIntensity(*modelAux * glm::vec4(auxPos, 1.0f)) / max_intensity;
        glm::vec4 color = device_transfer_function->getMaterial(normalized_intensity)->color;
        
        device_sample_colors[id] = color;

    }
}

__global__ void getColorFromNF(
    glm::vec4* device_sample_colors,
    AppData* device_appdata,
    NiftiFile* device_nf,
    TransferFunction* device_transfer_function,
    glm::mat4* device_transform,
    glm::mat4 aux_transform,
    glm::mat4 centerGridModel,
    glm::mat4 inverseView,
    dim3 size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int idz = blockIdx.z * blockDim.z + threadIdx.z;

    int id =
        idx * device_appdata->SCR_HEIGHT * device_appdata->samples_per_ray +
        idy * device_appdata->samples_per_ray +
        idz;
        
    if (idx < size.x && idy < size.y && idz < size.z) {
    //if (id < size.x * size.y * size.z ) {
        

        float intensity = 0.0f;

        //glm::vec3 position = *device_transform * glm::vec4(idx, idy,idz, 1.0f);

        glm::vec3 position = glm::vec3(idx, idy,idz);

        position =  centerGridModel * glm::vec4(position, 1.0f);
        /*//uncomment for perpetive
        float range = 0.0001f + (((float)idz) / (size.z));
        float x = position.x;
        float y = position.y;
        float z = position.z;
        x = range * x;
        y = range * y;
        z = 1.0f * z;
        position = glm::vec3(x, y, z);
        */
        position = inverseView * glm::vec4(position, 1.0f);

        position = aux_transform * glm::vec4(position,1.0f);
        
        glm::vec4 cf = device_transfer_function->getMaterial(0.0f / device_nf->header.cal_max)->color;
        if (device_nf->isInside(position)) {
            
            //int index = 0;
            //int counter = 0;
            //intensity = device_nf->volume[index];

            int index = device_nf->transformVector3Position(position);
            intensity = device_nf->volume[index];
            
            glm::vec3 difference = position - glm::vec3((int)position.x, (int)position.y, (int)position.z);
            
            index = device_nf->transformVector3Position(position + glm::vec3(0.0f, 0.0f, 0.0f));
            float iX1 =  (index < device_nf->totaldim)? device_nf->volume[index] : 0.0f;
            glm::vec4 cX1 = device_transfer_function->getMaterial(iX1 / device_nf->header.cal_max)->color;

            index = device_nf->transformVector3Position(position + glm::vec3(0.0f, 0.0f, 1.0f));
            float iX2 = (index < device_nf->totaldim) ? device_nf->volume[index] : 0.0f;
            glm::vec4 cX2 = device_transfer_function->getMaterial(iX2 / device_nf->header.cal_max)->color;

            index = device_nf->transformVector3Position(position + glm::vec3(0.0f, 1.0f, 0.0f));
            float iX3 = (index < device_nf->totaldim) ? device_nf->volume[index] : 0.0f;
            glm::vec4 cX3 = device_transfer_function->getMaterial(iX3 / device_nf->header.cal_max)->color;

            index = device_nf->transformVector3Position(position + glm::vec3(0.0f, 1.0f, 1.0f));
            float iX4 = (index < device_nf->totaldim) ? device_nf->volume[index] : 0.0f;
            glm::vec4 cX4 = device_transfer_function->getMaterial(iX4 / device_nf->header.cal_max)->color;

            index = device_nf->transformVector3Position(position + glm::vec3(1.0f, 0.0f, 0.0f));
            float iX5 = (index < device_nf->totaldim) ? device_nf->volume[index] : 0.0f;
            glm::vec4 cX5 = device_transfer_function->getMaterial(iX5 / device_nf->header.cal_max)->color;

            index = device_nf->transformVector3Position(position + glm::vec3(1.0f, 0.0f, 1.0f));
            float iX6 = (index < device_nf->totaldim) ? device_nf->volume[index] : 0.0f;
            glm::vec4 cX6 = device_transfer_function->getMaterial(iX6 / device_nf->header.cal_max)->color;

            index = device_nf->transformVector3Position(position + glm::vec3(1.0f, 1.0f, 0.0f));
            float iX7 = (index < device_nf->totaldim) ? device_nf->volume[index] : 0.0f;
            glm::vec4 cX7 = device_transfer_function->getMaterial(iX7 / device_nf->header.cal_max)->color;

            index = device_nf->transformVector3Position(position + glm::vec3(1.0f, 1.0f, 1.0f));
            float iX8 = (index < device_nf->totaldim) ? device_nf->volume[index] : 0.0f;
            glm::vec4 cX8 = device_transfer_function->getMaterial(iX8 / device_nf->header.cal_max)->color;

            ////////////////
            glm::vec4 cY1 = cX1 * (1.0f - difference.y) + cX3 * (difference.y);

            glm::vec4 cY2 = cX2 * (1.0f - difference.y) + cX4 * (difference.y);
            
            glm::vec4 cY3 = cX5 * (1.0f - difference.y) + cX7 * (difference.y);
            
            glm::vec4 cY4 = cX6 * (1.0f - difference.y) + cX8 * (difference.y);


            glm::vec4 cZ1 = cY1 * (1.0f - difference.x) + cY3 * (difference.x);

            glm::vec4 cZ2 = cY2 * (1.0f - difference.x) + cY4 * (difference.x);

            cf = cZ1 * (1.0f - difference.z) + cZ2 * (difference.z);
            

        }

        //glm::vec4 color = device_transfer_function->getMaterial(intensity/ device_nf->header.cal_max)->color;
        
        //device_sample_colors[id] = color;
        device_sample_colors[id] = cf;
        //device_sample_colors[id] = glm::vec4(position,1.0f);
    }

}

void __device__ cubicInterpolation() {


}

__global__ void blendSampleColors(glm::vec4* device_screen_colors, glm::vec4* device_sample_colors, AppData* device_appdata, int size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    int pixel_id = idx * device_appdata->SCR_HEIGHT + idy;

    if (pixel_id < size) {
        
        glm::vec4 fragment_color = device_appdata->BACKGROUND_COLOUR;
        
        for (int i = device_appdata->samples_per_ray - 1; i >= 0; i--) {

            int sample_id = idx * device_appdata->SCR_HEIGHT * device_appdata->samples_per_ray + idy * device_appdata->samples_per_ray + i;
            
            glm::vec4 blend_color = device_sample_colors[sample_id];

            fragment_color = glm::vec4(
                fragment_color.r * (1 - blend_color.a) + blend_color.r * blend_color.a,
                fragment_color.g * (1 - blend_color.a) + blend_color.g * blend_color.a,
                fragment_color.b * (1 - blend_color.a) + blend_color.b * blend_color.a,
                1.0f);
        }

        float r = fragment_color.r;
        float g = fragment_color.g;
        float b = fragment_color.b;
        float a = fragment_color.a;

        device_screen_colors[pixel_id] = glm::vec4(r,g,b,a);    
    }
}


__global__ void setDeviceOctreeCorrectlyKernel(Octree* device_octree, Node* device_octree_array) {
    int id = threadIdx.x;
    if (id == 0) {
        device_octree->octree_array = device_octree_array;
    }
}

__global__ void setDeviceNFCorrectlyKernel(NiftiFile* device_nf, float* device_nf_volume) {
    int id = threadIdx.x;
    if (id == 0) {
        device_nf->volume = device_nf_volume;
    }
}

__global__ void setDeviceTransferFunctionCorrectlyKernel(TransferFunction* device_transfer_function, mat_interval* device_material_intervals) {
    int id = threadIdx.x;
    if (id == 0) {
        device_transfer_function->material_intervals = device_material_intervals;
    
    }

}

__global__ void updateDeviceAppdataCameraKernel(AppData* device_appdata, glm::vec3 cameraPos, glm::vec3 cameraFront, glm::vec3 cameraRight, glm::vec3 cameraUp, glm::vec3 top_left_corner){
    int id = threadIdx.x;
    if (id == 0) {
        device_appdata->cameraPos = cameraPos;
        device_appdata->cameraFront = cameraFront;
        device_appdata->cameraRight = cameraRight;
        device_appdata->cameraUp = cameraUp;
        device_appdata->top_left_corner = top_left_corner;
    }
}

//setup primary ray direction with kernel call
void myCUDAspace::setupRayDirectionCUDA(glm::vec3* screen_pixel_ray_dir, AppData* appdata) {
    
    cudaError_t cudaStatus;

    glm::vec3* dev_res = nullptr;
    AppData* dev_appdata = nullptr; // give application data access to GPU
    
    int screen_size = appdata->SCR_WIDTH * appdata->SCR_HEIGHT;
    
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers
    cudaStatus = cudaMalloc((void**)&dev_res, screen_size * sizeof(glm::vec3));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_appdata, sizeof(AppData));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    
    //copy data to GPU
    cudaStatus = cudaMemcpy(dev_appdata, appdata, sizeof(AppData), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    //kernel call
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(appdata->SCR_WIDTH / threadsPerBlock.x, appdata->SCR_WIDTH / threadsPerBlock.y);
    rayDirectionKernel <<< numBlocks, threadsPerBlock >>> (dev_res, dev_appdata);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "rayDirectionKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(screen_pixel_ray_dir, dev_res, screen_size * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_res);
    cudaFree(dev_appdata);

}


void myCUDAspace::renderLoop(glm::vec4* host_screen, AppData* host_appdata, Octree* host_octree, TransferFunction* host_transfer_function, NiftiFile* host_nf) {

    cudaError_t cudaStatus;

    //setup cuda device
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cout << "ERROR set device" << std::endl;
        goto Error;
   }

    //DEVICE APPDATA - contains usefull variables and options
    AppData* device_application_data = nullptr;   
    {
        cudaStatus = cudaMalloc((void**)&device_application_data, sizeof(AppData));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_application_data malloc" << std::endl;
            goto Error;
        }
        cudaStatus = cudaMemcpy(device_application_data, host_appdata, sizeof(AppData), cudaMemcpyHostToDevice);    
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_application_data memcpy" << std::endl;
            goto Error;
        }
    }
    
    //DEVICE NIFTIFILE
    NiftiFile* device_nf = nullptr; 
    {
        cudaStatus = cudaMalloc((void**)&device_nf, sizeof(NiftiFile));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_nf malloc" << std::endl;
            goto Error;
        }
        cudaStatus = cudaMemcpy(device_nf, host_nf, sizeof(NiftiFile), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_nf memcpy" << std::endl;
            goto Error;
        }
    }

    //DEVICE OCTREE
    Octree* device_octree = nullptr;
    {
        cudaStatus = cudaMalloc((void**)&device_octree, sizeof(Octree));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_octree malloc " << std::endl;
            goto Error;
        }
        cudaStatus = cudaMemcpy(device_octree, host_octree, sizeof(Octree), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_octree memcpy" << std::endl;
            goto Error;
        }
    }

    Node* device_octree_array = nullptr;
    {
        cudaStatus = cudaMalloc((void**)&device_octree_array, host_octree->number_of_nodes * sizeof(Node));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_octree_array malloc" << std::endl;
            goto Error;
        }
        cudaStatus = cudaMemcpy(device_octree_array, host_octree->octree_array, host_octree->number_of_nodes * sizeof(Node), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_octree_array memcpy" << std::endl;
            goto Error;
        }

    }

    //correctly set octree
    setDeviceOctreeCorrectlyKernel << <1, 1 >> > (device_octree, device_octree_array);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching setDeviceOctreeCorrectlyKernel!\n", cudaStatus);
        goto Error;
    }

    //DEVICE TRANSFER FUNCTION
    TransferFunction* device_transfer_function = nullptr;
    {
        cudaStatus = cudaMalloc((void**)&device_transfer_function, sizeof(TransferFunction));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_transfer_function malloc" << std::endl;
            goto Error;
        }
        
        cudaStatus = cudaMemcpy(device_transfer_function, host_transfer_function, sizeof(TransferFunction), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_transfer_function memcpy" << std::endl;
            goto Error;
        }
    
    }
    mat_interval* device_material_intervals = nullptr;
    {
        cudaStatus = cudaMalloc((void**)&device_material_intervals, host_transfer_function->size * sizeof(mat_interval));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_material_intervals malloc" << std::endl;
            goto Error;
        }

        cudaStatus = cudaMemcpy(device_material_intervals, host_transfer_function->material_intervals, host_transfer_function->size * sizeof(mat_interval), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_material_intervals memcpy" << std::endl;
            goto Error;
        }
    }
    
    //correctly set transfer function
    setDeviceTransferFunctionCorrectlyKernel <<<1, 1 >>> (device_transfer_function, device_material_intervals);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching setDeviceTransferFunctionCorrectlyKernel!\n", cudaStatus);
        goto Error;
    }

    //DEVICE SCREEN
    int screen_size = host_appdata->SCR_WIDTH * host_appdata->SCR_HEIGHT;
    glm::vec4* device_screen = nullptr;
    {
        cudaStatus = cudaMalloc((void**)&device_screen, screen_size * sizeof(glm::vec4));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_screen malloc" << std::endl;
            goto Error;
        }
    }
    
    //DEVICE PRIMARY RAYS
    glm::vec3* device_primary_rays = nullptr;
    {
        cudaStatus = cudaMalloc((void**)&device_primary_rays, screen_size * sizeof(glm::vec3));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR primaray rays malloc" << std::endl;
            goto Error;
        }
    }
    
    //DEVICE SAMPLE COLORS
    glm::vec4* device_sample_colors = nullptr;
    {
        cudaStatus = cudaMalloc((void**)&device_sample_colors, static_cast<unsigned long long>(host_appdata->samples_per_ray) * screen_size * sizeof(glm::vec4));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_sample_colors malloc" << std::endl;
            goto Error;
        }
    }
    
    //MODEL MATRIX
    glm::mat4 modelAux = glm::mat4(1.0f);
    glm::mat4* host_modelAux = &modelAux;
    glm::mat4* device_modelAux = nullptr;
    {
        modelAux = glm::translate(modelAux, glm::vec3(0.5f, 0.5f, 0.5f));
        //modelAux = glm::rotate(modelAux, glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        //modelAux = glm::rotate(modelAux, glm::radians(90.0f), glm::vec3(-1.0f, 0.0f, 0.0f));
        
        cudaStatus = cudaMalloc((void**)&device_modelAux, sizeof(glm::mat4));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_modelAux malloc" << std::endl;
            goto Error;
        }
        cudaStatus = cudaMemcpy((void*)device_modelAux, (void*)host_modelAux, sizeof(glm::mat4), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_modelAux memcpy" << std::endl;
            goto Error;
        }
    }
    
    bool running = true; 
    int time_duration = 10000;
    int run_counter = 0;

    std::cout << "clock start" << std::endl;
    clock_t start;
    clock_t stop;
    start = clock();
    
    //renderloop
    while (running) {

        //update primary ray direction
        {
            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks(host_appdata->SCR_WIDTH / threadsPerBlock.x, host_appdata->SCR_WIDTH / threadsPerBlock.y);
            rayDirectionKernel <<< numBlocks, threadsPerBlock >>> (device_primary_rays, device_application_data);
            //sync threads
            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching rayDirectionKernel Kernel!\n", cudaStatus);
                goto Error;
            }
        }
        
        //calculate sample color
        {
            //TODO - figure out what are the correct parameters to call kernels
            dim3 threadsPerBlock(8, 8, 8);
            dim3 numBlocks(host_appdata->SCR_WIDTH / threadsPerBlock.x, host_appdata->SCR_WIDTH / threadsPerBlock.y, host_appdata->samples_per_ray / threadsPerBlock.z + 1);
            calculateSampleColor <<< numBlocks, threadsPerBlock >>> 
                    (device_sample_colors,
                    device_application_data,    //
                    device_octree,              //
                    device_transfer_function,   //
                    device_primary_rays,        //
                    device_modelAux,            //
                    host_nf->header.cal_max,    //maximum intensity
                    static_cast<unsigned long long>(host_appdata->samples_per_ray)* screen_size // sample size
                    );

            //sync threads
            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calculateSampleColor Kernel!\n", cudaStatus);
                goto Error;
            }
        }

        //blend sample colors
        {
            //TODO - figure out what are the correct parameters to call kernels
            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks(host_appdata->SCR_WIDTH / threadsPerBlock.x, host_appdata->SCR_WIDTH / threadsPerBlock.y);
            blendSampleColors <<< numBlocks, threadsPerBlock >>> (device_screen, device_sample_colors, device_application_data, screen_size);
            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching blendSampleColors Kernel!\n", cudaStatus);
                goto Error;
            }
        }

        //calculate pixel colors
        //prepare_screen_pixel_colors(device_octree, device_primary_rays, device_screen, device_application_data, device_transfer_function, device_nf);

        //display


        /*
        run_counter++;
        if (clock() - start > time_duration) {
            running = false;
        }*/

        running = false;
    }
    
    stop = clock();

    std::cout << "clock finish in " << stop - start << std::endl;
    std::cout << "run_counter = " << run_counter <<std::endl;


    std::cout << "cudaMemcpy Size: " << screen_size * sizeof(glm::vec4) << std::endl;
    std::cout << "host_screen: " << host_screen << std::endl;
    std::cout << "device_screen: " << device_screen << std::endl;


    cudaStatus = cudaMemcpy((void*) host_screen, (void*) device_screen, screen_size * sizeof(glm::vec4), cudaMemcpyDeviceToHost);
    {
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR host_screen memcpy device to host" << std::endl;
            goto Error;
        }
    }

    Error:
    std::cout << "cudaStatus Description: " << cudaGetErrorString(cudaStatus) << std::endl;
    
    //free alocated device memory
    cudaFree(device_modelAux);
    cudaFree(device_sample_colors);
    cudaFree(device_primary_rays);
    cudaFree(device_screen);
    cudaFree(device_transfer_function);
    cudaFree(device_material_intervals);

    cudaFree(device_octree_array);
    cudaFree(device_octree);

    cudaFree(device_application_data);
    cudaFree(device_nf);
}


//adapt this function to myApp.cu, this is an example fo how to setup a renderloop
void myCUDAspace::renderLoop2(glm::vec4* host_screen, AppData* host_appdata, Octree* host_octree, TransferFunction* host_transfer_function, NiftiFile* host_nf) {

    cudaError_t cudaStatus;

    //setup cuda device
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cout << "ERROR set device" << std::endl;
        goto Error;
    }

    glm::vec4* device_screen = nullptr;
    AppData* device_application_data = nullptr;
    Octree* device_octree = nullptr;
    Node* device_octree_array = nullptr;
    TransferFunction* device_transfer_function = nullptr;
    mat_interval* device_material_intervals = nullptr;
    NiftiFile* device_nf = nullptr;
    float* device_nf_volume = nullptr;
    glm::vec3* device_primary_rays = nullptr;
    glm::vec4* device_sample_colors = nullptr;
    glm::mat4* device_modelAux = nullptr;

    int screen_size = host_appdata->SCR_WIDTH * host_appdata->SCR_HEIGHT;

    allocateDeviceMemory2(
        host_screen, host_appdata, host_octree, host_transfer_function, host_nf,
        &device_screen, &device_application_data, &device_octree, &device_octree_array,
        &device_transfer_function, &device_material_intervals, &device_nf, &device_nf_volume,
        &device_primary_rays, &device_sample_colors, &device_modelAux);

    bool running = true;
    int time_duration = 10000;
    int run_counter = 0;

    //std::cout << "clock start" << std::endl;
    clock_t start;
    clock_t stop;
    start = clock();

    //renderloop
    while (running) {
    
        updatePrimaryRayDirection(host_appdata, device_application_data, device_primary_rays);
    
        getSampleColors(host_appdata, host_nf, device_application_data, device_octree, device_transfer_function, device_primary_rays, device_sample_colors, device_modelAux, screen_size);
    
        blendSampleColors(host_appdata, device_application_data, device_screen, device_sample_colors, screen_size);
    
        running = false; // CHANGE THIS to do multiple runs
    }

    stop = clock();

    std::cout << "clock finish in " << stop - start << std::endl;
    std::cout << "run_counter = " << run_counter << std::endl;


    std::cout << "cudaMemcpy Size: " << screen_size * sizeof(glm::vec4) << std::endl;
    std::cout << "host_screen: " << host_screen << std::endl;
    std::cout << "device_screen: " << device_screen << std::endl;

    cudaStatus = cudaMemcpy((void*)host_screen, (void*)device_screen, screen_size * sizeof(glm::vec4), cudaMemcpyDeviceToHost);
    {
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR host_screen memcpy device to host" << std::endl;
            goto Error;
        }
    }

Error:
    std::cout << "cudaStatus Description: " << cudaGetErrorString(cudaStatus) << std::endl;

    deallocateDeviceMemory(device_screen, device_application_data, device_octree, device_octree_array,
        device_transfer_function, device_material_intervals, device_nf, device_nf_volume,
        device_primary_rays, device_sample_colors, device_modelAux);

}

cudaError_t myCUDAspace::allocateDeviceMemory(
    glm::vec4* host_screen, AppData* host_appdata, Octree* host_octree, TransferFunction* host_transfer_function, NiftiFile* host_nf,
    glm::vec4* device_screen, AppData* device_application_data, Octree* device_octree, Node* device_octree_array,
    TransferFunction* device_transfer_function, mat_interval* device_material_intervals, NiftiFile* device_nf,
    glm::vec3* device_primary_rays, glm::vec4* device_sample_colors, glm::mat4* device_modelAux) {

    cudaError_t cudaStatus = cudaSuccess;

    //setup cuda device
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cout << "ERROR set device" << std::endl;
        goto Error;
    }

    //DEVICE APPDATA - contains usefull variables and options
    device_application_data = nullptr;
    {
        cudaStatus = cudaMalloc((void**)&device_application_data, sizeof(AppData));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_application_data malloc" << std::endl;
            goto Error;
        }
        cudaStatus = cudaMemcpy(device_application_data, host_appdata, sizeof(AppData), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_application_data memcpy" << std::endl;
            goto Error;
        }
    }

    //DEVICE NIFTIFILE
    device_nf = nullptr;
    {
        cudaStatus = cudaMalloc((void**)&device_nf, sizeof(NiftiFile));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_nf malloc" << std::endl;
            goto Error;
        }
        cudaStatus = cudaMemcpy(device_nf, host_nf, sizeof(NiftiFile), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_nf memcpy" << std::endl;
            goto Error;
        }
    }

    //DEVICE OCTREE
    device_octree = nullptr;
    {
        cudaStatus = cudaMalloc((void**)&device_octree, sizeof(Octree));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_octree malloc " << std::endl;
            goto Error;
        }
        cudaStatus = cudaMemcpy(device_octree, host_octree, sizeof(Octree), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_octree memcpy" << std::endl;
            goto Error;
        }
    }

    device_octree_array = nullptr;
    {
        cudaStatus = cudaMalloc((void**)&device_octree_array, host_octree->number_of_nodes * sizeof(Node));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_octree_array malloc" << std::endl;
            goto Error;
        }
        cudaStatus = cudaMemcpy(device_octree_array, host_octree->octree_array, host_octree->number_of_nodes * sizeof(Node), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_octree_array memcpy" << std::endl;
            goto Error;
        }
    }

    //correctly set octree
    setDeviceOctreeCorrectlyKernel << <1, 1 >> > (device_octree, device_octree_array);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching setDeviceOctreeCorrectlyKernel!\n", cudaStatus);
        goto Error;
    }

    //DEVICE TRANSFER FUNCTION
    device_transfer_function = nullptr;
    {
        cudaStatus = cudaMalloc((void**)&device_transfer_function, sizeof(TransferFunction));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_transfer_function malloc" << std::endl;
            goto Error;
        }

        cudaStatus = cudaMemcpy(device_transfer_function, host_transfer_function, sizeof(TransferFunction), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_transfer_function memcpy" << std::endl;
            goto Error;
        }

    }
    device_material_intervals = nullptr;
    {
        cudaStatus = cudaMalloc((void**)&device_material_intervals, host_transfer_function->size * sizeof(mat_interval));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_material_intervals malloc" << std::endl;
            goto Error;
        }

        cudaStatus = cudaMemcpy(device_material_intervals, host_transfer_function->material_intervals, host_transfer_function->size * sizeof(mat_interval), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_material_intervals memcpy" << std::endl;
            goto Error;
        }
    }

    //correctly set transfer function
    setDeviceTransferFunctionCorrectlyKernel << <1, 1 >> > (device_transfer_function, device_material_intervals);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching setDeviceTransferFunctionCorrectlyKernel!\n", cudaStatus);
        goto Error;
    }

    //DEVICE SCREEN
    int screen_size = host_appdata->SCR_WIDTH * host_appdata->SCR_HEIGHT;
    device_screen = nullptr;
    {
        cudaStatus = cudaMalloc((void**)&device_screen, screen_size * sizeof(glm::vec4));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_screen malloc" << std::endl;
            goto Error;
        }
    }

    //DEVICE PRIMARY RAYS
    device_primary_rays = nullptr;
    {
        cudaStatus = cudaMalloc((void**)&device_primary_rays, screen_size * sizeof(glm::vec3));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR primaray rays malloc" << std::endl;
            goto Error;
        }
    }

    //DEVICE SAMPLE COLORS
    device_sample_colors = nullptr;
    {
        cudaStatus = cudaMalloc((void**)&device_sample_colors, static_cast<unsigned long long>(host_appdata->samples_per_ray) * screen_size * sizeof(glm::vec4));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_sample_colors malloc" << std::endl;
            goto Error;
        }
    }

    //MODEL MATRIX
    glm::mat4 modelAux = glm::mat4(1.0f);
    glm::mat4* host_modelAux = &modelAux;
    device_modelAux = nullptr;
    {
        modelAux = glm::translate(modelAux, glm::vec3(0.5f, 0.5f, 0.5f));
        modelAux = glm::rotate(modelAux, glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        modelAux = glm::rotate(modelAux, glm::radians(90.0f), glm::vec3(-1.0f, 0.0f, 0.0f));

        cudaStatus = cudaMalloc((void**)&device_modelAux, sizeof(glm::mat4));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_modelAux malloc" << std::endl;
            goto Error;
        }
        cudaStatus = cudaMemcpy((void*)device_modelAux, (void*)host_modelAux, sizeof(glm::mat4), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_modelAux memcpy" << std::endl;
            goto Error;
        }
    }


    Error:
    return cudaStatus;
}

cudaError_t myCUDAspace::allocateDeviceMemory2(
    glm::vec4* host_screen, AppData* host_appdata, Octree* host_octree, TransferFunction* host_transfer_function, NiftiFile* host_nf,
    glm::vec4** device_screen, AppData** device_application_data, Octree** device_octree, Node** device_octree_array,
    TransferFunction** device_transfer_function, mat_interval** device_material_intervals, NiftiFile** device_nf, float** device_nf_volume,
    glm::vec3** device_primary_rays, glm::vec4** device_sample_colors, glm::mat4** device_modelAux) {

    cudaError_t cudaStatus = cudaSuccess;

    //setup cuda device
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cout << "ERROR set device" << std::endl;
        goto Error;
    }

    //DEVICE APPDATA - contains usefull variables and options
    {
        cudaStatus = cudaMalloc((void**)device_application_data, sizeof(AppData));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_application_data malloc" << std::endl;
            goto Error;
        }
        cudaStatus = cudaMemcpy(*device_application_data, host_appdata, sizeof(AppData), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_application_data memcpy" << std::endl;
            goto Error;
        }
    }

    //DEVICE NIFTIFILE
    {
        cudaStatus = cudaMalloc((void**)device_nf, sizeof(NiftiFile));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_nf malloc" << std::endl;
            goto Error;
        }
        cudaStatus = cudaMemcpy(*device_nf, host_nf, sizeof(NiftiFile), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_nf memcpy" << std::endl;
            goto Error;
        }
        
        int volume_size = 1;
        for (int i = 0; i < host_nf->header.dim[0]; i++) {
            volume_size = volume_size * host_nf->header.dim[i + 1];
        }

        cudaStatus = cudaMalloc((void**)device_nf_volume, volume_size * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_nf_volume malloc" << std::endl;
            goto Error;
        }
        cudaStatus = cudaMemcpy(*device_nf_volume, host_nf->volume, volume_size * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_nf_volume memcpy" << std::endl;
            goto Error;
        }

        //correctly set device_nf
        setDeviceNFCorrectlyKernel << <1, 1 >> > (*device_nf, *device_nf_volume);
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching setDeviceNFCorrectlyKernel!\n", cudaStatus);
            goto Error;
        }

    }

    //DEVICE OCTREE
    {
        cudaStatus = cudaMalloc((void**)device_octree, sizeof(Octree));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_octree malloc " << std::endl;
            goto Error;
        }
        cudaStatus = cudaMemcpy(*device_octree, host_octree, sizeof(Octree), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_octree memcpy" << std::endl;
            goto Error;
        }
    }

    {
        cudaStatus = cudaMalloc((void**)device_octree_array, host_octree->number_of_nodes * sizeof(Node));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_octree_array malloc" << std::endl;
            goto Error;
        }
        cudaStatus = cudaMemcpy(*device_octree_array, host_octree->octree_array, host_octree->number_of_nodes * sizeof(Node), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_octree_array memcpy" << std::endl;
            goto Error;
        }
    }

    //correctly set octree
    setDeviceOctreeCorrectlyKernel << <1, 1 >> > (*device_octree, *device_octree_array);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching setDeviceOctreeCorrectlyKernel!\n", cudaStatus);
        goto Error;
    }

    //DEVICE TRANSFER FUNCTION
    {
        cudaStatus = cudaMalloc((void**)device_transfer_function, sizeof(TransferFunction));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_transfer_function malloc" << std::endl;
            goto Error;
        }

        cudaStatus = cudaMemcpy(*device_transfer_function, host_transfer_function, sizeof(TransferFunction), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_transfer_function memcpy" << std::endl;
            goto Error;
        }

    }
    
    {
        cudaStatus = cudaMalloc((void**)device_material_intervals, host_transfer_function->size * sizeof(mat_interval));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_material_intervals malloc" << std::endl;
            goto Error;
        }

        cudaStatus = cudaMemcpy(*device_material_intervals, host_transfer_function->material_intervals, host_transfer_function->size * sizeof(mat_interval), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_material_intervals memcpy" << std::endl;
            goto Error;
        }
    }

    //correctly set transfer function
    setDeviceTransferFunctionCorrectlyKernel << <1, 1 >> > (*device_transfer_function, *device_material_intervals);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching setDeviceTransferFunctionCorrectlyKernel!\n", cudaStatus);
        goto Error;
    }

    //DEVICE SCREEN
    int screen_size = host_appdata->SCR_WIDTH * host_appdata->SCR_HEIGHT;
    {
        cudaStatus = cudaMalloc((void**)device_screen, screen_size * sizeof(glm::vec4));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_screen malloc" << std::endl;
            goto Error;
        }
    }

    //DEVICE PRIMARY RAYS
    {
        cudaStatus = cudaMalloc((void**)device_primary_rays, screen_size * sizeof(glm::vec3));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR primaray rays malloc" << std::endl;
            goto Error;
        }
    }

    //DEVICE SAMPLE COLORS
    {
        cudaStatus = cudaMalloc((void**)device_sample_colors, static_cast<unsigned long long>(host_appdata->samples_per_ray) * screen_size * sizeof(glm::vec4));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_sample_colors malloc" << std::endl;
            goto Error;
        }
    }

    //MODEL MATRIX
    glm::mat4 modelAux = glm::mat4(1.0f);
    glm::mat4* host_modelAux = &modelAux;
    {
        //modelAux = glm::scale(modelAux, glm::vec3(-1.0f, -1.0f, -1.0f));
        modelAux = glm::translate(modelAux, glm::vec3(0.5f, 0.5f, 0.5f));
        //modelAux = glm::rotate(modelAux, glm::radians(180.0f), host_appdata->cameraRight);
        
        cudaStatus = cudaMalloc((void**)device_modelAux, sizeof(glm::mat4));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_modelAux malloc" << std::endl;
            goto Error;
        }
        cudaStatus = cudaMemcpy((void*)*device_modelAux, (void*)host_modelAux, sizeof(glm::mat4), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_modelAux memcpy" << std::endl;
            goto Error;
        }
    }


Error:
    return cudaStatus;
}


//CAREFUL ITS NOT CURRENTLY EVALUATING THE CUDA STATUS OF DEALLOCATING DEVICE CUDA RESOURCES
cudaError_t myCUDAspace::deallocateDeviceMemory(glm::vec4* device_screen, AppData* device_application_data, Octree* device_octree, Node* device_octree_array,
    TransferFunction* device_transfer_function, mat_interval* device_material_intervals, NiftiFile* device_nf, float* device_nf_volume,
    glm::vec3* device_primary_rays, glm::vec4* device_sample_colors, glm::mat4* device_modelAux) {

    cudaError_t cudaStatus = cudaSuccess;
   
    cudaFree(device_modelAux);
    cudaFree(device_sample_colors);
    cudaFree(device_primary_rays);
    cudaFree(device_screen);
    cudaFree(device_transfer_function);
    cudaFree(device_material_intervals);

    cudaFree(device_octree_array);
    cudaFree(device_octree);

    cudaFree(device_application_data);
    cudaFree(device_nf_volume);
    cudaFree(device_nf);

    return cudaStatus;
}

//updates primary ray direction
cudaError_t myCUDAspace::updatePrimaryRayDirection(AppData* host_appdata, AppData* device_application_data, glm::vec3* device_primary_rays) {
    cudaError_t cudaStatus = cudaSuccess;
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(host_appdata->SCR_WIDTH / threadsPerBlock.x+1, host_appdata->SCR_WIDTH / threadsPerBlock.y+1);
    rayDirectionKernel << < numBlocks, threadsPerBlock >> > (device_primary_rays, device_application_data);
    //sync threads
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching rayDirectionKernel Kernel!\n", cudaStatus);
    }

    return cudaStatus;
}

//updates camera location
cudaError_t myCUDAspace::updateCameraLocation(AppData* host_appdata, AppData* device_appdata) {
    cudaError_t cudaStatus = cudaSuccess;

    updateDeviceAppdataCameraKernel << <1, 1 >> >(device_appdata, host_appdata->cameraPos, host_appdata->cameraFront, host_appdata->cameraRight, host_appdata->cameraUp, host_appdata->top_left_corner);

    //sync threads
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching setDeviceAppdataCameraKernel!\n", cudaStatus);
    }

    return cudaStatus;
}

cudaError_t myCUDAspace::getSampleColors(
    AppData* host_appdata,
    NiftiFile* host_nf,
    AppData* device_application_data,
    Octree* device_octree,
    TransferFunction* device_transfer_function,
    glm::vec3* device_primary_rays,
    glm::vec4* device_sample_colors,
    glm::mat4* device_modelAux,
    int screen_size
    ){

    cudaError_t cudaStatus = cudaSuccess;

    //calculate sample color
    
    //TODO - figure out what are the correct parameters to call kernels
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks(host_appdata->SCR_WIDTH / threadsPerBlock.x+1, host_appdata->SCR_WIDTH / threadsPerBlock.y+1, host_appdata->samples_per_ray / threadsPerBlock.z + 1);
    calculateSampleColor << < numBlocks, threadsPerBlock >> >
        (device_sample_colors,
            device_application_data,    //
            device_octree,              //
            device_transfer_function,   //
            device_primary_rays,        //
            device_modelAux,            //
            host_nf->header.cal_max,    //maximum intensity
            static_cast<unsigned long long>(host_appdata->samples_per_ray) * screen_size // sample size
            );

    //sync threads
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calculateSampleColor Kernel!\n", cudaStatus);
    }

    return cudaStatus;
}

cudaError_t myCUDAspace::getSampleColorsFromNF(
    glm::vec4* device_sample_colors, //output
    AppData* host_appdata,
    NiftiFile* host_nf,
    AppData* device_appdata,
    NiftiFile* device_nf,
    TransferFunction* device_transfer_function,
    int screen_size) {

    cudaError_t cudaStatus = cudaSuccess;


    //put the sample grid in the world
    glm::mat4 modelCam = glm::mat4(1.0f);
  
    modelCam = glm::translate(modelCam,
        glm::vec3(
            -host_appdata->real_screen_width / 2.0f,
            -host_appdata->real_screen_height / 2.0f,
            0.0f//-host_appdata->viewplane_distance  / 2.0f
        )
    );
    modelCam = glm::scale(modelCam, 
        glm::vec3(
            (host_appdata->real_screen_width / host_appdata->SCR_WIDTH),
            host_appdata->real_screen_height / host_appdata->SCR_HEIGHT ,
            -host_appdata->viewplane_distance / host_appdata->samples_per_ray
        )
    );

    glm::mat4 projectionCam = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 10.1f);
    //projectionCam = glm::inverse(projectionCam);

    glm::mat4 viewCam = glm::lookAt(host_appdata->cameraPos,glm::vec3(0.0f,0.0f,0.0f),host_appdata->cameraUp);
    viewCam = glm::inverse(viewCam);

    //glm::mat4 transformCam =  viewCam * projectionCam * modelCam;
    glm::mat4 transformCam =  viewCam * modelCam;

    glm::mat4 toVolumeTransform = glm::mat4(1.0f);
    
    glm::mat4 tvtranslate1 = glm::translate(glm::mat4(1.0f),
        glm::vec3(0.5f, 0.5f, 0.5f));
    glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(host_nf->longest_dimension, host_nf->longest_dimension, host_nf->longest_dimension));
    glm::mat4 tvtranslate2 = glm::translate(glm::mat4(1.0f),
        glm::vec3(
            host_nf->header.dim[1] / 2.0f - host_nf->longest_dimension / 2.0f,
            host_nf->header.dim[2] / 2.0f - host_nf->longest_dimension / 2.0f,
            host_nf->header.dim[3] / 2.0f - host_nf->longest_dimension / 2.0f
        ));

    toVolumeTransform = tvtranslate1 * toVolumeTransform;
    toVolumeTransform = scale * toVolumeTransform;
    toVolumeTransform = tvtranslate2 * toVolumeTransform;

    //glm::mat4 transform = toVolumeTransform * transformCam;
    glm::mat4 transform = transformCam;

    glm::mat4 aux_transform = toVolumeTransform;

    glm::mat4* device_transform = nullptr;
    cudaStatus = cudaMalloc((void**)&device_transform, sizeof(glm::mat4));
    cudaStatus = cudaMemcpy(device_transform, &transform, sizeof(glm::mat4), cudaMemcpyHostToDevice);
    

    //TODO - figure out what are the correct parameters to call kernels
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks(host_appdata->SCR_WIDTH / threadsPerBlock.x + 1, host_appdata->SCR_WIDTH / threadsPerBlock.y + 1, host_appdata->samples_per_ray / threadsPerBlock.z + 1);
    dim3 size(host_appdata->SCR_WIDTH, host_appdata->SCR_HEIGHT, host_appdata->samples_per_ray);
    
    //std::cout << "nublocks: " << numBlocks.x << " " << numBlocks.y << " " << numBlocks.z << std::endl;

    getColorFromNF << < numBlocks, threadsPerBlock >> > (
        device_sample_colors,
        device_appdata,
        device_nf,
        device_transfer_function,
        device_transform,
        aux_transform,
        modelCam,
        viewCam,
        size
    );



    //sync threads
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching getColorFromNF Kernel!\n", cudaStatus);
    }

    cudaFree(device_transform);

    return cudaStatus;
}


cudaError_t myCUDAspace::blendSampleColors(AppData* host_appdata,
    AppData* device_application_data,
    glm::vec4* device_screen,
    glm::vec4* device_sample_colors,
    int screen_size){
    
    cudaError_t cudaStatus = cudaSuccess;

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(host_appdata->SCR_WIDTH / threadsPerBlock.x + 1, host_appdata->SCR_WIDTH / threadsPerBlock.y + 1);
    blendSampleColors << < numBlocks, threadsPerBlock >> > (device_screen, device_sample_colors, device_application_data, screen_size);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching blendSampleColors Kernel!\n", cudaStatus);
    }

    return cudaStatus;
}

cudaError_t resize(AppData* host_appdata, glm::vec3* host_primary_rays, glm::vec4* host_sample_colors, glm::vec4* host_screen,
    AppData** device_appdata, glm::vec3** device_primary_rays, glm::vec4** device_sample_colors, glm::vec4** device_screen)
{
    cudaError_t cudaStatus = cudaSuccess;
    
    //dealocate old device buffers, no need to dealocate octree or niftiFile, since these do not change
    cudaFree(*device_appdata);
    cudaFree(*device_primary_rays);
    cudaFree(*device_sample_colors);
    cudaFree(*device_screen);

    //realocate device buffers

    //DEVICE APPDATA - contains usefull variables and options
    {
        cudaStatus = cudaMalloc((void**)device_appdata, sizeof(AppData));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_application_data malloc" << std::endl;
            goto Error;
        }
        cudaStatus = cudaMemcpy(*device_appdata, host_appdata, sizeof(AppData), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_application_data memcpy" << std::endl;
            goto Error;
        }
    }

     //DEVICE SCREEN
    int screen_size = host_appdata->SCR_WIDTH * host_appdata->SCR_HEIGHT;
    {
        cudaStatus = cudaMalloc((void**)device_screen, screen_size * sizeof(glm::vec4));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_screen malloc" << std::endl;
            goto Error;
        }
    }

    //DEVICE PRIMARY RAYS
    {
        cudaStatus = cudaMalloc((void**)device_primary_rays, screen_size * sizeof(glm::vec3));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR primaray rays malloc" << std::endl;
            goto Error;
        }
    }

    //DEVICE SAMPLE COLORS
    {
        cudaStatus = cudaMalloc((void**)device_sample_colors, static_cast<unsigned long long>(host_appdata->samples_per_ray) * screen_size * sizeof(glm::vec4));
        if (cudaStatus != cudaSuccess) {
            std::cout << "ERROR device_sample_colors malloc" << std::endl;
            goto Error;
        }
    }

    Error:

    return cudaStatus;
}
