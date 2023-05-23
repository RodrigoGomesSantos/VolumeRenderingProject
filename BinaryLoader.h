#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string>
#include "nifti2.h"
#include "nifti1.h"

union NiftiHeader {
	nifti_1_header nifti1;
	nifti_2_header nifti2;
};


class NiftiFile {

private:
	std::string fileName;
	int niftiType;
	
	

	void setTotalDim();
	
public:
	nifti_2_header header;
	float* volume;
	int longest_dimension;
	int totaldim;
	bool sphereTest;

	NiftiFile(std::string filename);
	NiftiFile(); //empty for sphere test
	//NiftiFile(float value); //empty for sphere test
	void foo();

	~NiftiFile();


	__host__ __device__ int transformVector3Position(glm::vec3 v);
	
	__host__ __device__ bool isInside(glm::vec3 point);
	
	void displayNIFTI2Header();
	__host__ __device__ glm::vec3 toVolumeSpace(glm::vec3 point);
	int loadFileToMem();
	int loadSphereToMem();
	int loadZEROCornerSphereToMem();

};
