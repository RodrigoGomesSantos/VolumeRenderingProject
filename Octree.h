#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <list>

#include "BinaryLoader.h"


struct Node{
	int depth;
	float maximum_value;
	float minimum_value;
	glm::vec3 lower_corner;
	glm::vec3 upper_corner;	
};

class Octree {

public:

	Node* octree_array; //octree array
	unsigned int maximum_depth; //root has depth = 0
	NiftiFile* nf;
	unsigned int longest_dimension;
	unsigned int number_of_nodes;

	Octree(NiftiFile* nf);
	~Octree();

	void initializeOctreeNodes();
	void updateOctreeNodes();
	void createNode(int index, int depth, glm::vec3 lower_corner, glm::vec3 upper_corner);
	
	float getIntensity(glm::vec3 point);
	float searchPointGetIntensity(unsigned int index, glm::vec3 point);
	float searchPointGetIntensityPrinted(unsigned int index, glm::vec3 point);
	bool isLeaf(unsigned int index);
	bool isInside(unsigned int index, glm::vec3 p);

	void updateNode(unsigned int index);

	//DEVICE FUNCTIONS - can only be called inside kernels

	__device__ bool device_isInside(unsigned int index, glm::vec3 p);

	__device__ float device_getIntensity(glm::vec3 point);

	__device__ float device_searchPointGetIntensity(unsigned int index, glm::vec3 point);

};




