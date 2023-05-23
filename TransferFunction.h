#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector> 
#include <tuple> 

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Material.h"

struct mat_interval {
	Material::Material material;
	float lower_bound;
	float higher_bound;
};

class TransferFunction {

public:
	char* charfile = nullptr;
	mat_interval* material_intervals;
	int size;

	TransferFunction();
	~TransferFunction(); 

	/*
	* retrieves value associated material
	* value - values outside of the bounds specified in the transfer function result in Material::empty.
	* WARNING: bounds are included in the range
	*/
	__host__ __device__ Material::Material* TransferFunction::getMaterial(float value);

	void print();

};