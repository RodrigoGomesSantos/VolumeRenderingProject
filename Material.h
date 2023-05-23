#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace Material
{
	struct Material {
		char* name;
		glm::vec4 color; //rgba

		/*possible values inside interval[-1.0, 1.0].
		 -1 backscattering
		 0 diffuse
		 1 frontscattering
		*/
		double Henyey_Greenstein_scattering; 
	};

	enum MaterialId {
		red, green, blue,
		
		bone, muscle, eye, brain, cerebelum, cerebrospinal_fluid, brain_stem,

		glass, empty, air , default
	};

	__host__ __device__ Material getMaterialFromID(MaterialId id);
	/*
	//default material
	static Material default_mat{
		"DEFAULT",
			glm::vec4(0.0f, 0.0f, 0.0f, 0.1f),
			0.0f //Henyey_Greenstein_scattering
	};

	//TODO - tweek values
	static Material bone{
		"BONE",
		glm::vec4(0.8f, 0.8f, 0.7f, 1.0f),
		1.0f //Henyey_Greenstein_scattering
	};

	static Material muscle{
		"MUSCLE",
		glm::vec4(0.7f, 0.4f, 0.3f, 0.3f),
		1.0f //Henyey_Greenstein_scattering
	 };

	static Material brain{
		"BRAIN",
		glm::vec4(0.8f, 0.5f, 0.5f, 0.5f),
		1.0f //Henyey_Greenstein_scattering
	 };

	//EMPTY MATERIAL
	static Material empty{
		"EMPTY",
		glm::vec4(0.0f, 0.0f, 0.0f, 0.0f),
		1.0f //Henyey_Greenstein_scattering
	};

	static Material red{
		"RED",
		glm::vec4(1.0f,0.0f,0.0f,1.0f),
		1.0f //Henyey_Greenstein_scattering
	};

	static Material green{
		"GREEN",
		glm::vec4(0.0f,1.0f,0.0f,0.1f),
		1.0f //Henyey_Greenstein_scattering
	};

	static Material blue{
		"BLUE",
		glm::vec4(0.0f,0.0f,1.0f,0.1f),
		1.0f //Henyey_Greenstein_scattering
	};*/

};

