
#include "Material.h"

namespace Material {

	__host__ __device__ Material getMaterialFromID(MaterialId id) {
		Material result;

		//using https://imagecolorpicker.com/ to get the RGB values for each material

		switch (id) { 
		case MaterialId::red:
			result = Material{ "RED",glm::vec4(1.0f, 0.0f, 0.0f, 1.0f), 0.0f };
			break;

		case MaterialId::green:
			result = Material{ "GREEN",glm::vec4(0.0f, 1.0f, 0.0f, 1.0f), 0.0f };
			break;

		case MaterialId::blue:
			result = Material{ "BLUE",glm::vec4(0.0f, 0.0f, 1.0f, 1.0f), 0.0f };
			break;

		case MaterialId::glass:
			result = Material{ "GLASS",glm::vec4(0.2f, 0.2f, 0.2f, 0.1f), 0.0f };
			break;

		case MaterialId::muscle: //124,9,42
			result = Material{ "MUSCLE",glm::vec4(124.0f / 255.0f, 9.0f / 255.0f, 42.0f / 255.0f, 0.3f), 0.0f };
			break;

		case MaterialId::empty:
			result = Material{ "EMPTY",glm::vec4(0.0f, 0.0f, 0.0f, 0.0f), 0.0f };
			break;

		case MaterialId::bone:
			result = Material{ "BONE",glm::vec4(241.0f / 255.0f, 218.0f / 255.0f, 202.0f / 255.0f, 0.3f), 0.0f };
			break;

		case MaterialId::brain : ///
			result = Material{ "BRAIN",glm::vec4(223.0f / 255.0f, 155.0f / 255.0f, 141.0f / 255.0f, 0.7f), 0.0f };
			break;


		case MaterialId::brain_stem:
			result = Material{ "BRAIN_STEM",glm::vec4(241.0f / 255.0f, 218.0f / 255.0f, 202.0f / 255.0f, 0.9f), 0.0f };
			break;

		case MaterialId::cerebelum:
			result = Material{ "CEREBELUM",glm::vec4(241.0f / 255.0f, 218.0f / 255.0f, 202.0f / 255.0f, 0.9f), 0.0f };
			break;

		case MaterialId::cerebrospinal_fluid:
			result = Material{ "CEREBROSPINAL_FLUID",glm::vec4(241.0f / 255.0f, 218.0f / 255.0f, 202.0f / 255.0f, 0.9f), 0.0f };
			break;

		case MaterialId::eye:
			result = Material{ "EYE",glm::vec4(241.0f / 255.0f, 218.0f / 255.0f, 202.0f / 255.0f, 0.9f), 0.0f };
			break;

		default:
			result = Material{ "DEFAULT",glm::vec4(1.0f, 0.0f, 1.0f, 0.1f), 0.0f };
			break;
		}

		return result;
	};

}