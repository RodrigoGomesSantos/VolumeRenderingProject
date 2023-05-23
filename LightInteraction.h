#pragma once

#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


class LightInteraction
{
private:
	float D = 10.0f;
	//int scattering_probability = 50;
	float e = 2.58; // "e" growth constant
	//float extinction_coefficient = 0.5f; //value for test
	//float scattering_coefficient = 0.5f; //value for test
	//float absorption_coefficient = 0.5f; //value for test
	
	int samples_per_ray = 8;

	glm::vec3 getRandomDirection();

public:
	float radiance(float x, glm::vec3 w);
	float in_radiance(float x, glm::vec3 w);
	float optical_depth(float x, float xl);
	float inscattering(float xl, glm::vec3 w);
	float phase_function(glm::vec3 w, glm::vec3 wl);
	float extinction(float x, float xl);
	float extinction_coefficient(float t);
	float scattering_probability(float xl);

	

};