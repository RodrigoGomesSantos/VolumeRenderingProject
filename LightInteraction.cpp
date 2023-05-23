#include "LightInteraction.h"

#include <random>

float LightInteraction::radiance(float x, glm::vec3 w)
{
	glm::vec3 direction = glm::vec3(0.0f, 0.0f, 1.0f);
	float travel_distance = D / samples_per_ray;
	float res = 0.0f;
	for (int i = 0; i < samples_per_ray; i++) {
		float xl = i * travel_distance;
		res += extinction(x, xl) * scattering_probability(xl) * inscattering(xl,w);
	}

	return res;
}

float LightInteraction::in_radiance(float x, glm::vec3 w)
{//TODO - carefull might end up recursive
	glm::vec3 direction = glm::vec3(0.0f, 0.0f, 1.0f);
	float travel_distance = D / samples_per_ray;
	float res = 0.0f;
	for (int i = 0; i < samples_per_ray; i++) {
		float xl = i * travel_distance;
		res += extinction(x, xl) * scattering_probability(xl) * inscattering(xl, w);
	}

	return res;
}

float LightInteraction::optical_depth(float x, float xl)
{
	//for integral calculation
	int ndx = 10;    //number of smaples
	float dx = (xl - x) / ndx;      //sample distance
	float res = 0.0f;
	for (int i = 0; i < ndx; i++) {
		res += extinction_coefficient(i * dx);
	}

	return 0.0f;
}

float LightInteraction::inscattering(float xl, glm::vec3 w)
{//HERE
	int number_of_sample_directions = 10;
	
	float res = 0.0f;
	//shoot in random directions
	for (int i = 0; i < number_of_sample_directions; i++) {
		glm::vec3 wl = getRandomDirection();
		res += phase_function(w, wl) * in_radiance(xl,wl);
	}
	
	return res;
}

float LightInteraction::phase_function(glm::vec3 w, glm::vec3 wl)
{
	//TODO
	return 0.0f;
}

float LightInteraction::extinction(float x, float xl)
{
	return exp(-optical_depth(x, xl));
}

float LightInteraction::extinction_coefficient(float t)
{
	//TODO
	return 0.0f; 
}

float LightInteraction::scattering_probability(float xl)
{
	return 0.0f;
}

glm::vec3 LightInteraction::getRandomDirection() {

	
	return normalize(glm::vec3(
		(rand() / RAND_MAX) * 2 - 1,
		(rand() / RAND_MAX) * 2 - 1,
		(rand() / RAND_MAX) * 2 - 1));
}


