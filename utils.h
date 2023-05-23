#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define _USE_MATH_DEFINES
#include <math.h>

#include <string>

//ALGORITHMS
#define POINT 0 
#define VRC 1
#define TEXTURESHADER 2
#define DISPLAYCUDASHADER 3
#define CR 4
#define TEST 5

/*
This header contains some structures to wrap information
*/

struct AppData {
	
	// settings
	const bool simple_point = true; // displays only one point per voxel
	const bool conic = false;
	const bool sphere_test = false;
	const bool useCUDA = false;
	int algorithm = POINT;
	clock_t last_key_time = 0;
	unsigned int key_delay = 500;

	//viewport
	const unsigned int SCR_WIDTH = 300;
	const unsigned int SCR_HEIGHT = 300;
	const glm::vec4 BACKGROUND_COLOUR = glm::vec4(0.2f, 0.2f, 0.2f, 1.0f);

	//camera global variables
	glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 1.0f);
	glm::vec3 cameraFront = glm::normalize(glm::vec3(0.0f, 0.0f, 0.0f) - cameraPos);

	glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
	glm::vec3 cameraRight = glm::normalize(glm::cross(cameraFront, up));
	glm::vec3 cameraUp = glm::normalize(glm::cross(cameraRight, cameraFront));

	//time variables
	float deltaTime = 0.0f;	// Time between current frame and last frame
	float lastFrame = 0.0f; // Time of last frame


	float viewplane_distance = 2.0f;
	float view_angle = M_PI / 4;

	//this points to the top left corner of the screen
	//float real_screen_width = 2 * std::tan(view_angle) * viewplane_distance;	// for conic projection
	float real_screen_width = 2 * std::tan(view_angle);						// for ortographic projection
	float real_screen_height = real_screen_width * SCR_HEIGHT / SCR_WIDTH;
	
	//for conic projection
	/*
	glm::vec3 top_left_corner = cameraPos + (viewplane_distance * cameraFront) +
		(real_screen_width / 2) * (-cameraRight)
		+ (cameraUp * (real_screen_height / 2));
	*/
	//for ortographic projection
	glm::vec3 top_left_corner = cameraPos +
		(real_screen_width / 2) * (-cameraRight)
		+ (cameraUp * (real_screen_height / 2));
	
	const unsigned int samples_per_ray = 300;
	float front_clip_plane = 0.0f;
	float sample_distance = (viewplane_distance - front_clip_plane) / samples_per_ray; //sample distance
	
	//these variables are to store camera settings during runtime and are initialized with the initial camera settings
	glm::vec3 resetCameraPos = glm::vec3(0.456607f, 0.693644f, -0.55711);		//X0.456607 Y0.693644 Z-0.55711
	glm::vec3 resetCameraFront = glm::vec3(-0.456606f, -0.693643f, 0.557109f);	//X-0.456606 Y-0.693643 Z0.557109
	glm::vec3 resetCameraRight = glm::vec3(-0.19427f, -0.533349f, -0.823285f);	//X-0.19427 Y-0.533349 Z-0.823285
	glm::vec3 resetCameraUp = glm::vec3(0.868199f, -0.484147f, 0.108777f);			//X0.868199 Y-0.484147 Z0.108777
	glm::vec3 resetTop_left_corner = glm::vec3(1.51908f, 0.742847f, 0.374952f);	//X1.51908 Y0.742847 Z0.374952
};


/*
 WARNING: changes the contents of appdata!
 Updates the cordinates of the top left most point of the virtual screen
 */

/*
void updateTopLeftCorner(AppData* appdata) {

	if (appdata->conic) { //for conic projection
		appdata->top_left_corner = appdata->cameraPos + (appdata->viewplane_distance * appdata->cameraFront) +
			(appdata->real_screen_width / 2) * (-appdata->cameraRight)
			+ (appdata->cameraUp * (appdata->real_screen_height / 2));
	}
	else { //for ortographic projection
		appdata->top_left_corner = appdata->cameraPos +
			(appdata->real_screen_width / 2) * (-appdata->cameraRight);
	}	
}*/
