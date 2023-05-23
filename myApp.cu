
#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>
#include <map>
#include <random>

#include "Shader.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#include "utils.h"

//Load nifti2 file
#include "BinaryLoader.h"

#include "Octree.h"
#include "Material.h"
#include "TransferFunction.h"

//testing cuda
#include "kernel.h"

//from https://lencerf.github.io/post/2019-09-21-save-the-opengl-rendering-to-image-file/
void saveImage(char* filepath, GLFWwindow* w);

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window, AppData* appdata);

void printCamera(AppData* appdata);
void resetCameraAttributes(AppData* appdata);

//point to string
std::string pts(glm::vec3 point);

void volumePrepareForPipeline(float* voxels, float* volume_dimensions, NiftiFile* nf);
void prepareVolumeColors(float* voxels, float* volume_dimensions, NiftiFile* nf, TransferFunction* tf);
glm::vec4 sphereTest(float* volume_dimensions, int x, int y, int z, AppData* appdata);
glm::vec4 niftiColorTest(NiftiFile* nf, int x, int y, int z);
glm::vec4 niftiColorTest2(NiftiFile* nf, int x, int y, int z);


//void prepScreenPixColoursForPipeline(Octree* octree, float* pixels, glm::vec3* screen_pixel_ray_dir, glm::vec4* screen_pixel_color, AppData* appdata);
void prepScreenPixColoursForPipeline(Octree* octree, float* pixels, glm::vec3* screen_pixel_ray_dir, glm::vec4* screen_pixel_color, AppData* appdata, TransferFunction* tf, NiftiFile* nf);
void prepScreenPixColoursForPipelineCUDA(Octree* octree, float* pixels, glm::vec3* screen_pixel_ray_dir, glm::vec4* screen_pixel_color, AppData* appdata, TransferFunction* tf, NiftiFile* nf);

void renderLoopSimple(GLFWwindow* window, Shader* ourShader, unsigned int  VAO, glm::mat4 model, glm::mat4 view, glm::mat4 projection, int volume_dimensions_total_size, AppData* appdata);
void renderLoopMerge(GLFWwindow* window,
	Shader* ourShader,
	unsigned int  VAO,
	unsigned int  VBO,
	glm::vec4* host_screen,
	AppData* host_appdata,
	Octree* host_octree,
	TransferFunction* host_transfer_function,
	NiftiFile* host_nf,
	float* pixels
);

void renderLoop(
	GLFWwindow* window,
	NiftiFile* host_nf,
	int volume_dimensions_total_size,

	Shader* test_shader,

	Shader* vrcShader,
	unsigned int vrcVAO,
	unsigned int vrcVBO,
	glm::vec4* host_screen,
	AppData* host_appdata,
	Octree* host_octree,
	TransferFunction* host_transfer_function,
	float* pixels,

	Shader* pointShader,
	unsigned int pointVAO,
	glm::mat4 model,
	glm::mat4 view,
	glm::mat4 projection
);

void renderCUDALoop(GLFWwindow* window, Shader* ourShader, unsigned int VAO, AppData* appdata);

void transformSScreenVec4toFloat(AppData* appdata, float* pixels, glm::vec4* screen_colors);

//void rendering_to_a_texture(GLFWwindow* window, AppData* appdata);

//random functions
void initialize_random_directions(glm::vec3* random_directions, unsigned int number_of_directions);
glm::vec3 getRandomDirection();

//geometry related fucntions
bool insideRadiusCircle(glm::vec3 point, float radius);
bool pointMoved(glm::vec3 current_position, glm::vec3 previous_position);
//
double Henyey_Greenstein_Phaze_Function(glm::vec3 u1, glm::vec3 u2);

int main(int argc, char** argv)
{
	// glfw: initialize and configure
	// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	//Application Data
	AppData appdata = AppData(); //change settings in utils.h file
	TransferFunction tf = TransferFunction();

	//TODO - texture visualization
	// https://www.3dgep.com/opengl-interoperability-with-cuda/
	//an implementation using textures to display the results of CUDA kernels
	
	//https://learnopengl.com/Getting-started/Textures
	
	// glfw window creation
	// --------------------
	GLFWwindow* window = glfwCreateWindow(appdata.SCR_WIDTH, appdata.SCR_HEIGHT, "Nifti 2 Volume Viewer", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}
	glfwSetWindowPos(window, 100, 100);

	// Enable blending
	//if (appdata.simple_point)
	{
		glDisable(GL_CULL_FACE);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_BLEND);
		glEnable(GL_DEPTH_TEST);
	}
	
	//setup random generator
	srand(static_cast <unsigned> (time(0)));//carefull here, only want to initialize once 

	//rendering_to_a_texture(window, &appdata);

	//matrices
	glm::mat4 model = glm::mat4(1.0f);
	model = glm::translate(model, glm::vec3(-0.5f, -0.5f, -0.5f));
	
	glm::mat4 view;
	view = glm::lookAt(appdata.cameraPos,
		glm::vec3(0.0f, 0.0f, 0.0f),
		appdata.cameraUp);

	glm::mat4 projection;
	
	//projection = glm::perspective(glm::radians(45.0f), (float)appdata.SCR_WIDTH / (float)appdata.SCR_HEIGHT, 0.1f, 10.0f);
	//projection = glm::perspective(glm::radians(45.0f), (float)appdata.SCR_WIDTH / (float)appdata.SCR_HEIGHT, 0.0001f, 100.0f);
	projection = glm::ortho<float>(-1.0f,1.0f,-1.0f,1.0f,-1.5f,1.5f);
	
	/*
	glm::vec3 pO(0.0f, 0.0f, 0.0f);
	glm::vec3 pA(1.0f, 1.0f, 0.0f);
	glm::vec3 pB(0.5f,0.5f,0.2f);

	std::cout << "pO: " << pts(pO) << std::endl;
	std::cout << "pA: " << pts(pA) << std::endl;
	std::cout << "pB: " << pts(pB) << std::endl;

	pO = projection * glm::vec4(pO, 1.0f);
	pA = projection * glm::vec4(pA, 1.0f);
	pB = projection * glm::vec4(pB, 1.0f);

	std::cout << "pO: " << pts(pO) << std::endl;
	std::cout << "pA: " << pts(pA) << std::endl;
	std::cout << "pB: " << pts(pB) << std::endl;
	*/ 

	// build and compile our shader program
	// ------------------------------------

	Shader* ourShader = nullptr; //this pointer has the shader that is being used on the display step (DrawArray/DrawElem/etc.)

	Shader* pointShader = new Shader("3.3.point_shader.vs", "3.3.point_shader.fs");	
	Shader* volumeRayCastShader = new Shader("3.3.vrc_shader.vs", "3.3.vrc_shader.fs");
	Shader* textureShader = new Shader("3.3.texture_shader.vs", "3.3.texture_shader.fs");
	Shader* display_cuda_shader = new Shader("3.3.display_CUDA_shader.vs", "3.3.display_CUDA_shader.fs");
	Shader* test_shader = new Shader("3.3.test_shader.vs", "3.3.test_shader.fs");
	
	/*
	switch (appdata.algorithm) {

	case SIMPLE:
		ourShader = simpleShader; // you can name your shader files however you like
		break;
	case VRC:
		ourShader = volumeRayCastShader; //im basically just using opengl to display what i calculated with the CPU
		break;
	case TEXTURESHADER:
		ourShader = textureShader;
		break;
	case DISPLAYCUDASHADER:
		ourShader = display_cuda_shader;

	default:
			std::cout << "Invalid algorithm chosen!" << std::endl;
	}
	*/

	//File Loading 
	NiftiFile* nf = nullptr;

	if (appdata.sphere_test)
		nf = new NiftiFile();
	if (!appdata.sphere_test) {
		//nf = new NiftiFile("avg152T1_LR_nifti2.nii");
		nf = new NiftiFile("MNI152_T1_1mm_nifti2.nii");
	}
		
	//Transfer Function values calculation
	//nf->displayNIFTI2Header();

	/*
	std::string ok = "";
	std::cout << "press ENTER to resume program" << std::endl;
	std::cin >> ok;
	*/
	float* pixels = nullptr;
	//if (!appdata.simple_point)
	pixels = (float*)malloc(sizeof(float) * appdata.SCR_WIDTH * appdata.SCR_HEIGHT * 7);


	float volume_dimensions[3] = {
	nf->header.dim[1],
	nf->header.dim[2],
	nf->header.dim[3]};
	
	//initialize dictionary
	/*
	std::map<int, int> tf_dictionary;

	for (int i = 0; i < (int)nf->header.cal_max; i++) {
		tf_dictionary[i] = 0;
	}

	for (int x = 0; x < volume_dimensions[0]; x++)
		for (int y = 0; y < volume_dimensions[1]; y++)
			for (int z = 0; z < volume_dimensions[2]; z++) {
				int index = x * volume_dimensions[1] * volume_dimensions[2] + y * volume_dimensions[2] + z;
				int key = (int) nf->volume[index];
				tf_dictionary[key] = tf_dictionary[key]++;
			}
			
	std::cout << "------------" << std::endl;
	std::cout << "|DICTIONARY|" << std::endl;
	std::cout << "------------" << std::endl;
	for (std::pair<const int, int> key : tf_dictionary) {
		std::cout << key.first << ": ";
		for (int i = 0; i < key.second / 100; i++) {
			std::cout << "O";
		}
		std::cout << std::endl;
	}
	std::cout << "-----------" << std::endl;
	*/

	float volume_dimensions_total_size = volume_dimensions[0] * volume_dimensions[1] * volume_dimensions[2];
	
	float* voxels = nullptr; // this variable is what has the information ready for the device code
	//if (appdata.simple_point) // disabled "if" to load everything
	{
		voxels = (float*)malloc(sizeof(float) * volume_dimensions_total_size * 7);
		//volumePrepareForPipeline(voxels, volume_dimensions, nf);
		prepareVolumeColors(voxels, volume_dimensions, nf, &tf);
	}

	////set simple_point to false to generate OCTREE
	Octree* octree = nullptr;
	glm::vec3* screen_pixel_ray_dir = nullptr;
	glm::vec4* screen_pixel_color = nullptr;
	glm::vec4* screen = nullptr;

	//if (!appdata.simple_point) // disabled "if" to load everything
	{
		std::cout << "generating octree" << std::endl;
		int time_start = glfwGetTime();
		octree = new Octree(nf); //generates an octree for the nifti file
		int duration = glfwGetTime() - time_start;
		std::cout << "octree creation and atribution duration: " << duration << std::endl;
	
		std::cout << "Allocating Screen Buffers... ";
		screen_pixel_ray_dir = (glm::vec3*)malloc(sizeof(glm::vec3) * appdata.SCR_WIDTH * appdata.SCR_HEIGHT);
		screen_pixel_color = (glm::vec4*)malloc(sizeof(glm::vec4) * appdata.SCR_WIDTH * appdata.SCR_HEIGHT);
		std::cout << "DONE" << std::endl;

		//time_start = glfwGetTime();
		
		//prepScreenPixColoursForPipeline_with_CUDA(octree, pixels, screen_pixel_ray_dir, screen_pixel_color, &appdata);
		//prepScreenPixColoursForPipeline(octree, pixels, screen_pixel_ray_dir, screen_pixel_color, &appdata, &tf, nf);
		/*prepScreenPixColoursForPipelineCUDA(octree, pixels, screen_pixel_ray_dir, screen_pixel_color, &appdata, &tf, nf);


		duration = glfwGetTime() - time_start;
		std::cout << "pixel creation duration: " << duration << std::endl;

		std::cout << "DONE " << std::endl;
		*/

		screen = (glm::vec4*)malloc((long long)appdata.SCR_WIDTH * (long long)appdata.SCR_HEIGHT * sizeof(glm::vec4));
		/*
		float measureTime = glfwGetTime();
		myCUDAspace::renderLoop(screen, &appdata, octree, &tf, nf);
		std::cout << "CUDA loop time: " << glfwGetTime() - measureTime << std::endl;
		
		measureTime = glfwGetTime();
		transformSScreenVec4toFloat(&appdata, pixels, screen);
		std::cout << "Transform time: " << glfwGetTime() - measureTime << std::endl;
		*/
	}
	/*
	unsigned int VBO, VAO;// EBO;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	//glGenBuffers(1, &EBO);

	// bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	//glBufferData(GL_ARRAY_BUFFER, sizeof(screen_corners), screen_corners, GL_STATIC_DRAW);
	
	if(appdata.useCUDA)
		glBufferData(GL_ARRAY_BUFFER, 7 * sizeof(float) * appdata.SCR_WIDTH * appdata.SCR_WIDTH, pixels, GL_STATIC_DRAW);
	else
	{
		if (appdata.simple_point)
			glBufferData(GL_ARRAY_BUFFER, 7 * sizeof(float) * (long)volume_dimensions_total_size, voxels, GL_STATIC_DRAW);
		else
			glBufferData(GL_ARRAY_BUFFER, 7 * sizeof(float) * appdata.SCR_WIDTH * appdata.SCR_WIDTH, pixels, GL_DYNAMIC_DRAW);
	}

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
	//glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0); // no alpha value
	glEnableVertexAttribArray(0);

	// color attribute
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
	//glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float))); //no alpha value
	glEnableVertexAttribArray(1);


	// note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
	// VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
	glBindVertexArray(0);
	*/

	///////////////////////////////////////
	//POINT BUFFERS
	unsigned int pointVBO, pointVAO;
	glGenVertexArrays(1, &pointVAO);
	glGenBuffers(1, &pointVBO);
	{
		// bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
		glBindVertexArray(pointVAO);

		glBindBuffer(GL_ARRAY_BUFFER, pointVBO);

		glBufferData(GL_ARRAY_BUFFER, 7 * sizeof(float) * (long)volume_dimensions_total_size, voxels, GL_STATIC_DRAW);

		// position attribute
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);

		// color attribute
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(1);

		// note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		// You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
		// VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
		glBindVertexArray(0);
	}

	///////////////////////////////////////
	//VRC BUFFERS
	unsigned int vrcVBO, vrcVAO;
	glGenVertexArrays(1, &vrcVAO);
	glGenBuffers(1, &vrcVBO);
	{
		// bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
		glBindVertexArray(vrcVAO);
		glBindBuffer(GL_ARRAY_BUFFER, vrcVBO);

		glBufferData(GL_ARRAY_BUFFER, 7 * sizeof(float) * appdata.SCR_WIDTH * appdata.SCR_WIDTH, pixels, GL_DYNAMIC_DRAW);

		// position attribute
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);

		// color attribute
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(1);

		// note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		// You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
		// VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
		glBindVertexArray(0);
	}

	// render loop
	// -----------
	/*
	switch (algorithm) {
	case POINT:
		renderLoopSimple(window, ourShader, VAO, model, view, projection, volume_dimensions_total_size, &appdata);
		break;
	case VRC:
		renderLoopMerge(window, ourShader, VAO, VBO,
			model, view, projection,
			volume_dimensions_total_size, screen,
			&appdata, octree, &tf, nf, pixels);
		break;
	default:
		std::cout << "WARNING: No algorhtim chosen, invalid or not defined!" << std::endl;
	}*/
	////////////////////////

	std::cout << "real_screen_width" << appdata.real_screen_width << std::endl;

	renderLoop(/*resources fo all algorithms*/
		window,
		nf,
		volume_dimensions_total_size,
		test_shader,
		volumeRayCastShader,
		vrcVAO,
		vrcVBO,
		screen,
		&appdata,
		octree,
		&tf,
		pixels,

		pointShader,
		pointVAO,
		model,
		view,
		projection
	);
	
	
		
	// optional: de-allocate all resources once they've outlived their purpose:
	// ------------------------------------------------------------------------
	//glDeleteVertexArrays(1, &VAO);
	//glDeleteBuffers(1, &VBO);
	//glDeleteProgram(shaderProgram);


	//if (!appdata.simple_point) // disabled "if" to load everything
	{
		free(pixels);//free screen pixels
		free(screen_pixel_color);
		free(screen_pixel_ray_dir);
		delete octree;
		
	}
	//else // disabled "if" to load everything
	{
		free(voxels);
	}
	free(screen);
	//delete ourShader; //delete all shaders
	delete(pointShader);
	delete(volumeRayCastShader);
	delete(textureShader);
	delete(display_cuda_shader);

	delete nf;

	//free(all_screen);
	// glfw: terminate, clearing all previously allocated GLFW resources.
	// ------------------------------------------------------------------
	glfwTerminate();
	return 0;
}

void renderLoopSimple(GLFWwindow* window, Shader* ourShader, unsigned int  VAO, glm::mat4 model, glm::mat4 view, glm::mat4 projection, int volume_dimensions_total_size, AppData* appdata) {

	std::cout << "Rendering" << std::endl;

	while (!glfwWindowShouldClose(window))
	{
		//update delta time
		float currentFrame = glfwGetTime();
		appdata->deltaTime = currentFrame - appdata->lastFrame;
		appdata->lastFrame = currentFrame;

		// input
		// -----
		processInput(window, appdata);

		// render
		// ------
		//clear last frame and zbuffer
		glClearColor(appdata->BACKGROUND_COLOUR.r, appdata->BACKGROUND_COLOUR.g, appdata->BACKGROUND_COLOUR.b, appdata->BACKGROUND_COLOUR.a);
		
		if(appdata->simple_point)
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		else
			glClear(GL_COLOR_BUFFER_BIT);

		view = glm::lookAt(appdata->cameraPos, appdata->cameraPos + appdata->cameraFront, appdata->cameraUp);

		// send the matrices to the shader (this is usually done each frame since transformation matrices tend to change a lot)
		int modelLoc = glGetUniformLocation(ourShader->ID, "model");
		glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

		int viewLoc = glGetUniformLocation(ourShader->ID, "view");
		glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

		int projectionLoc = glGetUniformLocation(ourShader->ID, "projection");
		glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

		//activate the shader
		ourShader->use();

		// render the Volume
		glBindVertexArray(VAO);

		if (appdata->simple_point)
			glDrawArrays(GL_POINTS, 0, volume_dimensions_total_size); //for volume
		else
			glDrawArrays(GL_POINTS, 0, appdata->SCR_WIDTH * appdata->SCR_HEIGHT); //for screen pixels
		//glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0); //to draw from EBO
		glBindVertexArray(0); //no need to unbind everytime

		// glfw: swap buffers look
		//  poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}

void renderLoopMerge(GLFWwindow* window,
	Shader* ourShader,
	unsigned int  VAO,
	unsigned int  VBO,
	glm::vec4* host_screen,
	AppData* host_appdata,
	Octree* host_octree,
	TransferFunction* host_transfer_function,
	NiftiFile* host_nf,
	float* pixels
	) {

	std::cout << "Rendering" << std::endl;

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

	/*cudaStatus = myCUDAspace::allocateDeviceMemory(
		host_screen, host_appdata, host_octree, host_transfer_function, host_nf,
		device_screen, device_application_data, device_octree, device_octree_array,
		device_transfer_function, device_material_intervals, device_nf,
		device_primary_rays, device_sample_colors, device_modelAux);
		*/
	cudaStatus = myCUDAspace::allocateDeviceMemory2(
		host_screen, host_appdata, host_octree, host_transfer_function, host_nf,
		&device_screen, &device_application_data, &device_octree, &device_octree_array,
		&device_transfer_function, &device_material_intervals, &device_nf, &device_nf_volume,
		&device_primary_rays, &device_sample_colors, &device_modelAux);
	{
		if (cudaStatus != cudaSuccess) {
			std::cout << "ERROR in allocateDeviceMemory" << std::endl;
			goto Error;
		}
	}

	unsigned int counter = 0;
	glm::vec3 previous_camera_position = glm::vec3(0.0f, 0.0f, 0.0f);
	float clock =  glfwGetTime();

	while (!glfwWindowShouldClose(window))
	{

		//update delta time
		float currentFrame = glfwGetTime();
		host_appdata->deltaTime = currentFrame - host_appdata->lastFrame;
		host_appdata->lastFrame = currentFrame;
		/*
		std::cout << "frame: " << counter << " Delta: " << host_appdata->deltaTime << std::endl;
		std::cout << "camera cordinates: " 
			<< host_appdata->cameraPos.x << " "
			<< host_appdata->cameraPos.y << " "
			<< host_appdata->cameraPos.z << " "
			<< std::endl;
			*/
		// input
		// -----
		processInput(window, host_appdata);

		// render
		//-------

		//no new calculations required if not moving from previous frame
		if (pointMoved( host_appdata->cameraPos, previous_camera_position)) {
			std::cout << "camera moving" << std::endl;
			previous_camera_position = host_appdata->cameraPos;
			
			/*
			std::cout << "updating previous camera position to " 				
			<< previous_camera_position.x << " " 
			<< previous_camera_position.y << " "
			<< previous_camera_position.z << std::endl;
			*/
			
			cudaStatus = myCUDAspace::updateCameraLocation(host_appdata, device_application_data);

			clock = glfwGetTime();
			cudaStatus = myCUDAspace::updatePrimaryRayDirection(host_appdata, device_application_data, device_primary_rays);
			std::cout << "updatePrimaryRayDirection:" << glfwGetTime() - clock << std::endl;
			{
				if (cudaStatus != cudaSuccess) {
					std::cout << "ERROR in updatePrimaryRayDirection" << std::endl;
					goto Error;
				}
			}

			clock = glfwGetTime();
			cudaStatus = myCUDAspace::getSampleColors(host_appdata, host_nf, device_application_data, device_octree, device_transfer_function, device_primary_rays, device_sample_colors, device_modelAux, screen_size);
			std::cout << "getSampleColors:" << glfwGetTime() - clock << std::endl;
			{
				if (cudaStatus != cudaSuccess) {
					std::cout << "ERROR in getSampleColors" << std::endl;
					goto Error;
				}
			}

			clock = glfwGetTime();
			cudaStatus = myCUDAspace::blendSampleColors(host_appdata, device_application_data, device_screen, device_sample_colors, screen_size);
			std::cout << "blendSampleColors:" << glfwGetTime() - clock << std::endl;
			{
				if (cudaStatus != cudaSuccess) {
					std::cout << "ERROR in blendSampleColors" << std::endl;
					goto Error;
				}
			}

			cudaStatus = cudaMemcpy((void*)host_screen, (void*)device_screen, screen_size * sizeof(glm::vec4), cudaMemcpyDeviceToHost);
			{
				if (cudaStatus != cudaSuccess) {
					std::cout << "ERROR host_screen memcpy device to host" << std::endl;
					goto Error;
				}
			}

			transformSScreenVec4toFloat(host_appdata, pixels, host_screen);
		}

		//display
		//clear last frame
		glClearColor(host_appdata->BACKGROUND_COLOUR.r, host_appdata->BACKGROUND_COLOUR.g, host_appdata->BACKGROUND_COLOUR.b, host_appdata->BACKGROUND_COLOUR.a);
		glClear(GL_COLOR_BUFFER_BIT);

		//activate the shader
		ourShader->use();

		// render the Volume
		glBindVertexArray(VAO);

		//Update VBO
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferSubData(GL_ARRAY_BUFFER, 0, 7 * sizeof(float) * host_appdata->SCR_WIDTH * host_appdata->SCR_WIDTH, pixels);

		glDrawArrays(GL_POINTS, 0, host_appdata->SCR_WIDTH * host_appdata->SCR_HEIGHT); //for screen pixels
		//glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0); //to draw from EBO
		
		glBindVertexArray(0); //no need to unbind everytime

		// glfw: swap buffers look
		//  poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		glfwSwapBuffers(window);
		glfwPollEvents();
		counter++;
	}

Error:
	std::cout << "cudaStatus Description: " << cudaGetErrorString(cudaStatus) << std::endl;

	myCUDAspace::deallocateDeviceMemory(
		device_screen, device_application_data, device_octree, device_octree_array,
		device_transfer_function, device_material_intervals, device_nf, device_nf_volume,
		device_primary_rays, device_sample_colors, device_modelAux);
}

//using GLFW only for the display step WARNING OUTDATED
void renderCUDALoop(GLFWwindow* window, Shader* ourShader, unsigned int VAO, AppData* appdata ) {

	std::cout << "Rendering with CUDA" << std::endl;

	while (!glfwWindowShouldClose(window))
	{
		// input
		// -----
		processInput(window, appdata);

		// render
		// ------
		//clear last frame
		glClearColor(appdata->BACKGROUND_COLOUR.r, appdata->BACKGROUND_COLOUR.g, appdata->BACKGROUND_COLOUR.b, appdata->BACKGROUND_COLOUR.a);
		glClear(GL_COLOR_BUFFER_BIT);
		
		//activate the shader
		ourShader->use();

		// render the Volume
		glBindVertexArray(VAO);

		glDrawArrays(GL_POINTS, 0, appdata->SCR_WIDTH * appdata->SCR_HEIGHT); //for screen pixels
		glBindVertexArray(0); //no need to unbind everytime

		// glfw: swap buffers look
		//  poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

}

/////////////////////////////////
//*resources fo all algorithms*//
/////////////////////////////////
void renderLoop(/*resources fo all algorithms*/
	GLFWwindow* window,
	NiftiFile* host_nf,
	int volume_dimensions_total_size,

	Shader* test_shader,

	Shader* vrcShader,
	
	unsigned int vrcVAO,
	unsigned int vrcVBO,
	glm::vec4* host_screen,
	AppData* host_appdata,
	Octree* host_octree,
	TransferFunction* host_transfer_function,
	float* pixels,

	Shader* pointShader,
	unsigned int pointVAO,
	glm::mat4 model,
	glm::mat4 view,
	glm::mat4 projection
) {

	std::cout << "Rendering" << std::endl;

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

	cudaStatus = myCUDAspace::allocateDeviceMemory2(
		host_screen, host_appdata, host_octree, host_transfer_function, host_nf,
		&device_screen, &device_application_data, &device_octree, &device_octree_array,
		&device_transfer_function, &device_material_intervals, &device_nf, &device_nf_volume,
		&device_primary_rays, &device_sample_colors, &device_modelAux);
	{
		if (cudaStatus != cudaSuccess) {
			std::cout << "ERROR in allocateDeviceMemory" << std::endl;
			goto Error;
		}
	}
	/*//testing cubic interpolation 
	std::cout << "////////////////" << std::endl;
	std::cout << "//SHOWING PATH//" << std::endl;
	std::cout << "////////////////" << std::endl;
	host_octree->searchPointGetIntensityPrinted(0, glm::vec3(0.5f, 0.5f, 0.5f));
	std::cout << std::endl;
	*/

	unsigned int counter = 0;
	glm::vec3 previous_camera_position = glm::vec3(0.0f, 0.0f, 0.0f);
	float clock = glfwGetTime();

	while (!glfwWindowShouldClose(window))
	{

		//update delta time
		float currentFrame = glfwGetTime();
		host_appdata->deltaTime = currentFrame - host_appdata->lastFrame;
		host_appdata->lastFrame = currentFrame;
		
		// input
		// -----
		processInput(window, host_appdata);

		// render
		//-------
		switch (host_appdata->algorithm) {

		case VRC: {
			//no new calculations required if not moving from previous frame
			if (pointMoved(host_appdata->cameraPos, previous_camera_position)) {
				std::cout << "camera moving" << std::endl;
				previous_camera_position = host_appdata->cameraPos;

				cudaStatus = myCUDAspace::updateCameraLocation(host_appdata, device_application_data);

				clock = glfwGetTime();
				cudaStatus = myCUDAspace::updatePrimaryRayDirection(host_appdata, device_application_data, device_primary_rays);
				std::cout << "updatePrimaryRayDirection:" << glfwGetTime() - clock << std::endl;
				{
					if (cudaStatus != cudaSuccess) {
						std::cout << "ERROR in updatePrimaryRayDirection" << std::endl;
						goto Error;
					}
				}

				clock = glfwGetTime();
				cudaStatus = myCUDAspace::getSampleColors(host_appdata, host_nf, device_application_data, device_octree, device_transfer_function, device_primary_rays, device_sample_colors, device_modelAux, screen_size);
				std::cout << "getSampleColors:" << glfwGetTime() - clock << std::endl;
				{
					if (cudaStatus != cudaSuccess) {
						std::cout << "ERROR in getSampleColors" << std::endl;
						goto Error;
					}
				}

				clock = glfwGetTime();
				cudaStatus = myCUDAspace::blendSampleColors(host_appdata, device_application_data, device_screen, device_sample_colors, screen_size);
				std::cout << "blendSampleColors:" << glfwGetTime() - clock << std::endl;
				{
					if (cudaStatus != cudaSuccess) {
						std::cout << "ERROR in blendSampleColors" << std::endl;
						goto Error;
					}
				}

				cudaStatus = cudaMemcpy((void*)host_screen, (void*)device_screen, screen_size * sizeof(glm::vec4), cudaMemcpyDeviceToHost);
				{
					if (cudaStatus != cudaSuccess) {
						std::cout << "ERROR host_screen memcpy device to host" << std::endl;
						goto Error;
					}
				}

				transformSScreenVec4toFloat(host_appdata, pixels, host_screen);
			}

			//display
			//clear last frame
			glDisable(GL_DEPTH_TEST);
			glClearColor(host_appdata->BACKGROUND_COLOUR.r, host_appdata->BACKGROUND_COLOUR.g, host_appdata->BACKGROUND_COLOUR.b, host_appdata->BACKGROUND_COLOUR.a);
			glClear(GL_COLOR_BUFFER_BIT);

			//used to rotate the final image to the correct cordinate system
			glm::mat4 rotate = glm::rotate(glm::mat4(1.0f),glm::radians(180.0f), glm::vec3(0.0f,0.0f,1.0f));

			// send the matrices to the shader
			int rotateLoc = glGetUniformLocation(vrcShader->ID, "rotate");
			glUniformMatrix4fv(rotateLoc, 1, GL_FALSE, glm::value_ptr(rotate));


			//activate the shader
			vrcShader->use();

			// render the Volume
			glBindVertexArray(vrcVAO);

			//Update VBO
			glBindBuffer(GL_ARRAY_BUFFER, vrcVBO);
			glBufferSubData(GL_ARRAY_BUFFER, 0, 7 * sizeof(float) * host_appdata->SCR_WIDTH * host_appdata->SCR_WIDTH, pixels);

			glDrawArrays(GL_POINTS, 0, host_appdata->SCR_WIDTH * host_appdata->SCR_HEIGHT); //for screen pixels
			glBindVertexArray(0); //no need to unbind everytime
		}
			break;

		case POINT: {
			glEnable(GL_DEPTH_TEST);
			glClearColor(host_appdata->BACKGROUND_COLOUR.r, host_appdata->BACKGROUND_COLOUR.g, host_appdata->BACKGROUND_COLOUR.b, host_appdata->BACKGROUND_COLOUR.a);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			
			view = glm::lookAt(host_appdata->cameraPos, glm::vec3(0.0f,0.0f,0.0f), host_appdata->cameraUp);

			// send the matrices to the shader (this is usually done each frame since transformation matrices tend to change a lot)
			int modelLoc = glGetUniformLocation(pointShader->ID, "model");
			glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

			int viewLoc = glGetUniformLocation(pointShader->ID, "view");
			glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

			int projectionLoc = glGetUniformLocation(pointShader->ID, "projection");
			glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

			//activate the shader
			pointShader->use();

			// render the Volume
			glBindVertexArray(pointVAO);
			glDrawArrays(GL_POINTS, 0, volume_dimensions_total_size); //for volume
			
			glBindVertexArray(0); //no need to unbind everytime
		}
			break;

		case TEST: {
			if (pointMoved(host_appdata->cameraPos, previous_camera_position)) {
				std::cout << "camera moving" << std::endl;
				previous_camera_position = host_appdata->cameraPos;

				cudaStatus = myCUDAspace::updateCameraLocation(host_appdata, device_application_data);
		
				std::cout << "getSampleColorsFromNF: ";
				clock = glfwGetTime();
				cudaStatus = myCUDAspace::getSampleColorsFromNF( device_sample_colors, host_appdata, host_nf, device_application_data, device_nf, device_transfer_function, screen_size);
				std::cout << glfwGetTime() - clock << std::endl;
				{
					if (cudaStatus != cudaSuccess) {
						std::cout << "ERROR in getSampleColorsFromNF" << std::endl;
						goto Error;
					}
				}

				std::cout << "blendSampleColors: ";
				clock = glfwGetTime();
				cudaStatus = myCUDAspace::blendSampleColors(host_appdata, device_application_data, device_screen, device_sample_colors, screen_size);
				std::cout << glfwGetTime() - clock << std::endl;
				{
					if (cudaStatus != cudaSuccess) {
						std::cout << "ERROR in blendSampleColors" << std::endl;
						goto Error;
					}
				}

				cudaStatus = cudaMemcpy((void*)host_screen, (void*)device_screen, screen_size * sizeof(glm::vec4), cudaMemcpyDeviceToHost);
				{
					if (cudaStatus != cudaSuccess) {
						std::cout << "ERROR host_screen memcpy device to host" << std::endl;
						goto Error;
					}
				}

				transformSScreenVec4toFloat(host_appdata, pixels, host_screen);
				
			}

			//display
			//clear last frame
			glDisable(GL_DEPTH_TEST);
			glClearColor(host_appdata->BACKGROUND_COLOUR.r, host_appdata->BACKGROUND_COLOUR.g, host_appdata->BACKGROUND_COLOUR.b, host_appdata->BACKGROUND_COLOUR.a);
			glClear(GL_COLOR_BUFFER_BIT);


			//used to rotate the final image to the correct cordinate system
			//glm::mat4 rotate = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 0.0f, 1.0f));
			glm::mat4 rotate = glm::mat4(1.0f);

			// send the matrices to the shader
			int rotateLoc = glGetUniformLocation(vrcShader->ID, "rotate");
			glUniformMatrix4fv(rotateLoc, 1, GL_FALSE, glm::value_ptr(rotate));

			//activate the shader
			vrcShader->use();

			// render the Volume
			glBindVertexArray(vrcVAO);

			//Update VBO
			glBindBuffer(GL_ARRAY_BUFFER, vrcVBO);
			glBufferSubData(GL_ARRAY_BUFFER, 0, 7 * sizeof(float) * host_appdata->SCR_WIDTH * host_appdata->SCR_WIDTH, pixels);

			glDrawArrays(GL_POINTS, 0, host_appdata->SCR_WIDTH * host_appdata->SCR_HEIGHT); //for screen pixels
			glBindVertexArray(0); //no need to unbind everytime
		}
			break;

		default:
			std::cout << "WARNING: NO ALGORITHM CHOSEN" << std::endl;
		}

		// glfw: swap buffers look
		//  poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		glfwSwapBuffers(window);
		glfwPollEvents();
		counter++;
	}

Error:
	std::cout << "cudaStatus Description: " << cudaGetErrorString(cudaStatus) << std::endl;

	myCUDAspace::deallocateDeviceMemory(
		device_screen, device_application_data, device_octree, device_octree_array,
		device_transfer_function, device_material_intervals, device_nf, device_nf_volume,
		device_primary_rays, device_sample_colors, device_modelAux);

}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window, AppData* appdata)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	float cameraSpeed = M_PI * 25.0f * appdata->deltaTime;
	float cameraZoomSpeed = 1.0f * appdata->deltaTime ;
	glm::mat4 rotationMat = glm::mat4(1.0f);
	glm::mat4 translationMat = glm::mat4(1.0f);

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		rotationMat = glm::rotate(rotationMat, glm::radians(cameraSpeed), -appdata->cameraRight);

	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		rotationMat = glm::rotate(rotationMat, glm::radians(cameraSpeed), appdata->cameraRight);

	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		rotationMat = glm::rotate(rotationMat, glm::radians(cameraSpeed), appdata->cameraUp);

	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		rotationMat = glm::rotate(rotationMat, glm::radians(cameraSpeed), -appdata->cameraUp);

	if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
		translationMat = glm::translate(translationMat, appdata->cameraFront * cameraZoomSpeed);

	if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
		translationMat = glm::translate(translationMat, -appdata->cameraFront * cameraZoomSpeed);
	
	appdata->cameraPos = rotationMat * translationMat * glm::vec4(appdata->cameraPos, 1.0f);
	appdata->cameraFront = glm::normalize(glm::vec3(0.0f, 0.0f, 0.0f) - appdata->cameraPos); //keep looking at the center of the world
	appdata->cameraRight = glm::normalize(glm::cross(appdata->cameraUp, appdata->cameraFront));
	appdata->cameraUp = glm::cross(appdata->cameraFront, appdata->cameraRight);
	appdata->top_left_corner = appdata->cameraPos +
		(appdata->real_screen_width / 2) * (-appdata->cameraRight)
		+ (appdata->cameraUp * (appdata->real_screen_height / 2));

	//change algorithm
	if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) {
		
		clock_t current_time = clock();
		
		if (current_time - appdata->last_key_time > appdata->key_delay) {
			std::cout << "CHANGING ALGORITHM" << std::endl;
			appdata->algorithm = (appdata->algorithm == POINT ? TEST : POINT);
			appdata->last_key_time = current_time;
		}
	}

	if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {

		clock_t current_time = clock();

		if (current_time - appdata->last_key_time > appdata->key_delay) {
			std::cout << "CHANGING ALGORITHM" << std::endl;
			appdata->algorithm = POINT;
			appdata->last_key_time = current_time;
		}
	}

	if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {

		clock_t current_time = clock();

		if (current_time - appdata->last_key_time > appdata->key_delay) {
			std::cout << "CHANGING ALGORITHM" << std::endl;
			appdata->algorithm = TEST;
			appdata->last_key_time = current_time;
		}
	}

	if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {

		clock_t current_time = clock();

		if (current_time - appdata->last_key_time > appdata->key_delay) {
			std::cout << "CHANGING ALGORITHM" << std::endl;
			appdata->algorithm = VRC;
			appdata->last_key_time = current_time;
		}
	}

	//save camera reset position
	if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS) {
		
		clock_t current_time = clock();

		if (current_time - appdata->last_key_time > appdata->key_delay) {

			appdata->resetCameraPos = appdata->cameraPos;
			appdata->resetCameraFront = appdata->cameraFront;
			appdata->resetCameraRight = appdata->cameraRight;
			appdata->resetCameraUp = appdata->cameraUp;
			appdata->resetTop_left_corner = appdata->top_left_corner;
			std::cout << "Saved camera reset atributes" << std::endl;

			appdata->last_key_time = current_time;
		}
	}

	//change to reset camera atributes
	if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS) {
		clock_t current_time = clock();

		if (current_time - appdata->last_key_time > appdata->key_delay) {
			resetCameraAttributes(appdata);
			std::cout << "camera atributes reset" << std::endl;
			appdata->last_key_time = current_time;
		}
	}

	//print camera state
	if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) {
		clock_t current_time = clock();
		if (current_time - appdata->last_key_time > appdata->key_delay) {
			printCamera(appdata);
			appdata->last_key_time = current_time;
		}
	}

	//gets lastframe time
	if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS) {
		std::cout << "DELTA TIME: " << appdata->deltaTime << std::endl;
	}

	//save frame into a png
	if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS) {
		std::cout << "Saving Image..." << std::endl;

		std::string path = "image_output/";
		std::string extension = ".png";

		std::string filename = path + "image_" + std::to_string(appdata->SCR_WIDTH) + "x" + std::to_string(appdata->SCR_HEIGHT)+
			"_a" + std::to_string(appdata->algorithm) + "_spr"+ std::to_string(appdata->samples_per_ray) + extension;
		
		std::cout << "filename:" << filename << std::endl;
		char* filenameChar = const_cast<char*>(filename.c_str());
		std::cout << "filenameChar:" << filenameChar << std::endl;


		saveImage(filenameChar, window);
		//delete(filenameChar);

		std::cout << "DONE!" << std::endl;
	}

	//Reseting camera to original position
	if(glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS){
		
		clock_t current_time = clock();
		if (current_time - appdata->last_key_time > appdata->key_delay) {
			std::cout << "Reseting camera to original position" << std::endl;
			appdata->cameraPos = glm::vec4(glm::vec3(0.0f, 0.0f, -1.0f), 1.0f);
			appdata->cameraFront = glm::normalize(glm::vec3(0.0f, 0.0f, 0.0f) - appdata->cameraPos); //keep looking at the center of the world
			appdata->cameraRight = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), appdata->cameraFront));
			appdata->cameraUp = glm::cross(appdata->cameraFront, appdata->cameraRight);
			appdata->top_left_corner = appdata->cameraPos +
				(appdata->real_screen_width / 2) * (-appdata->cameraRight)
				+ (appdata->cameraUp * (appdata->real_screen_height / 2));
			appdata->last_key_time = current_time;
		}
	}
	

}


// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}

//IN USE
void volumePrepareForPipeline( float* voxels, float* volume_dimensions, NiftiFile* nf)
{
	//define the voxel color and position
	for (int x = 0; x < volume_dimensions[0]; x++) {
		for (int y = 0; y < volume_dimensions[1]; y++) {
			for (int z = 0; z < volume_dimensions[2]; z++) {
				int index = (x * volume_dimensions[1] * volume_dimensions[2] + y * volume_dimensions[2] + z) * 7;

				/*x*/voxels[index + 0] = ((float)x) / volume_dimensions[0];
				/*y*/voxels[index + 1] = ((float)y) / volume_dimensions[1];
				/*z*/voxels[index + 2] = ((float)z) / volume_dimensions[2];

				glm::vec4 color = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
				
				color = niftiColorTest(nf, x, y, z);

				/*r*/voxels[index + 3] = color.r;
				/*g*/voxels[index + 4] = color.g;
				/*b*/voxels[index + 5] = color.b;
				/*a*/voxels[index + 6] = color.a;
			}
		}
	}
}

//IN USE
void prepareVolumeColors(float* voxels, float* volume_dimensions, NiftiFile* nf, TransferFunction* tf)
{
	//get longest dimension
	int longest_dimension = 0;
	for (int i = 0; i < nf->header.dim[0]; i++) {
		if (nf->header.dim[i+1] > longest_dimension)
			longest_dimension = nf->header.dim[i + 1];
	}

	//define the voxel color and position
	for (int x = 0; x < volume_dimensions[0]; x++) {
		for (int y = 0; y < volume_dimensions[1]; y++) {
			for (int z = 0; z < volume_dimensions[2]; z++) {



				int volumeIndex = nf->transformVector3Position(glm::vec3(x, y, z));
				//int index = (x * volume_dimensions[1] * volume_dimensions[2] + y * volume_dimensions[2] + z) * 7;
				int index = volumeIndex * 7;



				/*x*/voxels[index + 0] = (((float)x + longest_dimension / 2.0f) - (volume_dimensions[0] / 2.0f)) / longest_dimension;
				/*x*/voxels[index + 1] = (((float)y + longest_dimension / 2.0f) - (volume_dimensions[1] / 2.0f)) / longest_dimension;
				/*x*/voxels[index + 2] = (((float)z + longest_dimension / 2.0f) - (volume_dimensions[2] / 2.0f)) / longest_dimension;

				float intensity = nf->volume[volumeIndex];
				glm::vec4 color = tf->getMaterial(intensity/nf->header.cal_max)->color;

				/*r*/voxels[index + 3] = color.r;
				/*g*/voxels[index + 4] = color.g;
				/*b*/voxels[index + 5] = color.b;
				/*a*/voxels[index + 6] = color.a;
			}
		}
	}
}

//IN USE
glm::vec4 niftiColorTest(NiftiFile* nf, int x, int y, int z) {
	glm::vec4 color = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);

	int index = nf->transformVector3Position(glm::vec3(x, y, z));
	float intensity = nf->volume[index] / nf->header.cal_max;

	if (intensity >= 0.1f && intensity < 0.3f)
		color = glm::vec4(0.1f, 0.1f, 0.1f, 1.0f);
	if (intensity >= 0.3f && intensity < 0.4f)
		color = glm::vec4(0.0f, 0.0f, 0.8f, 1.0f);
	if (intensity >= 0.4f && intensity < 0.5f)
		color = glm::vec4(0.8f, 0.8f, 0.4f, 1.0f);
	if (intensity >= 0.5f && intensity < 0.6f)
		color = glm::vec4(0.1f, 0.5f, 0.5f, 1.0f);
	if (intensity >= 0.6f && intensity < 0.7f)
		color = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);
	if (intensity >= 0.7f && intensity <= 1.0f)
		color = glm::vec4(0.9f, 0.5f, 0.5f, 1.0f);
	return color;
}

//IN USE
glm::vec4 niftiColorTest2(NiftiFile* nf, int x, int y, int z) {
	glm::vec4 color = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
	int index = nf->transformVector3Position(glm::vec3(x, y, z));
	float intensity = nf->volume[index] / nf->header.cal_max;
	
	if (intensity > 0.0f) {
		color = glm::vec4(0.0f, intensity * 255.0f, 0.0f, 1.0f);
	}
	if (x == 0) {
		color = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);
	}
	if (y == 0) {
		color = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f);
	}
	if (z == 0) {
		color = glm::vec4(0.0f, 1.0f, 1.0f, 1.0f);
	}

	return color;
}

//WARNING OUTDATED
glm::vec4 sphereTest(float* volume_dimensions, int x, int y, int z, AppData* appdata) {
	glm::vec4 color = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
	glm::vec3 center = glm::vec3(volume_dimensions[0] / 2, volume_dimensions[1] / 2, volume_dimensions[2] / 2);
	float radius = volume_dimensions[0] / 2;
	float sphere_test = pow(x - center.x, 2) + pow(y - center.y, 2) + pow(z - center.z, 2);
	if (sphere_test <= pow(radius, 2)) {

		if (x > volume_dimensions[0] / 2)
			if (y > volume_dimensions[1] / 2)
				if (z > volume_dimensions[2] / 2)
					color = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
				else
					color = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
			else
				if (z > volume_dimensions[2] / 2)
					color = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
				else
					color = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f);
		else
			if (y > volume_dimensions[1] / 2)
				if (z > volume_dimensions[2] / 2)
					color = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);
				else
					color = glm::vec4(0.0f, 1.0f, 1.0f, 1.0f);
			else
				if (z > volume_dimensions[2] / 2)
					color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
				else
					color = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
	}

	if (z == 0 || z == volume_dimensions[2] - 1)
		color = glm::vec4(1.0f - appdata->BACKGROUND_COLOUR.r, 1.0f - appdata->BACKGROUND_COLOUR.g, 1.0f - appdata->BACKGROUND_COLOUR.b, 1.0f);

	return color;
}

//WARNIG OUTDATED
void prepScreenPixColoursForPipeline(Octree* octree, float* pixels, glm::vec3* screen_pixel_ray_dir, glm::vec4* screen_pixel_color, AppData* appdata, TransferFunction* tf, NiftiFile* nf) {

	std::cout << "preparing Screen Pixel Color For Pipeline" << std::endl;

	//matrices
	glm::mat4 modelAux = glm::mat4(1.0f);
	modelAux = glm::translate(modelAux, glm::vec3(0.5f, 0.5f, 0.5f));
	modelAux = glm::rotate(modelAux, glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	modelAux = glm::rotate(modelAux, glm::radians(90.0f), glm::vec3(-1.0f, 0.0f, 0.0f));

	glm::mat4 viewAux;
	viewAux = glm::lookAt(
		glm::vec3(0.0f, 0.0f, 0.0f),
		appdata->cameraPos,
		appdata->cameraUp);

	//setup screen_pixel_ray_dir	
	std::cout << "Setting up ray directions... ";

	for (int x = 0; x < appdata->SCR_WIDTH; x++) {
		for (int y = 0; y < appdata->SCR_HEIGHT; y++) {
			int index = x * appdata->SCR_HEIGHT + y;
			if (appdata->conic) {
				//conic projection
				screen_pixel_ray_dir[index] = normalize(appdata->top_left_corner +
					(x * appdata->real_screen_width / appdata->SCR_WIDTH) * (appdata->cameraRight) +
					(y * appdata->real_screen_height / appdata->SCR_HEIGHT) * (-appdata->cameraUp)
					- appdata->cameraPos);
			}
			else {
				//orthographic projection
				screen_pixel_ray_dir[index] = appdata->cameraFront;
			}
		}
	}
	std::cout << "DONE" << std::endl;

	std::cout << "Calculating pixel colors...";

	for (int x = 0; x < appdata->SCR_WIDTH; x++) {
		for (int y = 0; y < appdata->SCR_HEIGHT; y++) {
			glm::vec4 fragmentColor = appdata->BACKGROUND_COLOUR;

			//from back to front
			for (int i = appdata->samples_per_ray; i > 0; i--) {
				glm::vec3 auxPos;
				if (appdata->conic) {//for conic projection
					auxPos = appdata->cameraPos + i * appdata->sample_distance * screen_pixel_ray_dir[x * appdata->SCR_HEIGHT + y]; //this is the position of the sample in world cordinates
				}
				else {// for ortographic projection
					auxPos = appdata->top_left_corner +
						x * appdata->real_screen_width / appdata->SCR_WIDTH * appdata->cameraRight +
						y * appdata->real_screen_height / appdata->SCR_HEIGHT * (-appdata->cameraUp) +
						i * appdata->sample_distance * screen_pixel_ray_dir[x * appdata->SCR_HEIGHT + y]; //this is the position of the sample in world cordinates
				}

				float intensity = octree->getIntensity(modelAux * glm::vec4(auxPos, 1.0f));

		

				//blend colors
				glm::vec4 blend_color = glm::vec4(0.0f);
				float normalized_intensity = intensity / nf->header.cal_max;
				blend_color = tf->getMaterial(normalized_intensity)->color;

				fragmentColor = glm::vec4(
					fragmentColor.r * (1 - blend_color.a) + blend_color.r * blend_color.a,
					fragmentColor.g * (1 - blend_color.a) + blend_color.g * blend_color.a,
					fragmentColor.b * (1 - blend_color.a) + blend_color.b * blend_color.a,
					1.0f);
			}

			screen_pixel_color[x * appdata->SCR_HEIGHT + y] = fragmentColor;

		}

	}
	std::cout << "DONE" << std::endl;


	std::cout << "Generating pixels... ";
	for (int x = 0; x < appdata->SCR_WIDTH; x++) {
		for (int y = 0; y < appdata->SCR_HEIGHT; y++) {
			int index = (x * appdata->SCR_HEIGHT + y) * 7;
			/*x*/pixels[index + 0] = 2 * (((float)x) / appdata->SCR_WIDTH) - 1;
			/*y*/pixels[index + 1] = 2 * (((float)y) / appdata->SCR_HEIGHT) - 1;
			/*z*/pixels[index + 2] = 0.0f;
			/*r*/pixels[index + 3] = screen_pixel_color[x * appdata->SCR_HEIGHT + y].r;
			/*g*/pixels[index + 4] = screen_pixel_color[x * appdata->SCR_HEIGHT + y].g;
			/*b*/pixels[index + 5] = screen_pixel_color[x * appdata->SCR_HEIGHT + y].b;
			/*a*/pixels[index + 6] = screen_pixel_color[x * appdata->SCR_HEIGHT + y].a;
		}
	}
	std::cout << "DONE" << std::endl;
}

//WARNING OUTDATED
void prepScreenPixColoursForPipelineCUDA(Octree* octree, float* pixels, glm::vec3* screen_pixel_ray_dir, glm::vec4* screen_pixel_color, AppData* appdata, TransferFunction* tf, NiftiFile* nf) {

	std::cout << "preparing Screen Pixel Color For Pipeline" << std::endl;

	//matrices
	glm::mat4 modelAux = glm::mat4(1.0f);
	modelAux = glm::translate(modelAux, glm::vec3(0.5f, 0.5f, 0.5f));
	modelAux = glm::rotate(modelAux, glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	modelAux = glm::rotate(modelAux, glm::radians(90.0f), glm::vec3(-1.0f, 0.0f, 0.0f));

	glm::mat4 viewAux;
	viewAux = glm::lookAt(
		glm::vec3(0.0f, 0.0f, 0.0f),
		appdata->cameraPos,
		appdata->cameraUp);

	//setup screen_pixel_ray_dir
	myCUDAspace::setupRayDirectionCUDA(screen_pixel_ray_dir, appdata);	
	


	std::cout << "Calculating pixel colors...";
	for (int x = 0; x < appdata->SCR_WIDTH; x++) {
		for (int y = 0; y < appdata->SCR_HEIGHT; y++) {
			glm::vec4 fragmentColor = appdata->BACKGROUND_COLOUR;

			//from back to front
			for (int i = appdata->samples_per_ray; i > 0; i--) {
				glm::vec3 auxPos;
				if (appdata->conic) {//for conic projection
					auxPos = appdata->cameraPos + i * appdata->sample_distance * screen_pixel_ray_dir[x * appdata->SCR_HEIGHT + y]; //this is the position of the sample in world cordinates
				}
				else {// for ortographic projection
					auxPos = appdata->top_left_corner +
						x * appdata->real_screen_width / appdata->SCR_WIDTH * appdata->cameraRight +
						y * appdata->real_screen_height / appdata->SCR_HEIGHT * (-appdata->cameraUp) +
						i * appdata->sample_distance * screen_pixel_ray_dir[x * appdata->SCR_HEIGHT + y]; //this is the position of the sample in world cordinates
				}

				float intensity = octree->getIntensity(modelAux * glm::vec4(auxPos, 1.0f));


				//blend colors
				glm::vec4 blend_color = glm::vec4(0.0f);
				float normalized_intensity = intensity / nf->header.cal_max;
				blend_color = tf->getMaterial(normalized_intensity)->color;

				fragmentColor = glm::vec4(
					fragmentColor.r * (1 - blend_color.a) + blend_color.r * blend_color.a,
					fragmentColor.g * (1 - blend_color.a) + blend_color.g * blend_color.a,
					fragmentColor.b * (1 - blend_color.a) + blend_color.b * blend_color.a,
					1.0f);
			}

			screen_pixel_color[x * appdata->SCR_HEIGHT + y] = fragmentColor;

		}

	}
	std::cout << "DONE" << std::endl;


	std::cout << "Generating pixels... ";
	for (int x = 0; x < appdata->SCR_WIDTH; x++) {
		for (int y = 0; y < appdata->SCR_HEIGHT; y++) {
			int index = (x * appdata->SCR_HEIGHT + y) * 7;
			/*x*/pixels[index + 0] = 2 * (((float)x) / appdata->SCR_WIDTH) - 1;
			/*y*/pixels[index + 1] = 2 * (((float)y) / appdata->SCR_HEIGHT) - 1;
			/*z*/pixels[index + 2] = 0.0f;
			/*r*/pixels[index + 3] = screen_pixel_color[x * appdata->SCR_HEIGHT + y].r;
			/*g*/pixels[index + 4] = screen_pixel_color[x * appdata->SCR_HEIGHT + y].g;
			/*b*/pixels[index + 5] = screen_pixel_color[x * appdata->SCR_HEIGHT + y].b;
			/*a*/pixels[index + 6] = screen_pixel_color[x * appdata->SCR_HEIGHT + y].a;
		}
	}
	std::cout << "DONE" << std::endl;
}

//WARNING OUTDATED
void prepPixelsCUDA(Octree* octree, float* pixels, glm::vec3* screen_pixel_ray_dir, glm::vec4* screen_pixel_color, AppData* appdata, TransferFunction* tf, NiftiFile* nf) {

	std::cout << "preparing Screen Pixel Color For Pipeline" << std::endl;

	//matrices
	glm::mat4 modelAux = glm::mat4(1.0f);
	modelAux = glm::translate(modelAux, glm::vec3(0.5f, 0.5f, 0.5f));
	modelAux = glm::rotate(modelAux, glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	modelAux = glm::rotate(modelAux, glm::radians(90.0f), glm::vec3(-1.0f, 0.0f, 0.0f));

	glm::mat4 viewAux;
	viewAux = glm::lookAt(
		glm::vec3(0.0f, 0.0f, 0.0f),
		appdata->cameraPos,
		appdata->cameraUp);

	//setup screen_pixel_ray_dir	
	myCUDAspace::setupRayDirectionCUDA(screen_pixel_ray_dir, appdata);
	
	
	std::cout << "Calculating pixel colors...";

		//direction for scattering
	glm::vec3 random_direction = getRandomDirection();

	for (int x = 0; x < appdata->SCR_WIDTH; x++) {
		for (int y = 0; y < appdata->SCR_HEIGHT; y++) {
			glm::vec4 fragmentColor = appdata->BACKGROUND_COLOUR;

			//from back to front
			for (int i = appdata->samples_per_ray; i > 0; i--) {
				glm::vec3 auxPos;
				if (appdata->conic) {//for conic projection
					auxPos = appdata->cameraPos + i * appdata->sample_distance * screen_pixel_ray_dir[x * appdata->SCR_HEIGHT + y]; //this is the position of the sample in world cordinates
				}
				else {// for ortographic projection
					auxPos = appdata->top_left_corner +
							x * appdata->real_screen_width / appdata->SCR_WIDTH * appdata->cameraRight +
							y * appdata->real_screen_height / appdata->SCR_HEIGHT * (-appdata->cameraUp) +
							i * appdata->sample_distance * screen_pixel_ray_dir[x * appdata->SCR_HEIGHT + y]; //this is the position of the sample in world cordinates
				}

				float intensity = octree->getIntensity(modelAux * glm::vec4(auxPos, 1.0f));
					
				//blend colors
				glm::vec4 blend_color = glm::vec4(0.0f);
				float normalized_intensity = intensity / nf->header.cal_max;
					
				Material::Material* material = tf->getMaterial(normalized_intensity);
				//Henyey_Greenstein_Phaze_Function(screen_pixel_ray_dir[x * appdata->SCR_HEIGHT + y], random_direction);

				blend_color = material->color;
					 
				fragmentColor = glm::vec4(
						fragmentColor.r * (1 - blend_color.a) + blend_color.r * blend_color.a,
						fragmentColor.g * (1 - blend_color.a) + blend_color.g * blend_color.a,
						fragmentColor.b * (1 - blend_color.a) + blend_color.b * blend_color.a,
						1.0f);
			}

			screen_pixel_color[x * appdata->SCR_HEIGHT + y] = fragmentColor;

		}
	}

	std::cout << "DONE" << std::endl;

	std::cout << "Generating pixels... ";
	for (int x = 0; x < appdata->SCR_WIDTH; x++) {
		for (int y = 0; y < appdata->SCR_HEIGHT; y++) {
			int index = (x * appdata->SCR_HEIGHT + y) * 7;
			/*x*/pixels[index + 0] = 2 * (((float)x) / appdata->SCR_WIDTH) - 1;
			/*y*/pixels[index + 1] = 2 * (((float)y) / appdata->SCR_HEIGHT) - 1;
			/*z*/pixels[index + 2] = 0.0f;
			/*r*/pixels[index + 3] = screen_pixel_color[x * appdata->SCR_HEIGHT + y].r;
			/*g*/pixels[index + 4] = screen_pixel_color[x * appdata->SCR_HEIGHT + y].g;
			/*b*/pixels[index + 5] = screen_pixel_color[x * appdata->SCR_HEIGHT + y].b;
			/*a*/pixels[index + 6] = screen_pixel_color[x * appdata->SCR_HEIGHT + y].a;
		}
	}
	std::cout << "DONE" << std::endl;
}

//IN USE
void transformSScreenVec4toFloat(AppData* appdata, float * pixels, glm::vec4* screen_colors) {
	for (int x = 0; x < appdata->SCR_WIDTH; x++) {
		for (int y = 0; y < appdata->SCR_HEIGHT; y++) {
			int screen_index = x * appdata->SCR_HEIGHT + y;
			int pixelindex = (x * appdata->SCR_HEIGHT + y) * 7;
			/*x*/pixels[pixelindex + 0] = 2.0f * (((float)x) / appdata->SCR_WIDTH) - 1.0f;
			/*y*/pixels[pixelindex + 1] = 2.0f * (((float)y) / appdata->SCR_HEIGHT) - 1.0f;
			/*
			if (x == appdata->SCR_WIDTH-40 && y == appdata->SCR_HEIGHT - 40) {
				std::cout << "(0:0): ("
					<< pixels[pixelindex + 0] << ":"
					<< pixels[pixelindex + 1] << ")" << std::endl;

				std::cout << "screen index color: "
					<< screen_colors[screen_index].r << " "
					<< screen_colors[screen_index].g << " "
					<< screen_colors[screen_index].b << " "
					<< screen_colors[screen_index].a << " "
					<< std::endl;
			}*/
			/*z*/pixels[pixelindex + 2] = 0.0f;
			/*r*/pixels[pixelindex + 3] = screen_colors[screen_index].r;
			/*g*/pixels[pixelindex + 4] = screen_colors[screen_index].g;
			/*b*/pixels[pixelindex + 5] = screen_colors[screen_index].b;
			/*a*/pixels[pixelindex + 6] = screen_colors[screen_index].a;
		}
	}
}

/*
Use malloc before calling this function
*/
void initialize_random_directions( glm::vec3* random_directions, unsigned int number_of_directions) {
	for (int i = 0; i < number_of_directions; i++) {
		random_directions[i] = getRandomDirection();
	}
}

bool insideRadiusCircle(glm::vec3 point, float radius) {
	return

		point.x * point.x +

		point.y * point.y +

		point.z * point.z <=

		radius * radius;
}

glm::vec3 getRandomDirection() {
	glm::vec3 ray_direction;
	do {
		ray_direction = glm::vec3((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
	} while (!insideRadiusCircle(ray_direction, 1.0f)); // repeats until ray direction is inside unit cricle
	ray_direction = glm::normalize(ray_direction); // make ray_direction a unit vector

	return ray_direction;
}

double Henyey_Greenstein_Phaze_Function( glm::vec3 u1, glm::vec3 u2) {
	double result = 0.0;
	double angle =  acos(glm::dot(u1,u2));
	double g = 0.0;
	result = (1 / (4 * M_PI))* ((1 - g * g) / (1 + g * g - 2 * g * cos(angle)));
	//(1 / 2) * ((1 - g * g) / (1 + g * g - 2 * g * cos(angle)));
	return result;
}


//USE FOR POSTPROCESSING
void rendering_to_a_texture(GLFWwindow* window, AppData* appdata ) {

	//
	glm::vec4* host_screen = nullptr;
	AppData* host_appdata = nullptr;
	Octree* host_octree = nullptr;
	TransferFunction* host_transfer_function = nullptr;
	NiftiFile* host_nf = nullptr;

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
	//


	Shader* quadShader = new Shader("3.3.texture_shader.vs", "3.3.texture_shader.fs");

	float quad_vertices[] = {
		// positions          // colors           // texture coords
		 1.0f,  1.0f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f,   // top right
		 1.0f, -1.0f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f,   // bottom right
		-1.0f, -1.0f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f,   // bottom left
		-1.0f,  1.0f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f    // top left 
	};

	unsigned int quad_indices[] = {  // note that we start from 0!
	0, 1, 3,   // first triangle
	1, 2, 3    // second triangle
	};

	unsigned int quadVAO;
	unsigned int quadVBO;
	unsigned int quadEBO;

	glGenVertexArrays(1, &quadVAO);
	glGenBuffers(1, &quadVBO);
	glGenBuffers(1, &quadEBO);

	glBindVertexArray(quadVAO);
	glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quadEBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quad_indices), quad_indices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
	//glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0); // no alpha value
	glEnableVertexAttribArray(0);

	// color attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);

	// texture cords attribute
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	// You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
	// VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
	glBindVertexArray(0);


	unsigned int off_screen_framebuffer;
	glGenFramebuffers(1, &off_screen_framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, off_screen_framebuffer);

	// generate texture
	unsigned int textureColorbuffer;
	glGenTextures(1, &textureColorbuffer);
	glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, appdata->SCR_WIDTH, appdata->SCR_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);

	// attach it to currently bound framebuffer object
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorbuffer, 0);

	unsigned int rbo;
	glGenRenderbuffers(1, &rbo);
	glBindRenderbuffer(GL_RENDERBUFFER, rbo);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, appdata->SCR_WIDTH, appdata->SCR_HEIGHT);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);


	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
	else	
		std::cout << "Framebuffer setup complete" << std::endl;
	
	//unbind framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	//setup cuda "environment"
	myCUDAspace::allocateDeviceMemory2(
		host_screen, host_appdata, host_octree, host_transfer_function, host_nf,
		&device_screen, &device_application_data, &device_octree, &device_octree_array,
		&device_transfer_function, &device_material_intervals, &device_nf, &device_nf_volume,
		&device_primary_rays, &device_sample_colors, &device_modelAux);

	while (!glfwWindowShouldClose(window)) {
		// input
		// -----
		processInput(window, appdata);

		// first pass
		glBindFramebuffer(GL_FRAMEBUFFER, off_screen_framebuffer);
		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // we're not using the stencil buffer now
		glEnable(GL_DEPTH_TEST);


		//draw to the off screen framebuffer
		//DrawScene();
		//HERE DO THIS ONE FIRST





		// second pass
		glBindFramebuffer(GL_FRAMEBUFFER, 0); // back to default
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);



		// render quad to screen
		// ------
		//clear last frame and zbuffer
		glClearColor(appdata->BACKGROUND_COLOUR.r, appdata->BACKGROUND_COLOUR.g, appdata->BACKGROUND_COLOUR.b, appdata->BACKGROUND_COLOUR.a);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glClear(GL_COLOR_BUFFER_BIT);

		quadShader->use();
		glBindVertexArray(quadVAO);
		//glDisable(GL_DEPTH_TEST);
		glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		//glBindVertexArray(0); //no need to unbind everytime

		// glfw: swap buffers look
		//  poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		glfwSwapBuffers(window);
		glfwPollEvents();

	}


	//release CUDA "environment"
	myCUDAspace::deallocateDeviceMemory(device_screen, device_application_data, device_octree, device_octree_array,
			device_transfer_function, device_material_intervals, device_nf, device_nf_volume,
			device_primary_rays, device_sample_colors, device_modelAux);


}


bool pointMoved(glm::vec3 current_position, glm::vec3 previous_position) {
	glm::vec3 aux = current_position - previous_position;
	//std::cout << "AUX POINT: " << aux.x << " " << aux.y << " " << aux.z << std::endl;
	return aux.x != 0.0f || aux.y != 0.0f || aux.z != 0.0f;
}


void resetCameraAttributes(AppData* appdata) {
	appdata->cameraPos = appdata->resetCameraPos;
	appdata->cameraFront = appdata->resetCameraFront;
	appdata->cameraRight = appdata->resetCameraRight;
	appdata->cameraUp = appdata->resetCameraUp;
	appdata->top_left_corner = appdata->resetTop_left_corner;
}

void printCamera(AppData* appdata) {
	std::cout << "Camera Position: X" << appdata->cameraPos.x <<
		" Y" << appdata->cameraPos.y <<
		" Z" << appdata->cameraPos.z << std::endl;

	std::cout << "Camera Front: X" << appdata->cameraFront.x <<
		" Y" << appdata->cameraFront.y <<
		" Z" << appdata->cameraFront.z << std::endl;

	std::cout << "Camera Right: X" << appdata->cameraRight.x <<
		" Y" << appdata->cameraRight.y <<
		" Z" << appdata->cameraRight.z << std::endl;

	std::cout << "Camera Up: X" << appdata->cameraUp.x <<
		" Y" << appdata->cameraUp.y <<
		" Z" << appdata->cameraUp.z << std::endl;

	std::cout << "Camera top left corner: X" << appdata->top_left_corner.x <<
		" Y" << appdata->top_left_corner.y <<
		" Z" << appdata->top_left_corner.z << std::endl;
}


void saveImage(char* filepath, GLFWwindow* w)
{
	int width, height;
	glfwGetFramebufferSize(w, &width, &height);
	GLsizei nrChannels = 3;
	GLsizei stride = nrChannels * width;
	stride += (stride % 4) ? (4 - stride % 4) : 0;
	GLsizei bufferSize = stride * height;
	std::vector<char> buffer(bufferSize);
	glPixelStorei(GL_PACK_ALIGNMENT, 4);
	glReadBuffer(GL_FRONT);
	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer.data());
	stbi_flip_vertically_on_write(true);
	stbi_write_png(filepath, width, height, nrChannels, buffer.data(), stride);
}

//testing module operator "%"
void foo() {

	int var = 0;
	int module = 10;
	int iterations = 20;
	for (int i = 0; i < iterations; i++) {
		var = (var + 1) % module;
		std::cout << "var: " << var << std::endl;
	}
		
}

std::string pts(glm::vec3 point) {
	return std::to_string(point.x) + " " + std::to_string(point.y) + " " + std::to_string(point.z);
}