#pragma once

#include <iostream>

#include <vector>


class Convolution{

private:
	std::vector <int>* dimensions; //size of vector represents the number of dimensions, vector elements represent the max number in each dimension 
	std::vector<std::vector<float>>* image;
	std::vector<std::vector<float>>* result;

public:

	/*empty constructor*/
	Convolution();
	/* initialize dimension constructor*/
	Convolution(std::vector<int>* dimension);

	/*Destructor*/
	~Convolution();

	//setters
	bool setDimensions(std::vector<int>* dims) { dimensions = dims; return true; };
	bool setImage(std::vector<std::vector<float>>* img) { image = img; return true; };
	bool setResult(std::vector<std::vector<float>>* img) {result = img; return true; };

	//getters
	std::vector<int>* getDimensions() { return dimensions; };
	std::vector<std::vector<float>>* getImage() { return image; };
	std::vector<std::vector<float>>* getResult() { return result; };
	
	/*Given a convulotion multidimensional array of type T (e.g. glm::vec3, flaot, double, int) 
	*returns - the convoluted result
	*/
	void convolute(std::vector<std::vector<float>>* mvector);

	/*convolution for volumes*/
	void convolute3D(std::vector<std::vector<std::vector<float>>>* mvector);

	/*clamps n according to a dimension d*/
	int OOB(int n, int size);
	void test();
};





