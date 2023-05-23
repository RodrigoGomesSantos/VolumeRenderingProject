#include "OldConvolution.h"

/////////////////////////////////////////////////////////
//Declarations
void print_image(std::vector<std::vector<float>>* mvector);

//////////////////////////////////////////////////////////

Convolution::Convolution()
{
	std::vector<std::vector<float>> image =
	{
		{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}, //1
		{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}, //2
		{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}, //3
		{0.0f,0.0f,0.0f,0.0f,1.0f,1.0f,0.0f,0.0f,0.0f,0.0f}, //4
		{0.0f,0.0f,0.0f,1.0f,1.0f,1.0f,1.0f,0.0f,0.0f,0.0f}, //5
		{0.0f,0.0f,0.0f,1.0f,1.0f,1.0f,1.0f,0.0f,0.0f,0.0f}, //6
		{0.0f,0.0f,0.0f,0.0f,1.0f,1.0f,0.0f,0.0f,0.0f,0.0f}, //7
		{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}, //8
		{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}, //9
		{0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}  //10
	};

	std::vector<int> imagesize = { 10,10 };
	
	std::vector<std::vector<float>> instanciate_result = {};
	setDimensions(&imagesize);
	setImage(&image);
	setResult(&instanciate_result);
	test();
}

Convolution::Convolution(std::vector<int>* dimension)
{
	setDimensions(dimension);
}


Convolution::~Convolution()
{
	std::cout << "Convolution destroyed" << std::endl;
}

void Convolution::convolute(std::vector<std::vector<float>>* mvector) {

	//the only information thats missing is the number of dimensions
	//e.g. a 2D image

	//these values are always numbers (e.g. float, double, int) and represent the amount each cell contributes to the final result
	float m[3][3] = {
		{0.0f, 0.1f, 0.0f},
		{0.1f, 0.6f, 0.1f},
		{0.0f, 0.1f, 0.0f} };

	std::vector<std::vector<float>>* result = getResult();

	for (int i = 0; i < 10; i++) {

		result[0].push_back({}); // std::vector<float>
		
		for (int j = 0; j < 10; j++) {

			float pixel_result = 0;

			for (int mx = 0; mx < 3; mx++)
				for (int my = 0; my < 3; my++)
					pixel_result += mvector[0][OOB(i - 1 + mx, 0)][OOB(j - 1 + my, 1)] * m[mx][my];
				/*
				mvector[0][OOB(i - 1, 0)][OOB(j - 1, 1)] * m[0][0] + 
				mvector[0][OOB(i - 1, 0)][OOB(j, 1)] * m[0][1] + 
				mvector[0][OOB(i - 1, 0)][OOB(j + 1, 1)] * m[0][2] +

				mvector[0][OOB(i, 0)][OOB(j - 1, 1)] * m[1][0] + 
				mvector[0][OOB(i, 0)][OOB(j, 1)] * m[1][1] + 
				mvector[0][OOB(i, 0)][OOB(j + 1, 1)] * m[1][2] +

				mvector[0][OOB(i + 1, 0)][OOB(j - 1, 1)] * m[2][0] + 
				mvector[0][OOB(i + 1, 0)][OOB(j, 1)] * m[2][1] + 
				mvector[0][OOB(i + 1, 0)][OOB(j + 1, 1)] * m[2][2];
				*/

			result[0][i].push_back(pixel_result);
		}
	}
}

/*
void Convolution::convolute3D(std::vector<std::vector<std::vector<float>>>* mvector, std::vector<std::vector<std::vector<float>>>* m) {

	//the only information thats missing is the number of dimensions
	//e.g. a 2D image

	float m[3][3][3] = {
		{	{0.0f, 0.0f, 0.0f},
			{0.0f, 0.1f, 0.0f},
			{0.0f, 0.0f, 0.0f}	},

		{	{0.0f, 0.1f, 0.0f},
			{0.1f, 1.0f, 0.1f},
			{0.0f, 0.1f, 0.0f}	},

		{	{0.0f, 0.0f, 0.0f},
			{0.0f, 0.1f, 0.0f},
			{0.0f, 0.0f, 0.0f}	}
	};


	std::vector<std::vector<std::vector<float>>> result_ = {};
	std::vector<std::vector<std::vector<float>>>* result = &result_; //getResult();


	for (int i = 0; i < 10; i++) {
		result[0].push_back({}); // std::vector<std::vector<float>>
		for (int j = 0; j < 10; j++) {
			result[0][i].push_back({}); // std::vector<float>
			for (int k = 0; k < 10; k++) {

				float pixel_result = 0;
				for (int mx = 0; mx < 3; mx++)
					for (int my = 0; my < 3; my++)
						for (int mz = 0; mz < 3; mz++)
							pixel_result += mvector[0][OOB(i - 1 + mx, 0)][OOB(j - 1 + my, 1)][OOB(k - 1 + mz, 2)] * m[mx][my][mz];

				result[0][i][j].push_back(pixel_result);
			}
		}
	}
}

*/


int Convolution::OOB(int n, int d) {
	int res = std::max(n % dimensions[0][d], 0 ); //max is to deal with negative values
	return res;
}


void Convolution::test() {
	convolute(getImage());
	print_image(getResult());
}


void print_image(std::vector<std::vector<float>>* mvector) {

	std::cout << "-----------------" << std::endl;
	std::cout << "-----------------" << std::endl;
	
	for (int i = 0; i < mvector[0].size(); i++) {
		for (int j = 0; j < mvector[0][i].size(); j++) {
			std::cout << mvector[0][i][j] << "\t";
		}
		std::cout << std::endl;
	}
	std::cout << "-----------------" << std::endl;
	std::cout << "-----------------" << std::endl;
}


