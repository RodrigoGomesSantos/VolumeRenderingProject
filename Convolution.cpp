


#include <cmath>
#include <vector>
#include <iostream>

#include "Convolution.h"

//forward declarations
using volume = std::vector<
	std::vector<
	std::vector<
	float>>>;
struct point { int x, y, z; };

void generateSphereData(volume* data);
void printLayer(volume* data, int layer);
void instanciate_padded_data(volume* data, volume* padded_data, int pad, float pad_value);

void convolute_point(volume* padded_data, point p, volume* convolution_volume_matrix,
	volume* convoluted_data, int pad);
void convolute_3D(volume* padded_data, volume* convoluted_data, volume* data,
	volume* convolution_volume_matrix, int pad);

void convolution() {

	//represents a volume
	volume data;

	//populate the volume
	generateSphereData(&data);

	//print a slice of the volume
	int layer = 5;
	printLayer(&data, layer);

	//pad the volume
	volume padded_data;
	instanciate_padded_data(&data, &padded_data, 1, 0.0f);

	//print the padded volume
	printLayer(&padded_data, layer + 1);

	//setup convolution matrix its a 3x3x3 matrix? (its more like a volume)
	volume convolution_volume_matrix =
	{	//x == 0 layer
		{{0.0f,0.0f,0.0f},
		{0.0f,0.1f,0.0f},
		{0.0f,0.0f,0.0f}},
		//x == 1 layer
		{{0.0f,0.1f,0.0f},
		{0.1f,5.0f,0.1f},
		{0.0f,0.1f,0.0f}},
		//x == 2 layer
		{{0.0f,0.0f,0.0f},
		{0.0f,0.1f,0.0f},
		{0.0f,0.0f,0.0f}}
	};

	//convolute
	volume convoluted_data;
	generateSphereData(&convoluted_data);
	convolute_3D(&padded_data, &convoluted_data, &data, &convolution_volume_matrix, 1);

	//display convolution result
	printLayer(&convoluted_data, layer);
}



//implementations

void insertPadLine(std::vector<float>* pad_line, int pad, int size, float pad_value) {
	for (int i = 0; i < size + 2 * pad; i++) {
		pad_line->push_back(pad_value);
	}
}

void insertPadLayer(std::vector<std::vector<float>>* pad_layer, int pad, int x_size, int y_size, float pad_value) {
	for (int x = 0; x < x_size + 2 * pad; x++) {
		pad_layer->push_back(std::vector<float>{});
		insertPadLine(&pad_layer->at(x), pad, y_size, pad_value);
	}
}

void instanciate_padded_data(volume* data, volume* padded_data, int pad, float pad_value) {

	int x_size = data->size();
	int y_size = data->at(0).size();
	int z_size = data->at(0).at(0).size();

	//generates empty padded volume
	for (int x = 0; x < x_size + 2 * pad; x++) {
		padded_data->push_back({});
		for (int y = 0; y < y_size + 2 * pad; y++) {
			padded_data->at(x).push_back({});
			for (int z = 0; z < z_size + 2 * pad; z++) {
				padded_data->at(x).at(y).push_back(pad_value);
			}
		}
	}

	//fill empty padded volume
	for (int x = 0; x < x_size; x++) {
		for (int y = 0; y < y_size; y++) {
			for (int z = 0; z < z_size; z++) {
				padded_data[0][x + pad][y + pad][z + pad] = data[0][x][y][z];
			}
		}
	}
}

void generateSphereData(volume* data) {

	int x_size = 10;
	int y_size = 10;
	int z_size = 10;

	float center[3] = { 4.5f,4.5f,4.5f };
	float radius = 3.0f;

	for (int x = 0; x < x_size; x++) {
		data->push_back({});
		for (int y = 0; y < y_size; y++) {
			data->at(x).push_back({});
			for (int z = 0; z < z_size; z++) {
				if (pow(x - center[0], 2) +
					pow(y - center[1], 2) +
					pow(z - center[2], 2)
					<= radius * radius)
					data->at(x).at(y).push_back(1.0f);
				else
					data->at(x).at(y).push_back(0.0f);
			}
		}
	}

}


void printLayer(volume* data, int layer) {

	int x_size = data->size();
	int y_size = data->at(0).size();
	int z_size = data->at(0).at(0).size();

	if (layer >= x_size) {
		std::cout << "Invalid layer number" << std::endl;
		return;
	}
	std::cout << "/////////////////////////////" << std::endl;
	std::cout << "//volume layer values print//" << std::endl;
	std::cout << "/////////////////////////////" << std::endl;
	for (int y = 0; y < y_size; y++) {
		for (int z = 0; z < z_size; z++) {
			std::cout << data[0][layer][y][z] << "\t";
		}
		std::cout << std::endl;
	}

}

void convolute_point(volume* padded_data, point p, volume* convolution_volume_matrix,
	volume* convoluted_data, int pad) {

	float sum_result = 0.0f;
	for (int x = -pad; x <= pad; x++) {
		for (int y = -pad; y <= pad; y++) {
			for (int z = -pad; z <= pad; z++) {
				sum_result +=
					padded_data[0]
					[(p.x + pad) + x]
				[(p.y + pad) + y]
				[(p.z + pad) + z]
				*
					convolution_volume_matrix[0]
					[x + pad]
				[y + pad]
				[z + pad];
			}
		}
	}
	convoluted_data[0][p.x][p.y][p.z] = sum_result;
}


void convolute_3D(
	volume* padded_data,
	volume* convoluted_data,
	volume* data,
	volume* convolution_volume_matrix,
	int pad) {

	int x_size = data->size();
	int y_size = data->at(0).size();
	int z_size = data->at(0).at(0).size();

	for (int x = 0; x < x_size; x++) {
		for (int y = 0; y < y_size; y++) {
			for (int z = 0; z < z_size; z++) {
				convolute_point(padded_data, { x,y,z }, convolution_volume_matrix, convoluted_data, pad);
			}
		}
	}
}




