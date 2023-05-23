

#include "Octree.h"

/*
Octree::OctreeOld(NiftiFile* nf) {

	this->nf = nf;

	//get the highest dimension
	this->hdim = 0;
	for (int i = 0; i < 3; i++) {
		if (hdim < nf->header.dim[i + 1])
			hdim = nf->header.dim[i + 1];
	}

	int max_depth = 0;	
	while (std::pow(2, max_depth) < hdim)
		max_depth++;

	this->root = (Node*) malloc(sizeof(Node));
	this->root = new Node(max_depth,pow(2,max_depth), nf); //HERE changed hdim to 128
	std::cout << "Octree Root High|Low Values: " << this->root->high_value << "|" << this->root->low_value << std::endl;
}*/

//////////////
//	OCTREE	//
//////////////

Octree::Octree(NiftiFile* nf) {

	this->nf = nf;
	maximum_depth = 0;
	longest_dimension = 0;
	for (int dimension_index = 0; dimension_index < 3; dimension_index++) {
		if (longest_dimension < nf->header.dim[dimension_index + 1])
			longest_dimension = nf->header.dim[dimension_index + 1];
	}

	while (std::pow(2, maximum_depth) < longest_dimension)
		maximum_depth++;

	number_of_nodes= 0;

	for (int power = 0; power < maximum_depth+1; power++) {
		number_of_nodes += (int)pow(8,power);
	}

	octree_array = (Node*)malloc(number_of_nodes * sizeof(Node));
	initializeOctreeNodes();
	updateOctreeNodes();
	std::cout << "Octree Root High|Low Values: " << this->octree_array[0].maximum_value << "|" << this->octree_array[0].minimum_value << std::endl;
}

/*
float Octree::searchPointGetIntensityOLD(glm::vec3 point) {
	float res = 0.0f;
	res = root->searchPointGetIntensity(point);
	return res;
}
*/
/*
Octree::OctreeOldDestructor() {
	delete root;
}*/

Octree::~Octree() {
	free(this->octree_array);
}

void Octree::initializeOctreeNodes() {
	createNode(0, 0, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 1.0f, 1.0f));
}

void Octree::updateOctreeNodes() {
	updateNode(0);
}

void Octree::updateNode(unsigned int index) {

	Node* node = &this->octree_array[index];

	if (isLeaf(index)) {

		glm::mat4 scaleMatrix = glm::mat4(1.0);
		scaleMatrix = glm::scale(scaleMatrix, glm::vec3(longest_dimension, longest_dimension, longest_dimension));
		//scaleMatrix = glm::scale(scaleMatrix, glm::vec3(nf->header.dim[1], nf->header.dim[2], nf->header.dim[3]));
		glm::vec3 res = scaleMatrix * glm::vec4(node->lower_corner, 1);

		//center the dataset on the volume
		if (
			res.x >= (longest_dimension / 2.0f) - (nf->header.dim[1] / 2.0f) && res.x < (longest_dimension / 2.0f) + (nf->header.dim[1] / 2.0f) &&
			res.y >= (longest_dimension / 2.0f) - (nf->header.dim[2] / 2.0f) && res.y < (longest_dimension / 2.0f) + (nf->header.dim[2] / 2.0f) &&
			res.z >= (longest_dimension / 2.0f) - (nf->header.dim[3] / 2.0f) && res.z < (longest_dimension / 2.0f) + (nf->header.dim[3] / 2.0f))
		{

			res = glm::vec3(
				(int)(res.x + (nf->header.dim[1] / 2.0f) - longest_dimension / 2.0f),
				(int)(res.y + (nf->header.dim[2] / 2.0f) - longest_dimension / 2.0f),
				(int)(res.z + (nf->header.dim[3] / 2.0f) - longest_dimension / 2.0f));

			node->maximum_value = nf->volume[nf->transformVector3Position(res)];
			node->minimum_value = node->maximum_value;
		}
		else { //position is out of dataset range
			node->maximum_value = 0.0f;
			node->minimum_value = 0.0f;
		}

	}
	else { 
		for (int child_number = 1; child_number <= 8; child_number++) {
			updateNode((8 * index) + child_number);
		}

		for (int child_number = 1; child_number <= 8; child_number++) {

			Node* child_node = &octree_array[8 * index + child_number];

			if (node->maximum_value < child_node->maximum_value) {
				node->maximum_value = child_node->maximum_value;
			}
			if (node->minimum_value > child_node->minimum_value) {
				node->minimum_value = child_node->minimum_value;
			}
		}

	}
}

void Octree::createNode(int index, int depth, glm::vec3 lower_corner, glm::vec3 upper_corner) {
		
	octree_array[index] = {depth, 0.0f, 0.0f, lower_corner, upper_corner };

	if (!isLeaf(index)) {
		glm::vec3 distance = upper_corner - lower_corner;
		for (int x = 0; x < 2; x++)
			for (int y = 0; y < 2; y++)
				for (int z = 0; z < 2; z++) {
					int child_number = x * 4 + y * 2 + z + 1;
					int child_index = 8 * index + child_number;

					glm::vec4 l(lower_corner, 1.0f);
					glm::mat4 trans = glm::mat4(1.0f);
					trans = glm::translate(trans, glm::vec3(x * distance.x / 2, y * distance.y / 2, z * distance.y / 2));
					glm::vec3 child_lower_corner = trans * l;

					glm::vec4 u(child_lower_corner, 1.0f);
					trans = glm::mat4(1.0f);
					trans = glm::translate(trans, glm::vec3(distance.x / 2, distance.y / 2, distance.y / 2));
					glm::vec3 child_upper_corner = trans * u;

					createNode(child_index, depth + 1, child_lower_corner, child_upper_corner);
				}
	}
}

float Octree::getIntensity(glm::vec3 point) {
	return searchPointGetIntensity(0, point);
}

float Octree::searchPointGetIntensity(unsigned int index, glm::vec3 point) {
	float res = 0.0f;

	Node* node = &this->octree_array[index];

	if (isInside(index, point)) {
		if (node->maximum_value == node->minimum_value) { // early stop
			res = node->maximum_value;
		}
		else {
			for (int child = 1; child <= 8; child++) {

				int child_index = index * 8 + child;
				float aux = searchPointGetIntensity(child_index, point);
				if (aux > res)
					res = aux;
			}
		}
	}

	return res;
}

//TESTING CUBIC INTERPOLATION DO NOT USE
float Octree::searchPointGetIntensityPrinted(unsigned int index, glm::vec3 point) {
	float res = 0.0f;

	Node* node = &this->octree_array[index];

	if (isInside(index, point)) {
		
		std::cout << "NODE INDEX" << index << std::endl;
		std::cout << "NODE DEPTH" << node->depth << std::endl;
		std::cout << "NODE Lower corner: "
			<< node->lower_corner.x << " "
			<< node->lower_corner.y << " "
			<< node->lower_corner.z << " "
			<< std::endl;
		std::cout << "NODE Upper corner: "
			<< node->upper_corner.x << " "
			<< node->upper_corner.y << " "
			<< node->upper_corner.z << " "
			<< std::endl;

		std::cout << "NODE m:" << node->minimum_value << std::endl;

		std::cout << "NODE M:" << node->maximum_value << std::endl << std::endl;
		
		

		if (node->maximum_value == node->minimum_value) { // early stop
			res = node->maximum_value;
			if (isLeaf(index)) {
				std::cout << "Reached Leaf!" << std::endl;

				glm::vec3 aux = node->upper_corner - node->lower_corner;

				//soma das distancias

				float total_distance = 0.0f;
				for (int x = 0; x < 2; x++) {
					for (int y = 0; y < 2; y++) {
						for (int z = 0; z < 2; z++) {
							//add all distances
							node->lower_corner + glm::vec3(x * aux.x, y*aux.y ,z*aux.z);
						}
					}
				}

			}
			else {
				std::cout << "Early stop!" << std::endl;
			}
		}
		else {
			for (int child = 1; child <= 8; child++) {

				int child_index = index * 8 + child;
				float aux = searchPointGetIntensityPrinted(child_index, point);
				if (aux > res)
					res = aux;
			}
		}


	}

	return res;
}

bool Octree::isLeaf(unsigned int index) {
	bool result = this->octree_array[index].depth == this->maximum_depth;
	return result;
}

bool Octree::isInside(unsigned int index, glm::vec3 p) {




	Node* node = &octree_array[index];
	return (p.x >= node->lower_corner.x &&
		p.y >= node->lower_corner.y &&
		p.z >= node->lower_corner.z &&
		p.x < node->upper_corner.x&&
		p.y < node->upper_corner.y&&
		p.z < node->upper_corner.z);
}


//////////////////
//DEVICE METHODS//
//////////////////

__device__ bool Octree::device_isInside(unsigned int index, glm::vec3 p) {
	Node* node = &octree_array[index];
	return (p.x >= node->lower_corner.x &&
		p.y >= node->lower_corner.y &&
		p.z >= node->lower_corner.z &&
		p.x < node->upper_corner.x&&
		p.y < node->upper_corner.y&&
		p.z < node->upper_corner.z);
}

__device__ float Octree::device_getIntensity(glm::vec3 point) {
	return device_searchPointGetIntensity(0, point);
}

__device__ float Octree::device_searchPointGetIntensity(unsigned int index, glm::vec3 point) {
	float res = 0.0f;
		
	Node* node = &this->octree_array[index];

	if (device_isInside(index, point)) {
		if (node->maximum_value == node->minimum_value) { // early stop
			res = node->maximum_value;
		}
		else {
			for (int child = 1; child <= 8; child++) {

				int child_index = index * 8 + child;
				float aux = device_searchPointGetIntensity(child_index, point);
				if (aux > res)
					res = aux;
			}
		}
	}

	return res;
}
