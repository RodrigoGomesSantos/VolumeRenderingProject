
#include <iostream>
#include <fstream>

#include "TransferFunction.h"

using mat_range = std::tuple<Material::Material, float, float>;

TransferFunction::TransferFunction() {

    std::vector<mat_range> materials;
    /*
    materials.push_back(mat_range{ Material::getMaterialFromID(Material::MaterialId::empty),  0.0f, 0.1f});
    materials.push_back(mat_range{ Material::getMaterialFromID(Material::MaterialId::glass),  0.1f, 0.2f});
    materials.push_back(mat_range{ Material::getMaterialFromID(Material::MaterialId::muscle),  0.2f, 0.3f});
    materials.push_back(mat_range{ Material::getMaterialFromID(Material::MaterialId::bone),  0.3f, 0.5f});
    */
    
    materials.push_back(mat_range{ Material::getMaterialFromID(Material::MaterialId::empty),  0.0f, 1.0f });
    
    materials.push_back(mat_range{ Material::getMaterialFromID(Material::MaterialId::bone),  30.0f/255.0f, 80.0f / 255.0f });
    materials.push_back(mat_range{ Material::getMaterialFromID(Material::MaterialId::muscle),  140.0f/255.0f, 160.0f / 255.0f });
    materials.push_back(mat_range{ Material::getMaterialFromID(Material::MaterialId::brain),  105.0f / 255.0f, 120.0f / 255.0f });
    //materials.push_back(mat_range{ Material::getMaterialFromID(Material::MaterialId::empty), 149.0f/255.0f, 149.0f / 255.0f });
    //materials.push_back(mat_range{ Material::getMaterialFromID(Material::MaterialId::blue),  206.0f/255.0f, 1.0f });
    
    size = materials.size();

    material_intervals = (mat_interval*) malloc( size * sizeof(mat_interval));

    for (int i = 0; i < size; i++) {
        material_intervals[i] = 
            mat_interval { 
            (Material::Material) std::get<0>(materials[i]), 
            (float)std::get<1>(materials[i]), 
            (float)std::get<2>(materials[i])
        };
    }
}

TransferFunction::~TransferFunction()
{
    free(material_intervals);
}

__host__ __device__ Material::Material* TransferFunction::getMaterial(float value) {
    Material::Material* result = &material_intervals[0].material;
    for (int i = 0; i < this->size; i++) {
        if (value >= material_intervals[i].lower_bound &&
            value <= material_intervals[i].higher_bound) {
            result = &material_intervals[i].material;
        }
    }
    return result;
}

void TransferFunction::print() {
    std::cout << "#-------------------#" << std::endl;
    std::cout << "| Transfer Function |" << std::endl;
    std::cout << "#-------------------#" << std::endl;
    for (int i = 0; i < size; i++) {
        mat_interval* m = &(this->material_intervals[i]);
        std::cout << m->material.name<< " " << m->lower_bound << " " << m->higher_bound << std::endl;
    }
    std::cout << "#-------------------#" << std::endl;
}
