// reading an entire nifti2 binary file

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <fstream>
#include <string>

//NIFTI 2 inclusion
#include "nifti2.h"
#include "nifti1.h"

#include "BinaryLoader.h"


NiftiFile::NiftiFile(std::string filename)
{
    this->sphereTest = false;
    this->fileName = filename;
    loadFileToMem();
    niftiType = 0;
    if (header.magic == NULL) {
        std::cerr << "WARNING: Header not populated!" << std::endl;
    }
    niftiType = header.sizeof_hdr;
    setTotalDim();

    for (int i = 0; i < header.dim[0]; i++) {
        if ( header.dim[i + 1] > longest_dimension)
            longest_dimension = header.dim[i + 1];
    }

    
}

void NiftiFile::foo()
{
    this->sphereTest = true;
    this->fileName = "sphere";

    header = nifti_2_header();
    
    header.sizeof_hdr = 540;     /*!< MUST be 540           */ /* int sizeof_hdr; (348) */   /*   0 */
    header.magic;      /*!< MUST be valid signature. */  /* char magic[4];    */   /*   4 */
    header.datatype = 16;     /*!< Defines data type!    */ /* short datatype;       */   /*  12 */
    header.bitpix = 32;       /*!< Number bits/voxel.    */ /* short bitpix;         */   /*  14 */
    
    header.dim[0] = 3;       /*!< Data array dimensions.*/ /* short dim[8];         */   /*  16 */
    header.dim[1] = 100;       /*!< Data array dimensions.*/ /* short dim[8];         */   /*  16 */
    header.dim[2] = 100;       /*!< Data array dimensions.*/ /* short dim[8];         */   /*  16 */
    header.dim[3] = 100;       /*!< Data array dimensions.*/ /* short dim[8];         */   /*  16 */
    
    
    header.intent_p1 = 0.0;    /*!< 1st intent parameter. */ /* float intent_p1;      */   /*  80 */
    header.intent_p2 = 0.0;    /*!< 2nd intent parameter. */ /* float intent_p2;      */   /*  88 */
    header.intent_p3 = 0.0;    /*!< 3rd intent parameter. */ /* float intent_p3;      */   /*  96 */
    
    for(int i = 0; i < 8; i ++)
        header.pixdim[i] = 1.0;     /*!< Grid spacings.        */ /* float pixdim[8];      */   /* 104 */
    
    
    header.vox_offset = 0;      /*!< Offset into .nii file */ /* float vox_offset;     */   /* 168 */
    header.scl_slope= 0.0;      /*!< Data scaling: slope.  */ /* float scl_slope;      */   /* 176 */
    header.scl_inter = 0.0;     /*!< Data scaling: offset. */ /* float scl_inter;      */   /* 184 */
    header.cal_max = 255.0;     /*!< Max display intensity */ /* float cal_max;        */   /* 192 */
    header.cal_min = 0.0;       /*!< Min display intensity */ /* float cal_min;        */   /* 200 */
    header.slice_duration = 0.0;/*!< Time for 1 slice.     */ /* float slice_duration; */   /* 208 */
    header.toffset = 0.0;       /*!< Time axis shift.      */ /* float toffset;        */   /* 216 */
    header.slice_start = 0;     /*!< First slice index.    */ /* short slice_start;    */   /* 224 */
    header.slice_end = 100;     /*!< Last slice index.     */ /* short slice_end;      */   /* 232 */
    header.descrip;             /*!< any text you like.    */ /* char descrip[80];     */   /* 240 */
    header.aux_file;            /*!< auxiliary filename.   */ /* char aux_file[24];    */   /* 320 */
    header.qform_code;          /*!< NIFTI_XFORM_* code.   */ /* short qform_code;     */   /* 344 */
    header.sform_code;          /*!< NIFTI_XFORM_* code.   */ /* short sform_code;     */   /* 348 */
    header.quatern_b;           /*!< Quaternion b param.   */ /* float quatern_b;      */   /* 352 */
    header.quatern_c;       /*!< Quaternion c param.   */ /* float quatern_c;      */   /* 360 */
    header.quatern_d;    /*!< Quaternion d param.   */ /* float quatern_d;      */   /* 368 */
    header.qoffset_x;    /*!< Quaternion x shift.   */ /* float qoffset_x;      */   /* 376 */
    header.qoffset_y;    /*!< Quaternion y shift.   */ /* float qoffset_y;      */   /* 384 */
    header.qoffset_z;    /*!< Quaternion z shift.   */ /* float qoffset_z;      */   /* 392 */
    header.srow_x[4];    /*!< 1st row affine transform. */  /* float srow_x[4]; */   /* 400 */
    header.srow_y[4];    /*!< 2nd row affine transform. */  /* float srow_y[4]; */   /* 432 */
    header.srow_z[4];    /*!< 3rd row affine transform. */  /* float srow_z[4]; */   /* 464 */
    header.slice_code;      /*!< Slice timing order.   */ /* char slice_code;      */   /* 496 */
    header.xyzt_units;      /*!< Units of pixdim[1..4] */ /* char xyzt_units;      */   /* 500 */
    header.intent_code;     /*!< NIFTI_INTENT_* code.  */ /* short intent_code;    */   /* 504 */
    header.intent_name[16]; /*!< 'name' or meaning of data. */ /* char intent_name[16]; */  /* 508 */
    header.dim_info;        /*!< MRI slice ordering.   */      /* char dim_info;        */  /* 524 */
    header.unused_str[15];  /*!< unused, filled with \0 */

    loadSphereToMem();
    setTotalDim();
    for (int i = 0; i < header.dim[0]; i++) {
        if (header.dim[i + 1] > longest_dimension)
            longest_dimension = header.dim[i + 1];
    }
}

NiftiFile::NiftiFile() {

this->sphereTest = true;
this->fileName = "empty";

header = nifti_2_header();

header.sizeof_hdr = 540;     /*!< MUST be 540           */ /* int sizeof_hdr; (348) */   /*   0 */
header.magic;      /*!< MUST be valid signature. */  /* char magic[4];    */   /*   4 */
header.datatype = 16;     /*!< Defines data type!    */ /* short datatype;       */   /*  12 */
header.bitpix = 32;       /*!< Number bits/voxel.    */ /* short bitpix;         */   /*  14 */

header.dim[0] = 3;       /*!< Data array dimensions.*/ /* short dim[8];         */   /*  16 */
header.dim[1] = 100;       /*!< Data array dimensions.*/ /* short dim[8];         */   /*  16 */
header.dim[2] = 100;       /*!< Data array dimensions.*/ /* short dim[8];         */   /*  16 */
header.dim[3] = 100;       /*!< Data array dimensions.*/ /* short dim[8];         */   /*  16 */


header.intent_p1 = 0.0;    /*!< 1st intent parameter. */ /* float intent_p1;      */   /*  80 */
header.intent_p2 = 0.0;    /*!< 2nd intent parameter. */ /* float intent_p2;      */   /*  88 */
header.intent_p3 = 0.0;    /*!< 3rd intent parameter. */ /* float intent_p3;      */   /*  96 */

for (int i = 0; i < 8; i++)
    header.pixdim[i] = 1.0;     /*!< Grid spacings.        */ /* float pixdim[8];      */   /* 104 */


header.vox_offset = 0;      /*!< Offset into .nii file */ /* float vox_offset;     */   /* 168 */
header.scl_slope = 0.0;      /*!< Data scaling: slope.  */ /* float scl_slope;      */   /* 176 */
header.scl_inter = 0.0;     /*!< Data scaling: offset. */ /* float scl_inter;      */   /* 184 */
header.cal_max = 255.0;     /*!< Max display intensity */ /* float cal_max;        */   /* 192 */
header.cal_min = 0.0;       /*!< Min display intensity */ /* float cal_min;        */   /* 200 */
header.slice_duration = 0.0;/*!< Time for 1 slice.     */ /* float slice_duration; */   /* 208 */
header.toffset = 0.0;       /*!< Time axis shift.      */ /* float toffset;        */   /* 216 */
header.slice_start = 0;     /*!< First slice index.    */ /* short slice_start;    */   /* 224 */
header.slice_end = 100;     /*!< Last slice index.     */ /* short slice_end;      */   /* 232 */
header.descrip;             /*!< any text you like.    */ /* char descrip[80];     */   /* 240 */
header.aux_file;            /*!< auxiliary filename.   */ /* char aux_file[24];    */   /* 320 */
header.qform_code;          /*!< NIFTI_XFORM_* code.   */ /* short qform_code;     */   /* 344 */
header.sform_code;          /*!< NIFTI_XFORM_* code.   */ /* short sform_code;     */   /* 348 */
header.quatern_b;           /*!< Quaternion b param.   */ /* float quatern_b;      */   /* 352 */
header.quatern_c;       /*!< Quaternion c param.   */ /* float quatern_c;      */   /* 360 */
header.quatern_d;    /*!< Quaternion d param.   */ /* float quatern_d;      */   /* 368 */
header.qoffset_x;    /*!< Quaternion x shift.   */ /* float qoffset_x;      */   /* 376 */
header.qoffset_y;    /*!< Quaternion y shift.   */ /* float qoffset_y;      */   /* 384 */
header.qoffset_z;    /*!< Quaternion z shift.   */ /* float qoffset_z;      */   /* 392 */
header.srow_x[4];    /*!< 1st row affine transform. */  /* float srow_x[4]; */   /* 400 */
header.srow_y[4];    /*!< 2nd row affine transform. */  /* float srow_y[4]; */   /* 432 */
header.srow_z[4];    /*!< 3rd row affine transform. */  /* float srow_z[4]; */   /* 464 */
header.slice_code;      /*!< Slice timing order.   */ /* char slice_code;      */   /* 496 */
header.xyzt_units;      /*!< Units of pixdim[1..4] */ /* char xyzt_units;      */   /* 500 */
header.intent_code;     /*!< NIFTI_INTENT_* code.  */ /* short intent_code;    */   /* 504 */
header.intent_name[16]; /*!< 'name' or meaning of data. */ /* char intent_name[16]; */  /* 508 */
header.dim_info;        /*!< MRI slice ordering.   */      /* char dim_info;        */  /* 524 */
header.unused_str[15];  /*!< unused, filled with \0 */

loadZEROCornerSphereToMem();
for (int i = 0; i < header.dim[0]; i++) {
    if (header.dim[i + 1] > longest_dimension)
        longest_dimension = header.dim[i + 1];
}
}

void NiftiFile::displayNIFTI2Header() {
    std::cout << "SIZEOF_HDR: " << header.sizeof_hdr << std::endl;
    std::cout << "MAGIC: " << header.magic << std::endl;
    std::cout << "DATATYPE: " << header.datatype << std::endl;
    std::cout << "BITPIX: " << header.bitpix << std::endl;

    std::cout << "DIM:";
    int arrSize = sizeof(header.dim) / sizeof(header.dim[0]);
    for (int i = 0; i < (arrSize); i++) {
        std::cout << "\t" << header.dim[i];
    }
    std::cout << std::endl;

    std::cout << "INTENT_P1: " << header.intent_p1 << std::endl;
    std::cout << "INTENT_P2: " << header.intent_p2 << std::endl;
    std::cout << "INTENT_P3: " << header.intent_p3 << std::endl;

    std::cout << "PIXDIM:";
    arrSize = sizeof(header.pixdim) / sizeof(header.pixdim[0]);
    for (int i = 0; i < (arrSize); i++) {
        std::cout << "\t" << header.pixdim[i];
    }
    std::cout << std::endl;

    std::cout << "VOX_OFFSET: " << header.vox_offset << std::endl;

    std::cout << "SCL_SLOPE: " << header.scl_slope << std::endl;
    std::cout << "SCL_INTER: " << header.scl_inter << std::endl;

    std::cout << "CAL_MAX: " << header.cal_max << std::endl;
    std::cout << "CAL_MIN: " << header.cal_min << std::endl;

    std::cout << "SLICE_DURATION: " << header.slice_duration << std::endl;
    std::cout << "TOFFSET: " << header.toffset << std::endl;
    std::cout << "SLICE START: " << header.slice_start << std::endl;
    std::cout << "SLICE END: " << header.slice_end << std::endl;
    std::cout << "DESCRIP: " << header.descrip << std::endl;
    std::cout << "AUX FILE:" << header.aux_file << std::endl;

    std::cout << "QFORM_CODE:" << header.qform_code << std::endl;
    std::cout << "SFORM_CODE:" << header.sform_code << std::endl;

    std::cout << "QUATERN_B:" << header.quatern_b << std::endl;
    std::cout << "QUATERN_C:" << header.quatern_c << std::endl;
    std::cout << "QUATERN_D:" << header.quatern_d << std::endl;

    std::cout << "QOFFSET_X:" << header.qoffset_x << std::endl;
    std::cout << "QOFFSET_Y:" << header.qoffset_y << std::endl;
    std::cout << "QOFFSET_Z:" << header.qoffset_z << std::endl;

    std::cout << "SROW_X:\t" << header.srow_x[0] << "\t" << header.srow_x[1] << "\t" << header.srow_x[2] << "\t" << header.srow_x[3] << std::endl;
    std::cout << "SROW_Y:\t" << header.srow_y[0] << "\t" << header.srow_y[1] << "\t" << header.srow_y[2] << "\t" << header.srow_y[3] << std::endl;
    std::cout << "SROW_Z:\t" << header.srow_z[0] << "\t" << header.srow_z[1] << "\t" << header.srow_z[2] << "\t" << header.srow_z[3] << std::endl;

    std::cout << "SLICE_CODE: " << header.slice_code << std::endl;
    std::cout << "XYZT_UNITS: " << header.xyzt_units << std::endl;
    std::cout << "INTENT_CODE: " << header.intent_code << std::endl;
    std::cout << "INTENT_NAME: " << header.intent_name << std::endl;
    std::cout << "DIM_INFO: " << header.dim_info << std::endl;
    std::cout << "UNUSED_STR: " << header.unused_str << std::endl;
}

/**
* search for the volume index
* only works for data with dim[0] == 3
* v - 3 cordiante vector
* returns - the corresponding index
*/
__host__ __device__ int NiftiFile::transformVector3Position(glm::vec3 v) {
    // x * (Y * Z) + y * (Z) + z
    return (int)v.x * header.dim[2] * header.dim[3] + (int)v.y * header.dim[3] + (int)v.z;

}

__host__ __device__ bool NiftiFile::isInside(glm::vec3 point)
{
    return point.x >= 0.0f && point.x < this->header.dim[1] &&
        point.y >= 0.0f && point.y < this->header.dim[2] &&
        point.z >= 0.0f && point.z < this->header.dim[3];
}

__host__ __device__ glm::vec3 NiftiFile::toVolumeSpace(glm::vec3 point) {

    glm::mat4 tranlate = glm::translate(glm::mat4(1.0f),
        glm::vec3(
            0.5f,
            0.5f,
            0.5f));
    glm::mat4 scale = glm::scale(glm::mat4(1.0f), glm::vec3(longest_dimension, longest_dimension, longest_dimension));

    point = tranlate * glm::vec4(point, 1.0f);
    point = scale * glm::vec4(point, 1.0f);

    tranlate = glm::translate(glm::mat4(1.0f),
        glm::vec3(
            header.dim[1] / 2.0f - longest_dimension / 2.0f,
            header.dim[2] / 2.0f - longest_dimension / 2.0f,
            header.dim[3] / 2.0f - longest_dimension / 2.0f
        ));

    point = tranlate * glm::vec4(point, 1.0f);

    return point;
}



int NiftiFile::loadFileToMem() {

    std::streampos size;

    std::ifstream file(fileName /*"avg152T1_LR_nifti2.nii"*/, std::ios::in | std::ios::binary  | std::ios::ate);
    if (file.is_open())
    {
        size = file.tellg();

        std::cout << "FILE SIZE = " << size << std::endl;

        //NIFTI2 HEADER
        file.seekg(0, std::ios::beg);
        file.read((char*)&header, sizeof(nifti_2_header));
        
        switch(header.sizeof_hdr){

        case 540:
            std::cout << "Loading NIFTI 2 file" << std::endl;
            break;
        case 348:
            std::cout << "Loading NIFTI 1 file" << std::endl;
            //file.seekg(0, std::ios::beg);
            //file.read((char*)&header, sizeof(nifti_1_header));

            break;
        default:
            std::cerr << "file isn't in a vaild format, maybe file read in wrong endianess!: " << header.sizeof_hdr << std::endl;
            return 1;
        }


        //NIFTI2 DATA
        //nifti2 file data starts after byte 544

        /*|------------------|
          |GENERATE STRUCTURE|
          |------------------|*/

        //its known for this particular file that its dimensions are 3 (x,y,z respectively)
        int vsize = header.dim[1]*header.dim[2]*header.dim[3];
        volume = new float[vsize];

        //start reading from byte offset, for nifti2 is ususally 544, https://brainder.org/2015/04/03/the-nifti-2-file-format/
        file.seekg (header.vox_offset, std::ios::beg); 
          
        //std::cout << "\nCONTENT START"<< std::endl;
       
        //std::cout << "File good: " << file.good() << std::endl;

        file.read((char*) volume, vsize * sizeof(float));
          
        //std::cout << "File end: " << file.eof() << std::endl;
          
        //std::cout << "\nCONTENT END" << std::endl;

        //CLOSE FILE
        file.close();
        std::cout << "the entire file content is in memory" << std::endl;
    }
    else std::cout << "Unable to open file";
    return 0;
}


int NiftiFile::loadSphereToMem() {

    glm::vec3 center = glm::vec3( 100.0f/2,100.0f/2,100.0f/2 );
    float radius = 50.0f;
    float* res = (float* )malloc(sizeof(float) * 100 * 100 * 100);

    for (int i = 0; i < pow(100, 3); i++) {
        res[i] = 0.0f;
    }

    std::cout << "Zero'd sphere malloc!" << std::endl;

    for (int x = 0; x < 100; x++)
        for (int y = 0; y < 100; y++)
            for (int z = 0; z < 100; z++) {
                int index = x * 100 * 100 + y * 100 + z;
                if (pow(x - center[0], 2) +
                    pow(y - center[1], 2) +
                    pow(z - center[2], 2) <= pow(radius, 2)) {
                    
                    res[index] = y/100.0f * 255.0f;
                }

            }

    volume = res;

    std::cout << "sphere loaded into empty nifti file" << std::endl;
    return 0;
}

int NiftiFile::loadZEROCornerSphereToMem() {

    glm::vec3 center = glm::vec3(0.0f , 0.0f, 0.0f);
    float radius = 100.0f;
    float* res = (float*)malloc(sizeof(float) * 100 * 100 * 100);

    for (int i = 0; i < pow(100, 3); i++) {
        res[i] = 0.0f;
    }

    std::cout << "Zero'd sphere malloc!" << std::endl;

    for (int x = 0; x < 100; x++)
        for (int y = 0; y < 100; y++)
            for (int z = 0; z < 100; z++) {
                int index = x * 100 * 100 + y * 100 + z;
                if (pow(x - center[0], 2) +
                    pow(y - center[1], 2) +
                    pow(z - center[2], 2) <= pow(radius, 2)) {

                    res[index] = (pow(x - center[0], 2) + pow(y - center[1], 2) + pow(z - center[2], 2)) / pow(radius, 2) * 255.0f;
                }

            }

    volume = res;

    std::cout << "sphere loaded into empty nifti file" << std::endl;
    return 0;
}


NiftiFile::~NiftiFile() {
    if (sphereTest) {
        free(volume);
        std::cout << "destroying nifitiFile sphere volume" << std::endl;
    }

}

void NiftiFile::setTotalDim() {
    int aux = 1;
    for (int i = 0; i < header.dim[0]; i++) {
        aux = aux * header.dim[i + 1];
    }
    totaldim = aux;
}