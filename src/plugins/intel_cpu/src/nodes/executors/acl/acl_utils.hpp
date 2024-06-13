// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "memory_desc/cpu_memory_desc.h"
#include "arm_compute/core/Types.h"

namespace ov {
namespace intel_cpu {

/**
* @brief ACL supports arm_compute::MAX_DIMS maximum. The method squashes the last
* dimensions in order to comply with this limitation
* @param dims vector of dimensions to squash
* @return vector of dimensions that complies to ACL
*/
inline VectorDims collapse_dims_to_max_rank(VectorDims dims) {
    const size_t MAX_NUM_SHAPE = arm_compute::MAX_DIMS;
    VectorDims result_dims(MAX_NUM_SHAPE - 1);
    if (dims.size() >= MAX_NUM_SHAPE) {
        for (size_t i = 0; i < MAX_NUM_SHAPE - 1; i++) {
            result_dims[i] = dims[i];
        }
        for (size_t i = MAX_NUM_SHAPE - 1; i < dims.size(); i++) {
            result_dims[MAX_NUM_SHAPE - 2] *= dims[i];
        }
    } else {
        result_dims = dims;
    }
    return result_dims;
}

/**
* @brief ACL handles NH_C specifically, it thinks it is NC_W, so we need to change layout manually:
* e.g. NCHW (0, 1, 2, 3) -> NHWC (0, 2, 3, 1)
* @param _listDims list of dimensions to convert
* @return none
*/

inline void changeLayoutToNH_C(const std::vector<arm_compute::TensorShape*> &_listDims) {
    auto mover = [](arm_compute::TensorShape &_shape) {
        if (_shape.num_dimensions() > 4) { std::swap(_shape[2], _shape[3]); }
        if (_shape.num_dimensions() > 3) { std::swap(_shape[1], _shape[2]); }
        if (_shape.num_dimensions() > 2) { std::swap(_shape[0], _shape[1]); }
    };

    for (auto& dims : _listDims) {
        mover(*dims);
    }
}

/**
* @brief Return ComputeLibrary TensorShape with reverted layout schema used in ACL 
* @param dims vector of dimensions to convert
* @return ComputeLibrary TensorShape object
*/
inline arm_compute::TensorShape shapeCast(const VectorDims& dims) {
    arm_compute::TensorShape tensorShape;
    for (std::size_t i = 0; i < dims.size(); ++i) {
        tensorShape.set(dims.size() - i - 1, dims[i], false);
    }
    if (tensorShape.num_dimensions() == 0) {
        tensorShape.set(0, 1, false);
        tensorShape.set_num_dimensions(1);
    }
    return tensorShape;
}

enum ACLAxisCastMode {
    NO_LAYOUT_CONVERSION,
    NHWC_TO_NCHW,
    NCHW_TO_NHWC
};

/**
* @brief Return reverted axis used in ACL. If axis cast mode is  
* @param axis axis that needs to be converted
* @param shapeSize size of the shape, which axis needs to be converted
* @param axisCastMode specifies whether layout conversion is required or not
* @return reverted axis
*/
inline int axisCast(const std::size_t axis, const std::size_t shapeSize, ACLAxisCastMode axisCastMode = NO_LAYOUT_CONVERSION) {
    // CWHN (reverted NHWC) (0, 1, 2, 3) into WHCN (reverted NCHW) (1, 2, 0, 3)
    static std::vector<size_t> nhwcToNchw = {1, 2, 0, 3};
    // WHCN (reverted NCHW) (0, 1, 2, 3) into CWHN (reverted NHWC) (2, 0, 1, 3)
    static std::vector<size_t> nchwToNhwc = {2, 0, 1, 3};
    size_t revertedAxis = shapeSize - axis - 1;
    switch (axisCastMode) {
        case NHWC_TO_NCHW:
            return revertedAxis > 3 ? -1 : nhwcToNchw[revertedAxis];
        case NCHW_TO_NHWC:
            return revertedAxis > 3 ? -1 : nchwToNhwc[revertedAxis];
        default:
            return revertedAxis;
    }
}

inline Dim vectorProduct(const VectorDims& vec, size_t size) {
    Dim prod = 1;
    for (size_t i = 0; i < size; ++i)
        prod *= vec[i];
    return prod;
}

/**
* @brief Return ComputeLibrary DataType that corresponds to the given precision
* @param precision precision to be converted
* @return ComputeLibrary DataType or UNKNOWN if precision is not mapped to DataType
*/
inline arm_compute::DataType precisionToAclDataType(ov::element::Type precision) {
    switch (precision) {
        case ov::element::i8:    return arm_compute::DataType::S8;
        case ov::element::u8:    return arm_compute::DataType::U8;
        case ov::element::i16:   return arm_compute::DataType::S16;
        case ov::element::u16:   return arm_compute::DataType::U16;
        case ov::element::i32:   return arm_compute::DataType::S32;
        case ov::element::u32:   return arm_compute::DataType::U32;
        case ov::element::f16:  return arm_compute::DataType::F16;
        case ov::element::f32:  return arm_compute::DataType::F32;
        case ov::element::f64:  return arm_compute::DataType::F64;
        case ov::element::i64:   return arm_compute::DataType::S64;
        case ov::element::bf16:  return arm_compute::DataType::BFLOAT16;
        default:                                return arm_compute::DataType::UNKNOWN;
    }
}

/**
* @brief Return ComputeLibrary DataLayout that corresponds to MemoryDecs layout
* @param desc MemoryDecs from which layout is retrieved
* @param treatAs4D the flag that treats MemoryDecs as 4D shape
* @return ComputeLibrary DataLayout or UNKNOWN if MemoryDecs layout is not mapped to DataLayout
*/
inline arm_compute::DataLayout getAclDataLayoutByMemoryDesc(MemoryDescCPtr desc) {
    if (desc->hasLayoutType(LayoutType::ncsp)) {
        if (desc->getShape().getRank() <= 4) return arm_compute::DataLayout::NCHW;
        if (desc->getShape().getRank() == 5) return arm_compute::DataLayout::NCDHW;
    } else if (desc->hasLayoutType(LayoutType::nspc)) {
        if (desc->getShape().getRank() <= 4) return arm_compute::DataLayout::NHWC;
        if (desc->getShape().getRank() == 5) return arm_compute::DataLayout::NDHWC;
    }
    return arm_compute::DataLayout::UNKNOWN;
}

/**
* @brief run thread-safe configure for ComputeLibrary configuration function.
* Arm Compute Library 23.08 does not officially support thread-safe configure() calls.
* For example, calling configure for Eltwise operations from multiple streams leads to a data race and seg fault.
* @param config ComputeLibrary configuration function
*/
void configureThreadSafe(const std::function<void(void)>& config);

}   // namespace intel_cpu
}   // namespace ov
