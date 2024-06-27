// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A file contains public utilities.
 * @file utils.hpp
 */
#pragma once

#include "snippets/emitter.hpp"
#include "snippets/shape_types.hpp"
#include "snippets/lowered/expression.hpp"
#include "snippets/lowered/expression_port.hpp"

#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/constant.hpp"


namespace ov {
namespace snippets {
namespace utils {

// Get non-scalar Constant count that will be created after FakeQuantize decomposition.
// This count is needed to know exact count of non-scalar Constants during tokenization.
auto get_non_scalar_constant_count_for_fq(const std::shared_ptr<ov::op::v0::FakeQuantize>& fq) -> size_t;

inline auto is_scalar_constant(const std::shared_ptr<ov::Node>& source_output_node) -> bool {
    return ov::is_type<ov::op::v0::Constant>(source_output_node) && ov::shape_size(source_output_node->get_shape()) == 1;
}

inline auto normalize_rank(int32_t allocation_rank, const size_t shape_rank) -> int32_t {
    return allocation_rank < 0 ? allocation_rank + static_cast<int32_t>(shape_rank) + 1 : allocation_rank;
}

template <typename T, typename P>
constexpr bool one_of(T val, P item) { return val == item; }

template <typename T, typename P, typename... Args>
constexpr bool one_of(T val, P item, Args... item_others) {
    return val == item || one_of(val, item_others...);
}

template <typename T, typename P>
constexpr bool everyone_is(T val, P item) { return val == item; }

template <typename T, typename P, typename... Args>
constexpr bool everyone_is(T val, P item, Args... item_others) {
    return val == item && everyone_is(val, item_others...);
}

constexpr inline bool implication(bool cause, bool cond) {
    return !cause || !!cond;
}

template <typename T, typename U>
inline T div_up(const T a, const U b) {
    OPENVINO_ASSERT(b != 0, "Divider must not be zero");
    return static_cast<T>((a + b - 1) / b);
}

template<typename T, typename = typename std::enable_if<(std::is_same<T, size_t>::value || std::is_same<T, int64_t>::value), bool>::type>
constexpr inline T get_dynamic_value() {
    return std::numeric_limits<T>::max();
}

template<typename T, typename = typename std::enable_if<(std::is_same<T, size_t>::value || std::is_same<T, int64_t>::value), bool>::type>
constexpr inline bool is_dynamic_value(T value) {
    return value == get_dynamic_value<T>();
}

inline bool is_dynamic_vdims(const VectorDims& shape) {
    return std::any_of(shape.cbegin(), shape.cend(), [](size_t v){ return is_dynamic_value(v); });
}

inline bool is_dynamic_vdims(const VectorDimsPtr& shape) {
    return is_dynamic_vdims(*shape);
}

template<typename T, typename = typename std::enable_if<(std::is_same<T, size_t>::value || std::is_same<T, int64_t>::value), bool>::type>
inline T dynamic_safe_add(const T& lhs, const T& rhs) {
    if (utils::is_dynamic_value(lhs) || utils::is_dynamic_value(rhs)) {
        return utils::get_dynamic_value<T>();
    }
    return lhs + rhs;
}

template<typename T, typename = typename std::enable_if<(std::is_same<T, size_t>::value || std::is_same<T, int64_t>::value), bool>::type>
inline T dynamic_safe_mul(const T& lhs, const T& rhs) {
    if (utils::is_dynamic_value(lhs) || utils::is_dynamic_value(rhs)) {
        return utils::get_dynamic_value<T>();
    }
    return lhs * rhs;
}

template<typename T, typename = typename std::enable_if<(std::is_same<T, size_t>::value || std::is_same<T, int64_t>::value), bool>::type>
inline std::string value2str(const T& value) {
    return utils::is_dynamic_value(value) ? "?" : std::to_string(value);
}

template<typename T, typename = typename std::enable_if<(std::is_same<T, size_t>::value || std::is_same<T, int64_t>::value), bool>::type>
std::string vector2str(const std::vector<T>& values) {
    std::ostringstream str;
    bool first = true;
    for (auto& v : values) {
        if (!first) {
            str << ",";
        }
        str << value2str(v);
        first = false;
    }
    return str.str();
}

bool broadcast_merge_dim(size_t& dst, const size_t& d1, const size_t& d2);

VectorDims pshape_to_vdims(const PartialShape&);
ov::PartialShape vdims_to_pshape(const VectorDims&);

// dim_idx starts from the layout end: dim_idx = 0 -> last element in layout (layout.back())
inline size_t get_input_dim_idx(const std::vector<size_t>& layout, size_t dim_idx) {
    OPENVINO_ASSERT(dim_idx < layout.size(), "Incorrect dim_idx");
    return *(layout.rbegin() + dim_idx);
}
// dim_idx starts from the layout end: dim_idx = 0 -> last index in layout (layout.size() - 1)
inline size_t get_output_dim_idx(const std::vector<size_t>& layout, size_t dim_idx) {
    OPENVINO_ASSERT(dim_idx < layout.size(), "Incorrect dim_idx");
    return std::distance(layout.cbegin(), std::find(layout.cbegin(), layout.cend(), layout.size() - 1 - dim_idx));
}

// dim_idx starts from the layout end
size_t get_dim_idx(const lowered::ExpressionPort& port, size_t dim_idx);

/* ----- Shape `getters` ----- */
/**
 * @brief Returns a dense shape after applying the order.
 *        It means that the shape dimensions will be reordered in accordance with order indices to produce planar shape
 * @param shape preordered (original) partial shape
 * @param order order
 * @return reordered partial shape: `planar_shape[i]` = `shape[order[i]]`
 *         Example, shape = [16, 2, 32, 64], order = [2, 0, 1, 3]
 *                  planar_shape = [32, 16, 2, 64]
 */
ov::PartialShape get_planar_pshape(const ov::PartialShape& shape, const std::vector<size_t>& order);
/**
 * @brief Returns original shape before applying the order.
 *        It means that the shape dimensions have been already reordered in accordance with order indices to produce planar shape
 * @param shape planar (ordered) partial shape
 * @param order order
 * @return preordered partial shape: `shape[i]` = `planar_shape[order[i]]` where `shape` is shape before applying the order.
 *         Example, shape = [16, 2, 32, 64], order = [2, 0, 1, 3]
 *                  planar_shape = [2, 32, 16, 64]
 */
ov::PartialShape get_preordered_pshape(const ov::PartialShape& shape, const std::vector<size_t>& order);
/**
 * @brief Returns a dense shape of node input.
 *        It means that the node input shape dimensions will be reordered in accordance with order indices to produce planar shape
 * @param in input of node
 * @return new reordered partial shape: `planar_shape[i]` = `shape[order[i]]`
 */
ov::PartialShape get_planar_pshape(const Input<Node>& in);
/**
 * @brief Returns original shape of node output before applying the order.
 *        It means that the preordered output shape dimensions have been already reordered in accordance with order indices to produce planar shape
 * @param out output of node
 * @return preordered partial shape: `shape[i]` = `planar_shape[order[i]]` where `shape` is shape before applying the order.
 */
ov::PartialShape get_preordered_pshape(const Output<Node>& out);
/**
 * @brief Returns a dense shape after applying the order.
 *        It means that the shape dimensions will be reordered in accordance with order indices to produce planar shape
 * @param shape preordered (original) shape
 * @param order order
 * @return reordered partial shape: `planar_shape[i]` = `shape[order[i]]`
 *         Example, shape = [16, 2, 32, 64], order = [2, 0, 1, 3]
 *                  planar_shape = [32, 16, 2, 64]
 */
VectorDims get_planar_vdims(const VectorDims& shape, const std::vector<size_t>& order);
/**
 * @brief Returns original shape before applying the order.
 *        It means that the preordered shape dimensions have been already reordered in accordance with order indices to produce planar shape
 * @param shape planar (ordered) shape
 * @param order order
 * @return preordered shape: `shape[i]` = `planar_shape[order[i]]` where `shape` is shape before applying the order.
 *         Example, shape = [16, 2, 32, 64], order = [2, 0, 1, 3]
 *                  planar_shape = [2, 32, 16, 64]
 */
VectorDims get_preordered_vdims(const VectorDims& shape, const std::vector<size_t>& order);
/**
 * @brief Returns a dense shape of expression input port.
 *        It means that the input shape dimensions will be reordered in accordance with order indices to produce planar shape
 * @param expr_port input expression port
 * @return new reordered partial shape: `planar_shape[i]` = `shape[order[i]]`
 */
VectorDims get_planar_vdims(const snippets::lowered::ExpressionPort& expr_port);
/**
 * @brief Returns original shape before applying the order of expression output port.
 *        It means that the preordered output shape dimensions has been already reordered in accordance with order indices to produce planar shape
 * @param out input of node
 * @return preordered shape: `shape[i]` = `planar_shape[order[i]]` where `shape` is shape before applying the order.
 */
VectorDims get_preordered_vdims(const snippets::lowered::ExpressionPort& expr_port);
/* --------------------------- */

/**
 * @brief Returns element count of a shape
 * @param shape input shape
 * @return element count of input shape
 */
inline auto get_shape_size(const VectorDims& shape) -> size_t {
    return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
}

/**
 * @brief Get zero to several consecutive child shape infer exprs(such as reshape, rankNormalization) from start_expr.
 *        As Node maybe have multiple output. This functions return the first(left) legal sequence.
 * @param start_expr Collect from start_expr.
 * @return shape infer expression consumers as a sequence.
 */
std::vector<lowered::ExpressionPtr> get_first_child_shape_infer_expr_seq(const lowered::ExpressionPtr& start_expr);

/**
 * @brief Get zero to several consecutive parent shape infer exprs(such as reshape, rankNormalization) from start_expr.
 *        As Node maybe have multiple input. This functions return the first(left) legal sequence.
 * @param start_expr Collect from start_expr.
 * @return shape infer expression sources as a sequence.
 */
std::vector<lowered::ExpressionPtr> get_first_parent_shape_infer_expr_seq(const lowered::ExpressionPtr& start_expr);

/**
 * @brief Get leaf shape infer node in first child shape infer sequence from(include) start_node.
 *        If start_node is a not shape infer node and start_node has no child shape infer node, function will return nullptr.
 * @param start_node Search from start_node.
 * @return the leaf shape infer node of first child shape infer sequence or nullptr.
 */
std::shared_ptr<ov::Node> get_leaf_node_of_first_child_shape_infer_seq(const std::shared_ptr<ov::Node>& start_node);

/**
 * @brief Get leaf shape infer node in first parent shape infer sequence from(include) start_node.
 *        If start_node is a not shape infer node and start_node has no parent shape infer node, function will return nullptr.
 * @param start_node Search from start_node.
 * @return the first leaf shape infer node or nullptr.
 */
std::shared_ptr<ov::Node> get_leaf_node_of_first_parent_shape_infer_seq(const std::shared_ptr<ov::Node>& start_node);

/**
 * @brief Calculate leading dimension of the shape that should be read according to the layout
 * @param shape original (not reordered) input shape
 * @param layout specifies the order in what dimensions of in the input shape should be read
 * @return stride of the dimension idx = layout[layout.size() - 2] in the original shape
   Example:
         Original shape (shape) = [1, 49, 2, 23]
         Layout (transpose order) = [2, 0, 1, 3]

         dim_idx = layout.size() - 2 = 2
         // Since layout specifies the order of dimensions in which the shape should be read
         dim = layout[dim_idx] = 1
         stride(shape[1]) = shape[2] * shape[3] = 2 * 23
 */
size_t get_in_leading_dim(const VectorDims& shape, const std::vector<size_t>& layout);
inline size_t get_in_leading_dim(const lowered::PortDescriptorPtr& pd) {
    return get_in_leading_dim(pd->get_shape(), pd->get_layout());
}
/**
 *
 * @param shape reordered input shape that is stored according to the layout
 * @param layout specifies the order in what the dimensions of the input shape are stored
 * @return
     Output shape is already transposed, we need to correctly write the data with original shape by the order
     Example:
          Original transposed shape (shape) = [49, 2, 7, 39]
          Layout (transpose order) = [2, 0, 1, 3]

          dim_idx = layout.size() - 2 = 2
          // Since the shape dimensions are already reordered according to the layout
          dim = /find dim_idx index in layout/ = 0
          stride(shape[0]) = shape[1] * shape[2] * shape[3] = 2 * 7 * 39
 */
size_t get_out_leading_dim(const VectorDims& shape, const std::vector<size_t>& layout);
inline size_t get_out_leading_dim(const lowered::PortDescriptorPtr& pd) {
    return get_out_leading_dim(pd->get_shape(), pd->get_layout());
}

} // namespace utils
} // namespace snippets
} // namespace ov
