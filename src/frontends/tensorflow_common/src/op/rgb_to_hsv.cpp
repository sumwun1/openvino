// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_rgb_to_hsv_op(const NodeContext& node) {
    default_op_checks(node, 1, {"RBGToHSV"});
    auto images = node.get_input(0);
    auto node_name = node.get_name();

    auto const_minus_one_i = make_shared<v0::Constant>(element::i32, Shape{}, -1);
    auto channels = make_shared<v1::Split>(images, const_minus_one_i, 3);

    auto rr = channels->output(0);
    auto gg = channels->output(1);
    auto bb = channels->output(2);

    auto new_images = rgb_to_hsv(rr, gg, bb);

    set_node_name(node_name, new_images);
    return {new_images};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
