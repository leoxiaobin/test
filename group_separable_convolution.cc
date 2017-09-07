/*!
* Copyright (c) 2015 by Contributors
* \file convolution.cc
* \brief
* \author Bing Xu
*/

#include "./group_separable_convolution-inl.h"
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>

namespace mshadow {

	template<typename Dtype>
	inline void GroupSeparableConv2dForward(const Tensor<cpu, 4, Dtype> &out,
		const Tensor<cpu, 4, Dtype> &data,
		const Tensor<cpu, 3, Dtype> &wmat,
		const mxnet::op::sepconv::DepthwiseArgs args) {
		// NOT_IMPLEMENTED;


		return;
	}

	template<typename Dtype>
	inline void GroupSeparableConv2dBackwardInput(const Tensor<cpu, 4, Dtype> &in_grad,
		const Tensor<cpu, 4, Dtype> &out_grad,
		const Tensor<cpu, 3, Dtype> &wmat,
		const mxnet::op::sepconv::DepthwiseArgs args) {
		// NOT_IMPLEMENTED;
		return;
	}

	template<typename Dtype>
	inline void GroupSeparableConv2dBackwardFilter(const Tensor<cpu, 4, Dtype> &out_grad,
		const Tensor<cpu, 4, Dtype> &input,
		const Tensor<cpu, 3, Dtype> &wmat_diff,
		const mxnet::op::sepconv::DepthwiseArgs args) {
		// NOT_IMPLEMENTED;
		return;
	}

}  // namespace mshadow


namespace mxnet {
	namespace op {
		DMLC_REGISTER_PARAMETER(GroupSeparableConvolutionParam);

		template<>
		Operator* CreateOp<cpu>(GroupSeparableConvolutionParam param, int dtype,
			std::vector<TShape> *in_shape,
			std::vector<TShape> *out_shape,
			Context ctx) {
			Operator *op = NULL;

			MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
				op = new GroupSeparableConvolutionOp<cpu, DType>(param);
			})
				return op;
		}

		// DO_BIND_DISPATCH comes from operator_common.h
		Operator *GroupSeparableConvolutionProp::CreateOperatorEx(Context ctx,
			std::vector<TShape> *in_shape,
			std::vector<int> *in_type) const {
			std::vector<TShape> out_shape, aux_shape;
			std::vector<int> out_type, aux_type;
			CHECK(InferType(in_type, &out_type, &aux_type));
			CHECK(InferShape(in_shape, &out_shape, &aux_shape));
			DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
		}

		MXNET_REGISTER_OP_PROPERTY(GroupSeparableConvolution, GroupSeparableConvolutionProp)
			.add_argument("data", "Symbol", "Input data to the SeparableConvolutionOp.")
			.add_argument("weight", "Symbol", "Weight matrix.")
			.add_argument("bias", "Symbol", "Bias parameter.")
			.add_arguments(GroupSeparableConvolutionParam::__FIELDS__())
			.describe("Apply separable convolution to input then add a bias.");

	}  // namespace op
}  // namespace mxnet
