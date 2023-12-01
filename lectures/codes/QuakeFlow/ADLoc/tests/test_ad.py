# %%
import torch
import torch.nn as nn
from torch.autograd import Function
from numba import njit
import numpy as np


@njit
def compute(grad_input, grad_output, weight):
    for i in range(grad_output.shape[0]):
        for j in range(grad_output.shape[1]):
            grad_input[i, :] += grad_output[i, j] * weight[j, :]
    return grad_input


# Inherit from Function
class LinearFunction(Function):
    # Note that forward, setup_context, and backward are @staticmethods
    @staticmethod
    def forward(input, weight, bias):
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx, inputs, output):
        input, weight, bias = inputs
        ctx.save_for_backward(input, weight, bias)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        # print(grad_output.shape, weight.shape)
        grad_output = grad_output.numpy()
        weight = weight.numpy()
        grad_input = np.zeros(input.shape)

        # for i in range(grad_output.shape[0]):
        #     for j in range(grad_output.shape[1]):
        #         grad_input[i, :] += grad_output[i, j] * weight[j, :]
        grad_input = compute(grad_input, grad_output, weight)

        grad_input = torch.from_numpy(grad_input)
        # print(grad_weight)

        return grad_input, grad_weight, grad_bias


class Linear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter("bias", None)

        # self.weight.requires_grad = False
        # self.bias.requires_grad = False
        # Not a very smart way to initialize weights
        nn.init.uniform_(self.weight, -0.1, 0.1)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return "input_features={}, output_features={}, bias={}".format(
            self.input_features, self.output_features, self.bias is not None
        )


torch.manual_seed(0)
linear = Linear(5, 3)
input = torch.randn(2, 5)
input.requires_grad = True
output = linear(input)
grad_output = output.backward(torch.randn(2, 3))
print(output)
print(grad_output)
# print(linear)

# %%
