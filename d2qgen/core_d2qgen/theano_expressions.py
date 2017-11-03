'''
====Attributions====
Copyright (c) 2014 Université de Montréal
 
 Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

'''

"""Frequently used Theano expressions."""
from theano import tensor


def l2_norm(tensors):
    """Computes the total L2 norm of a set of tensors.

    Converts all operands to :class:`~tensor.TensorVariable`
    (see :func:`~tensor.as_tensor_variable`).

    Parameters
    ----------
    tensors : iterable of :class:`~tensor.TensorVariable` (or compatible)
        The tensors.

    """
    flattened = [tensor.as_tensor_variable(t).flatten() for t in tensors]
    flattened = [(t if t.ndim > 0 else t.dimshuffle('x'))
                 for t in flattened]
    joined = tensor.join(0, *flattened)
    return tensor.sqrt(tensor.sqr(joined).sum())


def hessian_times_vector(gradient, parameter, vector, r_op=False):
    """Return an expression for the Hessian times a vector.

    Parameters
    ----------
    gradient : :class:`~tensor.TensorVariable`
        The gradient of a cost with respect to `parameter`
    parameter : :class:`~tensor.TensorVariable`
        The parameter with respect to which to take the gradient
    vector : :class:`~tensor.TensorVariable`
        The vector with which to multiply the Hessian
    r_op : bool, optional
        Whether to use :func:`~tensor.gradient.Rop` or not. Defaults to
        ``False``. Which solution is fastest normally needs to be
        determined by profiling.

    """
    if r_op:
        return tensor.Rop(gradient, parameter, vector)
    return tensor.grad(tensor.sum(gradient * vector), parameter)
