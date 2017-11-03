'''
====Attributions====
Copyright (c) 2014 Université de Montréal
 
 Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

'''

from __future__ import print_function
import sys
import contextlib
from collections import OrderedDict

import numpy
import six
import theano
from theano import tensor
from theano import printing
from theano.gof.graph import Constant
from theano.tensor.shared_randomstreams import RandomStateSharedVariable
from theano.tensor.sharedvar import SharedVariable


def pack(arg):
    """Pack variables into a list.

    Parameters
    ----------
    arg : object
        Either a list or tuple, or any other Python object. Lists will be
        returned as is, and tuples will be cast to lists. Any other
        variable will be returned in a singleton list.

    Returns
    -------
    list
        List containing the arguments

    """
    if isinstance(arg, (list, tuple)):
        return list(arg)
    else:
        return [arg]


def unpack(arg, singleton=False):
    """Unpack variables from a list or tuple.

    Parameters
    ----------
    arg : object
        Either a list or tuple, or any other Python object. If passed a
        list or tuple of length one, the only element of that list will
        be returned. If passed a tuple of length greater than one, it
        will be cast to a list before returning. Any other variable
        will be returned as is.
    singleton : bool
        If ``True``, `arg` is expected to be a singleton (a list or tuple
        with exactly one element) and an exception is raised if this is not
        the case. ``False`` by default.

    Returns
    -------
    object
        A list of length greater than one, or any other Python object
        except tuple.

    """
    if isinstance(arg, (list, tuple)):
        if len(arg) == 1:
            return arg[0]
        else:
            if singleton:
                raise ValueError("Expected a singleton, got {}".
                                 format(arg))
            return list(arg)
    else:
        return arg


def shared_floatx_zeros(shape, **kwargs):
    r"""Creates a shared variable array filled with zeros.

    Parameters
    ----------
    shape : tuple
        A tuple of integers representing the shape of the array.
    \*\*kwargs
        Keyword arguments to pass to the :func:`shared_floatx` function.

    Returns
    -------
    :class:'tensor.TensorSharedVariable'
        A Theano shared variable filled with zeros.

    """
    return shared_floatx(numpy.zeros(shape), **kwargs)


def shared_floatx_nans(shape, **kwargs):
    r"""Creates a shared variable array filled with nans.

    Parameters
    ----------
    shape : tuple
         A tuple of integers representing the shape of the array.
    \*\*kwargs
        Keyword arguments to pass to the :func:`shared_floatx` function.

    Returns
    -------
    :class:'tensor.TensorSharedVariable'
        A Theano shared variable filled with nans.

    """
    return shared_floatx(numpy.nan * numpy.zeros(shape), **kwargs)


def shared_floatx(value, name=None, borrow=False, dtype=None):
    """Transform a value into a shared variable of type floatX.

    Parameters
    ----------
    value : :class:`~numpy.ndarray`
        The value to associate with the Theano shared.
    name : :obj:`str`, optional
        The name for the shared variable. Defaults to `None`.
    borrow : :obj:`bool`, optional
        If set to True, the given `value` will not be copied if possible.
        This can save memory and speed. Defaults to False.
    dtype : :obj:`str`, optional
        The `dtype` of the shared variable. Default value is
        :attr:`config.floatX`.

    Returns
    -------
    :class:`tensor.TensorSharedVariable`
        A Theano shared variable with the requested value and `dtype`.

    """
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(theano._asarray(value, dtype=dtype),
                         name=name,
                         borrow=borrow)


def shared_like(variable, name=None):
    """Construct a shared variable to hold the value of a tensor variable.

    Parameters
    ----------
    variable : :class:`~tensor.TensorVariable`
        The variable whose dtype and ndim will be used to construct
        the new shared variable.
    name : :obj:`str` or :obj:`None`
        The name of the shared variable. If None, the name is determined
        based on variable's name.

    """
    variable = tensor.as_tensor_variable(variable)
    if name is None:
        name = "shared_{}".format(variable.name)
    return theano.shared(numpy.zeros((0,) * variable.ndim,
                                     dtype=variable.dtype),
                         name=name)


def reraise_as(new_exc):
    """Reraise an exception as a different type or with a message.

    This function ensures that the original traceback is kept, making for
    easier debugging.

    Parameters
    ----------
    new_exc : :class:`Exception` or :obj:`str`
        The new error to be raised e.g. (ValueError("New message"))
        or a string that will be prepended to the original exception
        message

    Notes
    -----
    Note that when reraising exceptions, the arguments of the original
    exception are cast to strings and appended to the error message. If
    you want to retain the original exception arguments, please use:

    >>> try:
    ...     1 / 0
    ... except Exception as e:
    ...     reraise_as(Exception("Extra information", *e.args))
    Traceback (most recent call last):
      ...
    Exception: 'Extra information, ...

    Examples
    --------
    >>> class NewException(Exception):
    ...     def __init__(self, message):
    ...         super(NewException, self).__init__(message)
    >>> try:
    ...     do_something_crazy()
    ... except Exception:
    ...     reraise_as(NewException("Informative message"))
    Traceback (most recent call last):
      ...
    NewException: Informative message ...

    """
    orig_exc_type, orig_exc_value, orig_exc_traceback = sys.exc_info()

    if isinstance(new_exc, six.string_types):
        new_exc = orig_exc_type(new_exc)

    if hasattr(new_exc, 'args'):
        if len(new_exc.args) > 0:
            # We add all the arguments to the message, to make sure that this
            # information isn't lost if this exception is reraised again
            new_message = ', '.join(str(arg) for arg in new_exc.args)
        else:
            new_message = ""
        new_message += '\n\nOriginal exception:\n\t' + orig_exc_type.__name__
        if hasattr(orig_exc_value, 'args') and len(orig_exc_value.args) > 0:
            if getattr(orig_exc_value, 'reraised', False):
                new_message += ': ' + str(orig_exc_value.args[0])
            else:
                new_message += ': ' + ', '.join(str(arg)
                                                for arg in orig_exc_value.args)
        new_exc.args = (new_message,) + new_exc.args[1:]

    new_exc.__cause__ = orig_exc_value
    new_exc.reraised = True
    six.reraise(type(new_exc), new_exc, orig_exc_traceback)


def check_theano_variable(variable, n_dim, dtype_prefix):
    """Check number of dimensions and dtype of a Theano variable.

    If the input is not a Theano variable, it is converted to one. `None`
    input is handled as a special case: no checks are done.

    Parameters
    ----------
    variable : :class:`~tensor.TensorVariable` or convertible to one
        A variable to check.
    n_dim : int
        Expected number of dimensions or None. If None, no check is
        performed.
    dtype : str
        Expected dtype prefix or None. If None, no check is performed.

    """
    if variable is None:
        return

    if not isinstance(variable, tensor.Variable):
        variable = tensor.as_tensor_variable(variable)

    if n_dim and variable.ndim != n_dim:
        raise ValueError("Wrong number of dimensions:"
                         "\n\texpected {}, got {}".format(
                             n_dim, variable.ndim))

    if dtype_prefix and not variable.dtype.startswith(dtype_prefix):
        raise ValueError("Wrong dtype prefix:"
                         "\n\texpected starting with {}, got {}".format(
                             dtype_prefix, variable.dtype))


def named_copy(variable, new_name):
    """Clones a variable and set a new name to the clone."""
    result = variable.copy()
    result.name = new_name
    return result


def is_graph_input(variable):
    """Check if variable is a user-provided graph input.

    To be considered an input the variable must have no owner, and not
    be a constant or shared variable.

    Parameters
    ----------
    variable : :class:`~tensor.TensorVariable`

    Returns
    -------
    bool
        ``True`` If the variable is a user-provided input to the graph.

    """
    return (not variable.owner and
            not isinstance(variable, SharedVariable) and
            not isinstance(variable, Constant))


def is_shared_variable(variable):
    """Check if a variable is a Theano shared variable.

    Notes
    -----
    This function excludes shared variables that store the state of Theano
    random number generators.

    """
    return (isinstance(variable, SharedVariable) and
            not isinstance(variable, RandomStateSharedVariable) and
            not hasattr(variable.tag, 'is_rng'))


def dict_subset(dict_, keys, pop=False, must_have=True):
    """Return a subset of a dictionary corresponding to a set of keys.

    Parameters
    ----------
    dict_ : dict
        The dictionary.
    keys : iterable
        The keys of interest.
    pop : bool
        If ``True``, the pairs corresponding to the keys of interest are
        popped from the dictionary.
    must_have : bool
        If ``True``, a ValueError will be raised when trying to retrieve a
        key not present in the dictionary.

    Returns
    -------
    result : ``OrderedDict``
        An ordered dictionary of retrieved pairs. The order is the same as
        in the ``keys`` argument.

    """
    not_found = object()

    def extract(k):
        if pop:
            if must_have:
                return dict_.pop(k)
            return dict_.pop(k, not_found)
        if must_have:
            return dict_[k]
        return dict_.get(k, not_found)

    result = [(key, extract(key)) for key in keys]
    return OrderedDict([(k, v) for k, v in result if v is not not_found])


def dict_union(*dicts, **kwargs):
    r"""Return union of a sequence of disjoint dictionaries.

    Parameters
    ----------
    dicts : dicts
        A set of dictionaries with no keys in common. If the first
        dictionary in the sequence is an instance of `OrderedDict`, the
        result will be OrderedDict.
    \*\*kwargs
        Keywords and values to add to the resulting dictionary.

    Raises
    ------
    ValueError
        If a key appears twice in the dictionaries or keyword arguments.

    """
    dicts = list(dicts)
    if dicts and isinstance(dicts[0], OrderedDict):
        result = OrderedDict()
    else:
        result = {}
    for d in list(dicts) + [kwargs]:
        duplicate_keys = set(result.keys()) & set(d.keys())
        if duplicate_keys:
            raise ValueError("The following keys have duplicate entries: {}"
                             .format(", ".join(str(key) for key in
                                               duplicate_keys)))
        result.update(d)
    return result


def repr_attrs(instance, *attrs):
    r"""Prints a representation of an object with certain attributes.

    Parameters
    ----------
    instance : object
        The object of which to print the string representation
    \*attrs
        Names of attributes that should be printed.

    Examples
    --------
    >>> class A(object):
    ...     def __init__(self, value):
    ...         self.value = value
    >>> a = A('a_value')
    >>> repr(a)  # doctest: +SKIP
    <blocks.utils.A object at 0x7fb2b4741a10>
    >>> repr_attrs(a, 'value')  # doctest: +SKIP
    <blocks.utils.A object at 0x7fb2b4741a10: value=a_value>

    """
    orig_repr_template = ("<{0.__class__.__module__}.{0.__class__.__name__} "
                          "object at {1:#x}")
    if attrs:
        repr_template = (orig_repr_template + ": " +
                         ", ".join(["{0}={{0.{0}}}".format(attr)
                                    for attr in attrs]))
    repr_template += '>'
    orig_repr_template += '>'
    try:
        return repr_template.format(instance, id(instance))
    except Exception:
        return orig_repr_template.format(instance, id(instance))


def put_hook(variable, hook_fn, *args):
    r"""Put a hook on a Theano variables.

    Ensures that the hook function is executed every time when the value
    of the Theano variable is available.

    Parameters
    ----------
    variable : :class:`~tensor.TensorVariable`
        The variable to put a hook on.
    hook_fn : function
        The hook function. Should take a single argument: the variable's
        value.
    \*args : list
        Positional arguments to pass to the hook function.

    """
    return printing.Print(global_fn=lambda _, x: hook_fn(x, *args))(variable)


def ipdb_breakpoint(x):
    """A simple hook function for :func:`put_hook` that runs ipdb.

    Parameters
    ----------
    x : :class:`~numpy.ndarray`
        The value of the hooked variable.

    """
    import ipdb
    ipdb.set_trace()


def print_sum(x, header=None):
    if not header:
        header = 'print_sum'
    print(header + ':', x.sum())


def print_shape(x, header=None):
    if not header:
        header = 'print_shape'
    print(header + ':', x.shape)


@contextlib.contextmanager
def change_recursion_limit(limit):
    """Temporarily changes the recursion limit."""
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(limit)
    yield
    sys.setrecursionlimit(old_limit)
