import numpy as np
import torch


def to_lasagne_format(x, mask=False):
    """
    Assumes data is of shape (n[, nt], nx, ny).
    Reshapes to (n, n_channels, nx, ny[, nt])
    Note: Depth must be the last axis, the dimensions will be reordered
    """
    if x.ndim == 4:  # n 3D inputs. reorder axes
        x = np.transpose(x, (0, 2, 3, 1))

    if mask:  # Hacky solution
        x = x * (1 + 1j)

    x, real, imag = complex2real(x)

    return x, real, imag


def complex2real(x):
    '''
    Parameter
    ---------
    x: ndarray
        assumes at least 2d. Last 2D axes are split in terms of real and imag
        2d/3d/4d complex valued tensor (n, nx, ny) or (n, nx, ny, nt)

    Returns
    -------
    y: 4d tensor (n, 2, nx, ny)
    '''
    x_real = np.real(x)
    # np.imag(x)返回的是虚部的实数数值，而不是带有虚部单位j的复数形式。
    x_imag = np.imag(x)
    y = np.concatenate((x_real, x_imag), axis=0)
    # # re-order in convenient order
    # if x.ndim > 3:
    #     y = y.swapaxes(0, 1)
    return y, x_real, x_imag


def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2) ** 2)


def _maybe_view_as_subclass(original_array, new_array):
    if type(original_array) is not type(new_array):
        # if input was an ndarray subclass and subclasses were OK,
        # then view the result as that subclass.
        new_array = new_array.view(type=type(original_array))
        # Since we have done something akin to a view from original_array, we
        # should let the subclass finalize (if it has it implemented, i.e., is
        # not None).
        if new_array.__array_finalize__:
            new_array.__array_finalize__(original_array)
    return new_array


class DummyArray:
    """Dummy object that just exists to hang __array_interface__ dictionaries
    and possibly keep alive a reference to a base array.
    """

    def __init__(self, interface, base=None):
        self.__array_interface__ = interface
        self.base = base

def as_strided(x, shape=None, strides=None, subok=False, writeable=True):
    """
    Create a view into the array with the given shape and strides.

    .. warning:: This function has to be used with extreme care, see notes.

    Parameters
    ----------
    x : ndarray
        Array to create a new.
    shape : sequence of int, optional
        The shape of the new array. Defaults to ``x.shape``.
    strides : sequence of int, optional
        The strides of the new array. Defaults to ``x.strides``.
    subok : bool, optional
        .. versionadded:: 1.10

        If True, subclasses are preserved.
    writeable : bool, optional
        .. versionadded:: 1.12

        If set to False, the returned array will always be readonly.
        Otherwise it will be writable if the original array was. It
        is advisable to set this to False if possible (see Notes).

    Returns
    -------
    view : ndarray

    See also
    --------
    broadcast_to : broadcast an array to a given shape.
    reshape : reshape an array.
    lib.stride_tricks.sliding_window_view :
        userfriendly and safe function for the creation of sliding window views.

    Notes
    -----
    ``as_strided`` creates a view into the array given the exact strides
    and shape. This means it manipulates the internal data structure of
    ndarray and, if done incorrectly, the array elements can point to
    invalid memory and can corrupt results or crash your program.
    It is advisable to always use the original ``x.strides`` when
    calculating new strides to avoid reliance on a contiguous memory
    layout.

    Furthermore, arrays created with this function often contain self
    overlapping memory, so that two elements are identical.
    Vectorized write operations on such arrays will typically be
    unpredictable. They may even give different results for small, large,
    or transposed arrays.

    Since writing to these arrays has to be tested and done with great
    care, you may want to use ``writeable=False`` to avoid accidental write
    operations.

    For these reasons it is advisable to avoid ``as_strided`` when
    possible.
    """
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)
    interface = dict(x.__array_interface__)
    if shape is not None:
        interface['shape'] = tuple(shape)
    if strides is not None:
        interface['strides'] = tuple(strides)

    array = np.asarray(DummyArray(interface, base=x))
    # The route via `__interface__` does not preserve structured
    # dtypes. Since dtype should remain unchanged, we set it explicitly.
    array.dtype = x.dtype

    view = _maybe_view_as_subclass(x, array)

    if view.flags.writeable and not writeable:
        view.flags.writeable = False

    return view

def cartesian_mask(shape, acc, sample_n=10, centred=False):
    """
    Sampling density estimated from implementation of kt FOCUSS

    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..

    """
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(Ny, 0.5 / (Ny / 10.) ** 2)
    lmda = Ny / (2. * acc)
    n_lines = int(Ny / acc)

    # add uniform distribution
    pdf_x += lmda * 1. / Ny

    if sample_n:
        pdf_x[Ny // 2 - sample_n // 2:Ny // 2 + sample_n // 2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Ny))
    for i in range(N):
        idx = np.random.choice(Ny, n_lines, False, pdf_x)
        mask[i, idx] = 1

    if sample_n:
        mask[:, Ny // 2 - sample_n // 2:Ny // 2 + sample_n // 2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Ny, 0, size))

    mask = mask.reshape(shape)

    if not centred:
        mask = np.fft.ifftshift(mask, axes=(-1, -2))

    return mask