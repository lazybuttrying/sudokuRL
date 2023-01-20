import numpy as np
import torch
import ctypes
c = ctypes.CDLL("test.so")


# arr = [1, 2]
# arr = (ctypes.c_int * len(arr))(*arr)


# arr = np.zeros((3, 3,1)).astype(dtype=np.float64)
# arr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

arr = torch.zeros((3, 3, 1)).numpy().astype(
    dtype=np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))


print(c.score(arr))
