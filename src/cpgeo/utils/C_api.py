
import ctypes
import os
current_path = os.path.dirname(os.path.abspath(__file__))

faceApi = ctypes.CDLL(os.path.dirname(current_path) + '/dlls/CPGEO.dll')
faceApi.build_trees.restype = ctypes.POINTER(ctypes.c_void_p)
