import numpy as np
import math

OMEGA = 1.1

EPS = math.ulp(1.0)**4


E = np.array([
    [0,0],
    # Sides
    [0,1],
    [0,-1],
    [1,0],
    [-1,0],
    # Corners
    [1,1],
    [-1,-1],
    [1,-1],
    [-1,1]
],dtype=int)

XMIRR = np.apply_along_axis(lambda r: np.where(r)[0],0,np.array([[-E[i,0],E[i,1]]==E for i in np.arange(0,9)]).all(axis=2))[0]
YMIRR = np.apply_along_axis(lambda r: np.where(r)[0],0,np.array([[E[i,0],-E[i,1]]==E for i in np.arange(0,9)]).all(axis=2))[0]
XYMIRR = np.apply_along_axis(lambda r: np.where(r)[0],0,np.array([[-E[i,0],-E[i,1]]==E for i in np.arange(0,9)]).all(axis=2))[0]

W = np.array([
    4/9,
    # Sides
    1/9,
    1/9,
    1/9,
    1/9,
    #Corners
    1/36,
    1/36,
    1/36,
    1/36
],dtype=float)

