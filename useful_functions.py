import numpy as np
import math
import cv2 as cv

def add_circle_to_array(arr:np.ndarray, center, radius, value = 1, relative = False):
    """
    Multifunctional method.
    
    To all values within a specified circle, sets True or False values when the array contains booleans.

    To all values within a specified circle, set value or add value if relative is set to True when the array contains floats. This works also with int but the value is then converted to int.
    """
    for x in np.arange(0,arr.shape[0]):
        xdist = np.abs(center[0] - x)
        if xdist > radius:
            continue
        for y in np.arange(0,arr.shape[0]):
            ydist = np.abs(center[1] - y)
            if ydist > radius:
                continue
            r = np.sqrt(xdist**2 + ydist**2)
            if arr.dtype==bool and r < radius:
                if type(value) == bool:
                    arr[x,y] = value
                else:
                    arr[x,y] = True
            elif arr.dtype==float and r < radius:
                if relative:
                    arr[x,y] += value
                else:
                    arr[x,y] = value
            elif arr.dtype==int and r < radius:
                if relative:
                    arr[x,y] = int(arr[x,y] + value)
                else:
                    arr[x,y] = int(value)
    return arr



def show_array(windowName:str, array:np.ndarray, colormap:str = 'normal'):
    image = np.zeros(shape=(array.shape[0],array.shape[1],3))
    match colormap:
        case 'normal' | 1:
            image[:,:,0] = array
            image[:,:,1] = array
            image[:,:,2] = array
        case 'cyclic' | 2:
            xScale = 1
            yScale = 0.85
            zScale = 0.74
            image[:,:,0] = np.sin(math.pi*(array%xScale))
            image[:,:,1] = np.sin(math.pi*(array%yScale))
            image[:,:,2] = np.sin(math.pi*(array%zScale))
        
    cv.imshow(windowName, image)


def show_two_arrays(windowName:str, array1:np.ndarray, array2:np.ndarray):
    image = np.zeros(shape=(array1.shape[0],array1.shape[1],3))
    image[:,:,0] = array1
    image[:,:,1] = array2

    cv.imshow(windowName, image)

def show_three_arrays(windowName:str, array1:np.ndarray, array2:np.ndarray, array3:np.ndarray):
    image = np.zeros(shape=(array1.shape[0],array1.shape[1],3))
    image[:,:,0] = array1
    image[:,:,1] = array2
    image[:,:,2] = array3

    cv.imshow(windowName, image)
