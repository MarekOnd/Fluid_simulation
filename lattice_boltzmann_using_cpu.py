
import cv2 as cv
import numpy as np
import time

from numba import njit, jit

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

INV = np.array([
    0,
    2,
    1,
    4,
    3,
    6,
    5,
    8,
    7
])

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

OMEGA = 0.5

def create_densities_array(size = 20):
    return np.zeros(shape=(9,size,size))

def get_microscopic_velocity(densities):
    vx = np.sum(E[:,0]*densities*W)
    vy = np.sum(E[:,1]*densities*W)
    return vx, vy

def get_microscopic_density(densities):
    return np.sum(densities)

@jit
def get_equilibrium_densities(densities):
    u = get_microscopic_velocity(densities)
    rho = get_microscopic_density(densities)
    neq = np.array([rho*W[i]*(1+3*np.dot(E[i],u)+np.dot(E[i],u)**2*9/2)-np.dot(u,u)*3/2 for i in np.arange(0,len(E))])
    return neq

@jit
def move_update_densities_array(oldDensities):
    newDensities = np.zeros(shape = oldDensities.shape)
    for x in np.arange(0,oldDensities.shape[1]):
        for y in np.arange(0,oldDensities.shape[2]):
            newDensitiesOnXY = np.zeros(9)
            for i in np.arange(0,len(E)):
                xSource = x - E[i,0]
                ySource = y - E[i,1]

                if xSource >= 0 and xSource < oldDensities.shape[1]:
                    if ySource >= 0 and ySource < oldDensities.shape[2]:
                        newDensitiesOnXY[i] += oldDensities[i,xSource,ySource]
                    else: # y is out of bounds, mirroring x
                        newDensitiesOnXY[i] += oldDensities[np.where((np.array([E[i,0],-E[i,1]],dtype=int)==E).all(axis=1)),xSource,y]
                else:
                    if ySource >= 0 and ySource < oldDensities.shape[2]: # x is out of bounds, mirroring y
                        newDensitiesOnXY[i] += oldDensities[np.where((np.array([-E[i,0],E[i,1]],dtype=int)==E).all(axis=1)),x,ySource]
                    else: # x,y is out of bounds, mirroring x,y
                        newDensitiesOnXY[i] += oldDensities[np.where((np.array([-E[i,0],-E[i,1]],dtype=int)==E).all(axis=1)),x,y]  
            newDensities[:,x,y] = newDensitiesOnXY
    return newDensities

@jit
def equilibrium_update_densities_array(densitiesArrayToUpdate):
    for x in np.arange(0,densitiesArrayToUpdate.shape[1]):
        for y in np.arange(0,densitiesArrayToUpdate.shape[2]):
            densitiesArrayToUpdate[:,x,y] = densitiesArrayToUpdate[:,x,y] + OMEGA*(get_equilibrium_densities(densitiesArrayToUpdate[:,x,y]) - densitiesArrayToUpdate[:,x,y])

def main():
    cv.namedWindow('Output',cv.WINDOW_NORMAL)
    arr = create_densities_array(20)
    
    for i in np.arange(0,1000):
        if i < 50:
            arr[:,10,10] = np.array([1 for i in np.arange(0,9)])
            
        equilibrium_update_densities_array(arr)
        arr = move_update_densities_array(arr)
        #print(np.sum(arr,axis=0))
        if i%10:
            #show_array(np.sum(arr,axis=0))
            show_array(arr[0,:,:])
        print(f'frame: {i}, mass: {np.sum(arr)}', end='                               \r')
    pass

def show_array(array):
    image = np.zeros(shape=(array.shape[0],array.shape[1],3))
    image[:,:,0] = array
    cv.imshow('Output', image)
    cv.waitKey(10)




if __name__ == '__main__':
    main()