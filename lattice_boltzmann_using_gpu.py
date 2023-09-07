import cv2 as cv
import numpy as np
from numba import cuda

import timeit

from constants import *




def main():
    start_visualization()
    pass

def start_visualization():
    cv.namedWindow('Output',cv.WINDOW_NORMAL)
    X = 100
    Y = 200
    densitiesGrid = np.ones((9,X,Y))/9
    objectsGrid = np.zeros((X,Y),dtype=bool)
    add_sphere_to_array(objectsGrid, (100, 50), 5)
    add_sphere_to_array(objectsGrid, (20, 70), 7)
    add_sphere_to_array(objectsGrid, (50, 70),5)
    add_sphere_to_array(objectsGrid, (750, 50), 15)
    velocitiesOut = np.zeros((2,X,Y))
    gpuDensitiesGrid = cuda.to_device(densitiesGrid)

    blockSize = (32,32)
    gridSize = (
        int(np.ceil(densitiesGrid.shape[1]/blockSize[0])),
        int(np.ceil(densitiesGrid.shape[2]/blockSize[1]))
    )
    for i in np.arange(0,10000):
        densitiesGrid[:,:,190:Y]-=densitiesGrid[:,:,190:Y]
        if i < 2000:
            densitiesGrid[1,0:X,0] += np.array([0.1 for i in np.arange(0,X)]) # flow
            #arr[:,250,250]= np.array([0.5 for i in np.arange(0,9)]) #
        
        equilibrium_update_densities_array[gridSize,blockSize](densitiesGrid, gpuDensitiesGrid, velocitiesOut)
        densitiesGrid = gpuDensitiesGrid.copy_to_host()
        move_update_densities_array[gridSize,blockSize](densitiesGrid, gpuDensitiesGrid, objectsGrid)
        densitiesGrid = gpuDensitiesGrid.copy_to_host()
        if i%15 == 0:
            #show_array(densitiesGrid[1,:,:])
            #show_array(np.sum(densitiesGrid,axis=0))
            velNorm = np.sqrt((velocitiesOut[0,:,:]*velocitiesOut[0,:,:]+velocitiesOut[1,:,:]*velocitiesOut[1,:,:]))
            show_array(velNorm/np.max(velNorm)*6)
            print(f'frame: {i}, mass: {np.sum(densitiesGrid)}', end='                               \r')
    pass


@cuda.jit
def move_update_densities_array(densitiesGrid, movedDensitiesGrid, objectsGrid):
    x, y = cuda.grid(2)
    if x < densitiesGrid.shape[1] and y < densitiesGrid.shape[2] and not objectsGrid[x,y]:
        i = 0
        while i < 9:
            density = 0
            xSource = x - E[i,0]
            ySource = y - E[i,1]
            if xSource >= 0 and xSource < densitiesGrid.shape[1]:
                if ySource >= 0 and ySource < densitiesGrid.shape[2] and not objectsGrid[xSource,ySource]:
                    density += densitiesGrid[i,xSource,ySource]
                else: # y is out of bounds, mirroring along y axis, -x
                    density += densitiesGrid[YMIRR[i],x,y]
                    pass
            else:
                if ySource >= 0 and ySource < densitiesGrid.shape[2] and not objectsGrid[x,ySource]: # x is out of bounds, mirroring along x axis, -y
                    density += densitiesGrid[XMIRR[i],x,ySource]
                    pass
                else: # x,y is out of bounds, mirroring x,y
                    density += densitiesGrid[XYMIRR[i],xSource,y]
                    pass
                
            movedDensitiesGrid[i,x,y] = density
            i+=1

@cuda.jit
def equilibrium_update_densities_array(densitiesGrid, equalizedDensitiesGrid, velocitiesOut):
    x, y = cuda.grid(2)
    if x < densitiesGrid.shape[1] and y < densitiesGrid.shape[2]:
        ux = 0
        uy = 0
        rho = 0
        i = 0
        while i < 9:
            density = densitiesGrid[i,x,y]
            ux += E[i,0]*density*W[i]
            uy += E[i,1]*density*W[i]
            rho += density
            i+=1
        velocitiesOut[0,x,y] = ux
        velocitiesOut[1,x,y] = uy
        if rho != 0: # doesnt have to calculate empty frames
            i = 0
            normSquaredOfU = ux*ux+uy*uy
            while i < 9:
                dotOfEu = E[i,0]*ux + E[i,1]*uy
                updatedDensity = rho * W[i] * (1 + 3 * dotOfEu + dotOfEu * dotOfEu * 9 / 2 - normSquaredOfU * 3 / 2)
                equalizedDensitiesGrid[i,x,y] = densitiesGrid[i,x,y] + OMEGA*(updatedDensity - densitiesGrid[i, x, y])
                i+=1



def show_array(array):
    image = np.zeros(shape=(array.shape[0],array.shape[1],3))
    image[:,:,0] = array%1
    image[:,:,1] = array/2%1
    image[:,:,2] = array/3
    cv.imshow('Output', image)
    cv.waitKey(1)

def add_sphere_to_array(arr, center, radius):
    for x in np.arange(0,arr.shape[0]):
        for y in np.arange(0,arr.shape[0]):
            r = np.sqrt((x-center[0])**2 + (y-center[1])**2)
            if r < radius:
                arr[x,y] = True


if __name__ == '__main__':
    main()
    #print(timeit.timeit(lambda:main(),number = 1))