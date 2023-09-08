import cv2 as cv
import numpy as np
from numba import cuda

import time

from constants import *




def main():
    start_visualization()
    pass

def start_visualization():
    cv.namedWindow('Output',cv.WINDOW_NORMAL)
    X = 500
    Y = 500
    densitiesGrid = np.zeros((9,X,Y),dtype=float)
    # Initial velocity - Wind tunnel
    densitiesGrid[1,:,:] = np.ones((X,Y),dtype=float)/20

    # Adds objects
    objectsGrid = np.zeros((X,Y),dtype=bool)
    add_sphere_to_array(objectsGrid, (100, 50), 5)
    add_sphere_to_array(objectsGrid, (20, 70), 7)
    add_sphere_to_array(objectsGrid, (50, 70),10)
    add_sphere_to_array(objectsGrid, (400, 100), 20)

    # Adds inputs and outputs
    flowGrid = np.zeros((9,X,Y),dtype=float)
    flowGrid[1,10:X-10,0] = np.ones(X-20)/100
    flowGrid[:,0:X,Y-1] = -np.ones((9,X))*10
    

    # Velocity array for visualization
    velocities = np.zeros((2,X,Y),dtype=float)
    velocitiesOutput = cuda.to_device(velocities)



    blockSize = (8,8)
    gridSize = (
        int(np.ceil(densitiesGrid.shape[1]/blockSize[0])),
        int(np.ceil(densitiesGrid.shape[2]/blockSize[1]))
    )

    # Send the densities grids to device
    densitiesGridInput = cuda.to_device(densitiesGrid)
    densitiesGridOutput = cuda.to_device(densitiesGrid)

    # Initialize constant memory on GPU
    objectsGridDeviceArray = cuda.to_device(objectsGrid)
    flowGridDeviceArray = cuda.to_device(flowGrid)


    startTime = time.time()
    i = 1
    ITERATIONS = 1000000
    while i < ITERATIONS:
        
        equilibrium_update_densities_array[gridSize,blockSize](densitiesGridInput,
                                                               densitiesGridOutput,
                                                               objectsGridDeviceArray,
                                                               flowGridDeviceArray,
                                                               velocitiesOutput)
        densitiesGridInput.copy_to_device(densitiesGridOutput)
        move_update_densities_array[gridSize,blockSize](densitiesGridInput,
                                                        densitiesGridOutput,
                                                        objectsGridDeviceArray,
                                                        flowGridDeviceArray)
        densitiesGridInput.copy_to_device(densitiesGridOutput)
        if i%50 == 0:
            #show_array(densitiesGrid[1,:,:])
            #show_array(np.sum(densitiesGridOutput.copy_to_host(),axis=0))

            velocitiesOutput.copy_to_host(velocities)
            
            #velNorm = np.sqrt((velocities[0,:,:]*velocities[0,:,:]+velocities[1,:,:]*velocities[1,:,:]))
            #show_array(velNorm*1000)
            show_arrays(velocities[0,:,:]*1000,velocities[1,:,:]*1000)

            elapsedTime = time.time() - startTime
            framerate = i/elapsedTime


            print(f'Frame: {i}, Elapsed time: {elapsedTime}, FPS: {framerate}, Mass: {np.sum(densitiesGridInput.copy_to_host())},', end='                                      \r')
        i += 1
    pass


@cuda.jit
def move_update_densities_array(densitiesGrid,
                                movedDensitiesGrid,
                                objectsGrid,
                                flowGrid):
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
            
            density += flowGrid[i,x,y]
            if density < 0:
                density = 0

            movedDensitiesGrid[i,x,y] = density
            i+=1

@cuda.jit
def equilibrium_update_densities_array(densitiesGrid,
                                       equalizedDensitiesGrid,
                                       objectsGrid,
                                       flowGrid,
                                       velocitiesOut):
    x, y = cuda.grid(2)
    if x < densitiesGrid.shape[1] and y < densitiesGrid.shape[2] and not objectsGrid[x,y]:
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
    image[:,:,1] = array%0.85
    image[:,:,2] = array%0.74
    cv.imshow('Output', image)
    cv.waitKey(5)
def show_arrays(array1, array2):
    image = np.zeros(shape=(array1.shape[0],array1.shape[1],3))
    image[:,:,0] = array1
    image[:,:,1] = array2

    cv.imshow('Output', image)
    cv.waitKey(5)

def add_sphere_to_array(arr, center, radius):
    for x in np.arange(0,arr.shape[0]):
        for y in np.arange(0,arr.shape[0]):
            r = np.sqrt((x-center[0])**2 + (y-center[1])**2)
            if r < radius:
                arr[x,y] = True


if __name__ == '__main__':
    main()
    #print(timeit.timeit(lambda:main(),number = 1))