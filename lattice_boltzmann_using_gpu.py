import cv2 as cv
import numpy as np
from numba import cuda

import time

from constants import *

import threading as thr

import math




def main():
    start_visualization()
    pass

def start_visualization():
    cv.namedWindow('Output',cv.WINDOW_NORMAL)
    X = 300
    Y = 500
    densitiesGrid = np.zeros((9,X,Y),dtype=float)
    # Initial velocity - Wind tunnel
    densitiesGrid[1,:,:] = np.ones((X,Y),dtype=float)

    # Adds objects
    objectsGrid = np.zeros((X,Y),dtype=bool)
    #add_circle_to_array(objectsGrid, (100, 50), 5)
    #add_circle_to_array(objectsGrid, (20, 70), 7)
    #add_circle_to_array(objectsGrid, (50, 70),10)
    #add_circle_to_array(objectsGrid, (400, 100), 20)

    # Adds inputs and outputs
    flowGrid = np.zeros((9,X,Y),dtype=float)
    #add_circle_to_array(flowGrid[0,:,:], (150, 150), 20)
    #flowGrid/=50
    flowGrid[1,0:X,0] = np.ones(X)/20
    flowGrid[:,0:X,Y-1] = -np.ones((9,X))*100

    
    

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

    # TODO: find a better way for drawing
    global mouseDown
    mouseDown = False
    def mouse_callback(e, x, y, p1, p2, og, ogdevice):
        global mouseDown
        if e==cv.EVENT_LBUTTONDOWN or e==cv.EVENT_LBUTTONUP:
            mouseDown = not mouseDown
        if mouseDown:
            objectsGridDeviceArray.copy_to_device(add_circle_to_array(objectsGrid,(y,x),3))
    cv.setMouseCallback('Output', lambda e,x,y,p1,p2,og = objectsGrid, ogdevice = objectsGridDeviceArray: mouse_callback(e, x, y, p1, p2,og, ogdevice))

    i = 1
    startTime = time.time()

    
    ITERATIONS = 1000000
    while i < ITERATIONS:
        
        equilibrium_update_densities_array[gridSize,blockSize](densitiesGridInput,
                                                               densitiesGridOutput,
                                                               objectsGridDeviceArray,
                                                               flowGridDeviceArray,
                                                               velocitiesOutput)
        move_update_densities_array[gridSize,blockSize](densitiesGridOutput,
                                                        densitiesGridInput,
                                                        objectsGridDeviceArray,
                                                        flowGridDeviceArray,
                                                        i)
        if i%100 == 0:
            output_update(velocities,velocitiesOutput,densitiesGridInput,startTime, i)
        i += 1



def output_update(velocities, velocitiesOutput, densitiesGridInput, startTime, iterationIndex):
    velocitiesOutput.copy_to_host(velocities)
            
    velNorm = np.sqrt((velocities[0,:,:]*velocities[0,:,:]+velocities[1,:,:]*velocities[1,:,:]))
    show_array(velNorm*100,colormap='cyclic')
    #vx:np.ndarray = velocities[0,:,:]
    #vy = velocities[1,:,:]
    #show_arrays_directional(vx*200 +0.5, vy*200 + 0.5, vy*100+vx*100 + 0.5)
    elapsedTime = time.time() - startTime
    framerate = iterationIndex/elapsedTime
    print(f'Frame: {iterationIndex}, Elapsed time: {elapsedTime}, FPS: {framerate}, Mass: {np.sum(densitiesGridInput.copy_to_host())},', end='                                      \r')



@cuda.jit
def move_update_densities_array(densitiesGrid,
                                movedDensitiesGrid,
                                objectsGrid,
                                flowGrid,
                                iteration):
    
    x, y = cuda.grid(2)
    if x < densitiesGrid.shape[1] and y < densitiesGrid.shape[2]:
        i = 0
        while i < 9:
            density = 0
            xSource = x - E[i,0]
            ySource = y - E[i,1]

            if xSource >= 0 and xSource < densitiesGrid.shape[1]:
                # Is safely in bounds
                if ySource >= 0 and ySource < densitiesGrid.shape[2]:
                    if not objectsGrid[xSource,ySource]: # There is not object
                        density += densitiesGrid[i,xSource,ySource]
                    else: # There is an object
                        if xSource == x or ySource == y: # The object is edge neigbour (One of the coordinates is 0)
                            density += densitiesGrid[XYMIRR[i],x,y]
                        else: # The object is a corner neighbour
                            if objectsGrid[xSource,y]: # There is a Y wall or a corner
                                if objectsGrid[x,ySource]: # There is a corner
                                    density += densitiesGrid[XYMIRR[i],x,y]
                                else: # There is a Y wall
                                    density += densitiesGrid[YMIRR[i],x,ySource]
                            elif objectsGrid[x,ySource]: # There is an X wall (corner was discussed in previous if)
                                density += densitiesGrid[XMIRR[i],xSource,y]
                            else: # The density hits a corner (TODO: does not bounce back but splits to sides instead)
                                density += densitiesGrid[XYMIRR[i],x,y]
                # Is on Y edge
                elif not objectsGrid[xSource,y]: # y is out of bounds, mirroring along y axis, -x
                    density += densitiesGrid[YMIRR[i],xSource,y]
            else:
                # Is on X edge
                if ySource >= 0 and ySource < densitiesGrid.shape[2] and not objectsGrid[x,ySource]: # x is out of bounds, mirroring along x axis, -y
                    density += densitiesGrid[XMIRR[i],x,ySource]
                # Is in corner and is looking at 
                else: # x,y is out of bounds, mirroring x,y
                    density += densitiesGrid[XYMIRR[i],x,y]
            density += flowGrid[i,x,y]
            if density < 0:
                density = 0

            if objectsGrid[x,y]:
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



def show_array(array, colormap = 'normal'):
    image = np.zeros(shape=(array.shape[0],array.shape[1],3))
    match colormap:
        case 'normal':
            image[:,:,0] = array
            image[:,:,1] = array
            image[:,:,2] = array
        case 'cyclic':
            xScale = 1
            yScale = 0.85
            zScale = 0.74
            image[:,:,0] = np.sin(math.pi*(array%xScale))
            image[:,:,1] = np.sin(math.pi*(array%yScale))
            image[:,:,2] = np.sin(math.pi*(array%zScale))
        
    cv.imshow('Output', image)
    cv.waitKey(5)

def show_arrays(array1, array2):
    image = np.zeros(shape=(array1.shape[0],array1.shape[1],3))
    image[:,:,0] = array1
    image[:,:,1] = array2

    cv.imshow('Output', image)
    cv.waitKey(5)

def show_arrays_directional(array1, array2, array3):
    image = np.zeros(shape=(array1.shape[0],array1.shape[1],3))
    image[:,:,0] = array1
    image[:,:,1] = array2
    image[:,:,2] = array3

    cv.imshow('Output', image)
    cv.waitKey(5)

def add_circle_to_array(arr:np.ndarray, center, radius, value = 1, relative = False):
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
                arr[x,y] = True
            elif arr.dtype==float and r < radius:
                if relative:
                    arr[x,y] += value
                else:
                    arr[x,y] = value
    return arr






if __name__ == '__main__':
    main()
    #print(timeit.timeit(lambda:main(),number = 1))