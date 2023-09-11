# External libraries
import cv2 as cv
import numpy as np
from numba import cuda

import time

# External data types
from numba.cuda.cudadrv.devicearray import DeviceNDArray

# Internal libraries
from constants import *

from useful_functions import add_circle_to_array
from useful_functions import show_array, show_two_arrays, show_three_arrays
from initial_conditions import get_initial_conditions




def main():
    start_visualization()
    pass

def start_visualization(windowName = 'Output', initialConditionsName = 'wind tunnel', xSize = 300, ySize = 300):
    # Create output window
    cv.namedWindow(windowName,cv.WINDOW_NORMAL)
    
    densitiesGrid, objectsGrid, flowGrid = get_initial_conditions(initialConditionsName,xSize,ySize)

    # Velocity array for visualization
    velocities = np.zeros((2,xSize,ySize),dtype=float)
    velocitiesOutput = cuda.to_device(velocities)

    # Set block sizes for the GPU
    blockSize = (8,8)
    gridSize = (
        int(np.ceil(densitiesGrid.shape[1]/blockSize[0])),
        int(np.ceil(densitiesGrid.shape[2]/blockSize[1]))
    )

    # Send the densities grids to device
    # alternating left and right because the main loop has 2 update functions (equilibrium update and move update)
    # more efficient than copying two times during one cycle
    densitiesGridAlternatingLeft = cuda.to_device(densitiesGrid)
    densitiesGridAlternatingRight = cuda.to_device(densitiesGrid)

    # Initialize constant memory on GPU
    objectsGridDeviceArray = cuda.to_device(objectsGrid)
    flowGridDeviceArray = cuda.to_device(flowGrid)


    # Set user input methods used on runtime
    global mouseDownL, mouseDownR, brushSize
    mouseDownR = False
    mouseDownL = False
    brushSize = 3
    def mouse_callback(e, x, y, flags, param, og:np.ndarray, ogdevice:DeviceNDArray):
        global brushSize
        # Changing brush size with scrolling
        if e == cv.EVENT_MOUSEWHEEL:
            if cv.EVENT_MOUSEWHEEL > 0:
                brushSize += 1
            elif cv.EVENT_MOUSEWHEEL < 0:
                brushSize -=1
            if brushSize < 1:
                brushSize = 1

        # Left mouse click for creating objects
        global mouseDownL
        if e==cv.EVENT_LBUTTONDOWN or e==cv.EVENT_LBUTTONUP:
            mouseDownL = not mouseDownL
        if mouseDownL:
            objectsGridDeviceArray.copy_to_device(add_circle_to_array(objectsGrid,(y,x),brushSize))

        # Right mouse click for erasing objects
        global mouseDownR
        if e==cv.EVENT_RBUTTONDOWN or e==cv.EVENT_RBUTTONUP:
            mouseDownR = not mouseDownR
        if mouseDownR:
            objectsGridDeviceArray.copy_to_device(add_circle_to_array(objectsGrid,(y,x),brushSize,value=False))
    cv.setMouseCallback('Output', lambda e,x,y,flags,param,og = objectsGrid, ogdevice = objectsGridDeviceArray: mouse_callback(e, x, y, flags, param,og, ogdevice))

    # Main loop
    i = 1
    startTime = time.time()
    ITERATIONS = 1000000
    while i < ITERATIONS:
        # Recalculate densities
        # Left -> Right
        equilibrium_update_densities[gridSize,blockSize](densitiesGridAlternatingLeft,
                                                         densitiesGridAlternatingRight,
                                                         objectsGridDeviceArray,
                                                         flowGridDeviceArray,
                                                         velocitiesOutput)
        # Move densities
        # Right -> Left
        move_update_densities[gridSize,blockSize](densitiesGridAlternatingRight,
                                                  densitiesGridAlternatingLeft,
                                                  objectsGridDeviceArray,
                                                  flowGridDeviceArray)
        # Call update output every 100 frames
        if i%100 == 0:
            update_output(windowName, velocitiesOutput.copy_to_host(), densitiesGridAlternatingLeft.copy_to_host(),startTime, i)
        i += 1
        

def update_output(windowName:str,
                  velocities:np.ndarray,
                  densities:np.ndarray,
                  startTime:float,
                  iterationIndex:int,
                  outputType = 'default'):
    """
    Updates image and console output.
    """
    # Image output
    match outputType:
        case 'default' | 'velocity norm':
            velNorm = np.sqrt((velocities[0,:,:]*velocities[0,:,:]+velocities[1,:,:]*velocities[1,:,:]))
            show_array(windowName, velNorm*100,colormap='cyclic')
            cv.waitKey(1)
        case 'with direction':
            vx:np.ndarray = velocities[0,:,:]
            vy:np.ndarray = velocities[1,:,:]
            show_three_arrays(windowName, vx*200 +0.5, vy*200 + 0.5, vy*100+vx*100 + 0.5)
            cv.waitKey(1)
        case 'two outputs':
            velNorm = np.sqrt((velocities[0,:,:]*velocities[0,:,:]+velocities[1,:,:]*velocities[1,:,:]))
            show_array(windowName, velNorm*100,colormap='cyclic')
            
            vx:np.ndarray = velocities[0,:,:]
            vy:np.ndarray = velocities[1,:,:]
            show_three_arrays(windowName + '_direction', vx*200 +0.5, vy*200 + 0.5, vy*100+vx*100 + 0.5)
            cv.waitKey(1)

    # Console output
    elapsedTime = time.time() - startTime
    framerate = iterationIndex/elapsedTime
    print(f'Frame: {iterationIndex}, Elapsed time: {elapsedTime}, FPS: {framerate}, Mass: {np.sum(densities)},', end='                                      \r')



@cuda.jit# ('void(float32[:,:,:],float32[:,:,:],bool[:,:],float32[:,:,:])')
def move_update_densities(inputDensitiesGrid:DeviceNDArray,
                                outputDensitiesGrid:DeviceNDArray,
                                objectsGrid:DeviceNDArray,
                                flowGrid:DeviceNDArray):
    """
    Key arguments:
    inputDensitiesGrid - type float, shape (9,X,Y)
    outputDensitiesGrid - type float, shape (9,X,Y)
    objectsGrid - type bool, shape (X,Y)
    flowGrid - type float, shape (9,X,Y)
    """
    x, y = cuda.grid(2)
    # For each cell in grid
    if x < inputDensitiesGrid.shape[1] and y < inputDensitiesGrid.shape[2]:
        i = 0
        # For each density in each cell
        while i < 9:
            density = 0
            xSource = x - E[i,0]
            ySource = y - E[i,1]

            if xSource >= 0 and xSource < inputDensitiesGrid.shape[1]:
                # Is safely in bounds
                if ySource >= 0 and ySource < inputDensitiesGrid.shape[2]:
                    if not objectsGrid[xSource,ySource]: # There is not object
                        density += inputDensitiesGrid[i,xSource,ySource]
                    else: # There is an object
                        if xSource == x or ySource == y: # The object is edge neigbour (One of the coordinates is 0)
                            density += inputDensitiesGrid[XYMIRR[i],x,y]
                        else: # The object is a corner neighbour
                            if objectsGrid[xSource,y]: # There is a Y wall or a corner
                                if objectsGrid[x,ySource]: # There is a corner
                                    density += inputDensitiesGrid[XYMIRR[i],x,y]
                                else: # There is a Y wall
                                    density += inputDensitiesGrid[YMIRR[i],x,ySource]
                            elif objectsGrid[x,ySource]: # There is an X wall (corner was discussed in previous if)
                                density += inputDensitiesGrid[XMIRR[i],xSource,y]
                            else: # The density hits a corner (TODO: does not bounce back but splits to sides instead)
                                density += inputDensitiesGrid[XYMIRR[i],x,y]
                # Is on Y edge
                elif not objectsGrid[xSource,y]: # y is out of bounds, mirroring along y axis, -x
                    density += inputDensitiesGrid[YMIRR[i],xSource,y]
            else:
                # Is on X edge
                if ySource >= 0 and ySource < inputDensitiesGrid.shape[2] and not objectsGrid[x,ySource]: # x is out of bounds, mirroring along x axis, -y
                    density += inputDensitiesGrid[XMIRR[i],x,ySource]
                # Is in corner and is looking at 
                else: # x,y is out of bounds, mirroring x,y
                    density += inputDensitiesGrid[XYMIRR[i],x,y]
            # Take in effect outer flow of fluid (inputs and outputs)
            density += flowGrid[i,x,y]
            if density < 0:
                density = 0

            # Remove fluid from object cells
            if objectsGrid[x,y]:
                density = 0
            
            outputDensitiesGrid[i,x,y] = density
            i+=1

@cuda.jit# ('void(float32[:,:,:],float32[:,:,:],bool[:,:],float32[:,:,:],float32[:,:,:])') TODO
def equilibrium_update_densities(inputDensitiesGrid:DeviceNDArray,
                                       outputDensitiesGrid:DeviceNDArray,
                                       objectsGrid:DeviceNDArray,
                                       flowGrid:DeviceNDArray,
                                       velocitiesOut:DeviceNDArray):
    """
    Key arguments:
    inputDensitiesGrid - type float, shape (9,X,Y)
    equalizedDensitiesGrid - type float, shape (9,X,Y)
    objectsGrid - type bool, shape (X,Y)
    flowGrid - type float, shape (9,X,Y)
    velocitiesOut - type float, shape (2,X,Y)
    """
    x, y = cuda.grid(2)
    # For each cell in grid
    if x < inputDensitiesGrid.shape[1] and y < inputDensitiesGrid.shape[2]:
        # Calculate the macroscopic velocity (ux,uy), and density rho
        ux = 0
        uy = 0
        rho = 0
        i = 0
        while i < 9:
            density = inputDensitiesGrid[i,x,y]
            if density < 0:
                density = 0
            ux += E[i,0]*density*W[i]
            uy += E[i,1]*density*W[i]
            rho += density
            i+=1
        # Save the velocities
        velocitiesOut[0,x,y] = ux
        velocitiesOut[1,x,y] = uy

        # Update densities
        i = 0
        normSquaredOfU = ux*ux+uy*uy
        while i < 9:
            dotOfEu = E[i,0]*ux + E[i,1]*uy
            updatedDensity = rho * W[i] * (1 + 3 * dotOfEu + dotOfEu * dotOfEu * 9 / 2 - normSquaredOfU * 3 / 2)
            outputDensitiesGrid[i,x,y] = inputDensitiesGrid[i,x,y] + OMEGA*(updatedDensity - inputDensitiesGrid[i, x, y])
            i+=1

if __name__ == '__main__':
    # TODO argument parser
    main()
