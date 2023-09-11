import numpy as np

from modules.useful_functions import add_circle_to_array

def get_initial_conditions(name, X, Y):
    match name:
        case 'wind tunnel':
            # Initial velocity - Wind tunnel
            densitiesGrid = np.zeros((9,X,Y),dtype=float)
            densitiesGrid[1,:,:] = np.ones((X,Y),dtype=float)

            # Adds objects grid
            objectsGrid = np.zeros((X,Y),dtype=bool)

            # Adds inputs and outputs
            flowGrid = np.zeros((9,X,Y),dtype=float)
            flowGrid/=50
            flowGrid[1,0:X,0] = np.ones(X)/20
            flowGrid[:,0:X,Y-1] = -np.ones((9,X))*100
            return densitiesGrid, objectsGrid, flowGrid
        case 'central input':
            # Initial velocity - Wind tunnel
            densitiesGrid = np.zeros((9,X,Y),dtype=float)
            densitiesGrid[1,:,:] = np.ones((X,Y),dtype=float)

            # Adds objects grid
            objectsGrid = np.zeros((X,Y),dtype=bool)

            # Adds inputs and outputs
            flowGrid = np.zeros((9,X,Y),dtype=float)
            add_circle_to_array(flowGrid[0,:,:], (int(np.floor(X/2)), int(np.floor(Y/2))), 20)
            flowGrid/=50
            flowGrid[:,0:X,0] = -np.ones((9,X))*100
            flowGrid[:,0:X,Y-1] = -np.ones((9,X))*100
            flowGrid[:,0,0:Y] = -np.ones((9,Y))*100
            flowGrid[:,X-1,0:Y] = -np.ones((9,Y))*100
            return densitiesGrid, objectsGrid, flowGrid

    