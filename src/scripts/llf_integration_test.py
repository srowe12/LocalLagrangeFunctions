import numpy as np
import scipy as sp


def BuildPolyMatrix(centers):
    num_rows = np.shape(centers)[0]
    P = np.concatenate((np.ones((num_rows,1)),centers),axis=1)
    return P

def BuildInterpolationMatrix(centers):
    distance_matrix = BuildDistanceMatrix(centers)
    interp_matrix = .5*distance_matrix*np.log(distance_matrix)
    for row in range(interp_matrix.shape()): #Change
        interp_matrix[row,row] = 0
    poly_matrix = BuildPolyMatrix(centers)
    full_matrix = np.concatenate((interp_matrix, poly_matrix),axis=1)
    num_cols = np.shape(poly_matrix)[1]
    lower_poly_matrix = np.concatenate(np.transpose(poly_matrix), np.zeros(p,p),axis=0)
    return np.concatenate( (full_matrix,lower_poly_matrix),axis=0)

def BuildLocalLagrange():
    interp_matrix = BuildInterpolationMatrix()
    rhs = BuildRHS()
    solution = np.solve(interp_matrix,rhs)
    return solution

def ReadData(iter_num,location_str):

     centers_x_str = location_str  + "centers_x.txt"
     centers_y_str = location_str + "centers_y.txt"
     centers_x = np.loadtxt(centers_x_str)
     centers_y = np.loadtxt(centers_y_str)
     centers = np.concatenate((centers_x,centers_y),axis=1)
     indices_str = location_str + "indices_" + str(iter_num) + ".txt"
     coefs_str = location_str + "coefs_" + str(iter_num) + ".txt"
     indices = np.loadtxt(inices_str)
     coefs = np.loadtxt(coefs_str)
    
