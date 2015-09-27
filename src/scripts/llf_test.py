import numpy as np
import scipy as sp

class LLFTest:
    def __init__(self,num):
        self.location_str = self.GetLocationStr()
        self.ReadData(num)
        
        self.num_centers = np.shape(self.centers)[0]
        self.iter_num = num

    def BuildRHS(self):
        rhs = np.zeros((self.num_centers+3,1))
        local_index = np.where(self.indices==self.iter_num)
        rhs[local_index] =1
        return rhs 

    def BuildPolyMatrix(self):
        num_rows = np.shape(self.centers)[0]
        P = np.concatenate((np.ones((num_rows,1)),self.centers),axis=1)
        return P

    def BuildDistanceMatrix(self):
        norms = np.transpose([np.sum(self.centers*self.centers,axis=1)])
        DM = norms+np.transpose(norms)-2*np.dot(self.centers,np.transpose(self.centers))
        return DM
 
    def BuildInterpolationMatrix(self):
        distance_matrix = self.BuildDistanceMatrix()
        interp_matrix = .5*distance_matrix*np.log(distance_matrix)
        np.fill_diagonal(interp_matrix,0)
        poly_matrix = self.BuildPolyMatrix()
        full_matrix = np.concatenate((interp_matrix, poly_matrix),axis=1)
        num_cols = np.shape(poly_matrix)[1]
        lower_poly_matrix = np.concatenate((np.transpose(poly_matrix), np.zeros((num_cols,num_cols))),axis=1)
        return np.concatenate( (full_matrix,lower_poly_matrix),axis=0)
    
    def BuildLocalLagrange(self):
        interp_matrix = self.BuildInterpolationMatrix()
        rhs = self.BuildRHS()
        solution = np.linalg.solve(interp_matrix,rhs)
        return solution
    
    def ReadData(self,iter_num):
         centers_x_str = self.location_str  + "centers_x.txt"
         centers_y_str = self.location_str + "centers_y.txt"
         centers_x = np.loadtxt(centers_x_str)
         centers_y = np.loadtxt(centers_y_str)
         self.full_centers = np.transpose(np.concatenate(([centers_x],[centers_y]),axis=0))
         indices_str = self.location_str + "indices_" + str(iter_num) + ".txt"
         coefs_str = self.location_str + "coefs_" + str(iter_num) + ".txt"
         self.indices = np.loadtxt(indices_str,dtype="int")
         self.coefs = np.loadtxt(coefs_str)
         self.centers = self.full_centers[self.indices,:]
 
    def GetLocationStr(self):
        return "/home/srowe/repos/LocalLagrangeFunctions/build/"

