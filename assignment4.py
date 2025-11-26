"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""

import numpy as np
import time
import random
import math



class Assignment4:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """
        pass


    def sample_points(self, target_function, start_range, end_range, num_sample_points):
        """
        Create a matrix of sample points from the target function.
        
        Parameters
        ----------
        target_function : callable
            The function to sample from.
        start_range : float
            Start of the fitting range.
        end_range : float
            End of the fitting range.
        num_sample_points : int
            Number of points to sample.
            
        Returns
        -------
        np.ndarray
            A 2D array of shape (num_sample_points, 2) representing 
            (x, y) points from the function.
        """
        x_coordinates = np.linspace(start_range, end_range, num_sample_points)
        point_matrix = np.array([[x, target_function(x)] for x in x_coordinates])
        return point_matrix

    def build_parameter_matrix(self, points, start_range, end_range, polynomial_degree):
        """
        Build the parameter transformation matrix for Bezier curve fitting.
        
        Parameters
        ----------
        points : np.ndarray
            The matrix of sample points.
        start_range : float
            Start of the fitting range.
        end_range : float
            End of the fitting range.
        polynomial_degree : int
            Desired degree of the polynomial.
            
        Returns
        -------
        np.ndarray
            A matrix T where each row contains normalized parameter values 
            raised to powers from polynomial_degree down to 0.
        """
        transformation_matrix = []
        for point in points:
            normalized_parameter = (point[0] - start_range) / (end_range - start_range)
            parameter_powers = [normalized_parameter ** (polynomial_degree - power_index) 
                               for power_index in range(0, polynomial_degree + 1)]
            transformation_matrix.append(parameter_powers)

        transformation_matrix = np.array(transformation_matrix)
        return transformation_matrix

    def compute_bezier_matrix(self, polynomial_degree):
        """
        Compute the Bezier transformation matrix for the given polynomial degree.
        
        Parameters
        ----------
        polynomial_degree : int
            Degree of the polynomial (must be between 0 and 12).
            
        Returns
        -------
        np.ndarray
            The Bezier transformation matrix M for the specified degree.
        """
        if polynomial_degree == 0:
            bezier_matrix_for_degree = np.array([[1]])
        elif polynomial_degree == 1:
            bezier_matrix_for_degree = np.array([[-1, 1], [1, 0]])
        elif polynomial_degree == 2:
            bezier_matrix_for_degree = np.array([[1.0, -2.0, 1.0], [-2.0, 2.0, 0.0], [1.0, 0.0, 0.0]])
        elif polynomial_degree == 3:
            bezier_matrix_for_degree = np.array(
                [[-1.0, 3.0, -3.0, 1.0], [3.0, -6.0, 3.0, 0.0], [-3.0, 3.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        elif polynomial_degree == 4:
            bezier_matrix_for_degree = np.array([[1.0, -4.0, 6.0, -4.0, 1.0], [-4.0, 12.0, -12.0, 4.0, 0.0], [6.0, -12.0, 6.0, 0.0, 0.0],
                               [-4.0, 4.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]])
        elif polynomial_degree == 5:
            bezier_matrix_for_degree = np.array([[-1.0, 5.0, -10.0, 10.0, -5.0, 1.0], [5.0, -20.0, 30.0, -20.0, 5.0, 0.0],
                               [-10.0, 30.0, -30.0, 10.0, 0.0, 0.0], [10.0, -20.0, 10.0, 0.0, 0.0, 0.0],
                               [-5.0, 5.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        elif polynomial_degree == 6:
            bezier_matrix_for_degree = np.array([[1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0], [-6.0, 30.0, -60.0, 60.0, -30.0, 6.0, 0.0],
                               [15.0, -60.0, 90.0, -60.0, 15.0, 0.0, 0.0], [-20.0, 60.0, -60.0, 20.0, 0.0, 0.0, 0.0],
                               [15.0, -30.0, 15.0, 0.0, 0.0, 0.0, 0.0], [-6.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        elif polynomial_degree == 7:
            bezier_matrix_for_degree = np.array(
                [[-1.0, 7.0, -21.0, 35.0, -35.0, 21.0, -7.0, 1.0], [7.0, -42.0, 105.0, -140.0, 105.0, -42.0, 7.0, 0.0],
                 [-21.0, 105.0, -210.0, 210.0, -105.0, 21.0, 0.0, 0.0],
                 [35.0, -140.0, 210.0, -140.0, 35.0, 0.0, 0.0, 0.0], [-35.0, 105.0, -105.0, 35.0, 0.0, 0.0, 0.0, 0.0],
                 [21.0, -42.0, 21.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-7.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        elif polynomial_degree == 8:
            bezier_matrix_for_degree = np.array([[1.0, -8.0, 28.0, -56.0, 70.0, -56.0, 28.0, -8.0, 1.0],
                               [-8.0, 56.0, -168.0, 280.0, -280.0, 168.0, -56.0, 8.0, 0.0],
                               [28.0, -168.0, 420.0, -560.0, 420.0, -168.0, 28.0, 0.0, 0.0],
                               [-56.0, 280.0, -560.0, 560.0, -280.0, 56.0, 0.0, 0.0, 0.0],
                               [70.0, -280.0, 420.0, -280.0, 70.0, 0.0, 0.0, 0.0, 0.0],
                               [-56.0, 168.0, -168.0, 56.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [28.0, -56.0, 28.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [-8.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        elif polynomial_degree == 9:
            bezier_matrix_for_degree = np.array([[-1.0, 9.0, -36.0, 84.0, -126.0, 126.0, -84.0, 36.0, -9.0, 1.0],
                               [9.0, -72.0, 252.0, -504.0, 630.0, -504.0, 252.0, -72.0, 9.0, 0.0],
                               [-36.0, 252.0, -756.0, 1260.0, -1260.0, 756.0, -252.0, 36.0, 0.0, 0.0],
                               [84.0, -504.0, 1260.0, -1680.0, 1260.0, -504.0, 84.0, 0.0, 0.0, 0.0],
                               [-126.0, 630.0, -1260.0, 1260.0, -630.0, 126.0, 0.0, 0.0, 0.0, 0.0],
                               [126.0, -504.0, 756.0, -504.0, 126.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [-84.0, 252.0, -252.0, 84.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [36.0, -72.0, 36.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [-9.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        elif polynomial_degree == 10:
            bezier_matrix_for_degree = np.array([[1.0, -10.0, 45.0, -120.0, 210.0, -252.0, 210.0, -120.0, 45.0, -10.0, 1.0],
                               [-10.0, 90.0, -360.0, 840.0, -1260.0, 1260.0, -840.0, 360.0, -90.0, 10.0, 0.0],
                               [45.0, -360.0, 1260.0, -2520.0, 3150.0, -2520.0, 1260.0, -360.0, 45.0, 0.0, 0.0],
                               [-120.0, 840.0, -2520.0, 4200.0, -4200.0, 2520.0, -840.0, 120.0, 0.0, 0.0, 0.0],
                               [210.0, -1260.0, 3150.0, -4200.0, 3150.0, -1260.0, 210.0, 0.0, 0.0, 0.0, 0.0],
                               [-252.0, 1260.0, -2520.0, 2520.0, -1260.0, 252.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [210.0, -840.0, 1260.0, -840.0, 210.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [-120.0, 360.0, -360.0, 120.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [45.0, -90.0, 45.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [-10.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        elif polynomial_degree == 11:
            bezier_matrix_for_degree = np.array([[-1.0, 11.0, -55.0, 165.0, -330.0, 462.0, -462.0, 330.0, -165.0, 55.0, -11.0, 1.0],
                               [11.0, -110.0, 495.0, -1320.0, 2310.0, -2772.0, 2310.0, -1320.0, 495.0, -110.0, 11.0,
                                0.0],
                               [-55.0, 495.0, -1980.0, 4620.0, -6930.0, 6930.0, -4620.0, 1980.0, -495.0, 55.0, 0.0,
                                0.0],
                               [165.0, -1320.0, 4620.0, -9240.0, 11550.0, -9240.0, 4620.0, -1320.0, 165.0, 0.0, 0.0,
                                0.0],
                               [-330.0, 2310.0, -6930.0, 11550.0, -11550.0, 6930.0, -2310.0, 330.0, 0.0, 0.0, 0.0, 0.0],
                               [462.0, -2772.0, 6930.0, -9240.0, 6930.0, -2772.0, 462.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [-462.0, 2310.0, -4620.0, 4620.0, -2310.0, 462.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [330.0, -1320.0, 1980.0, -1320.0, 330.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [-165.0, 495.0, -495.0, 165.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [55.0, -110.0, 55.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [-11.0, 11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        else:
            bezier_matrix_for_degree = np.array(
                [[1.0, -12.0, 66.0, -220.0, 495.0, -792.0, 924.0, -792.0, 495.0, -220.0, 66.0, -12.0, 1.0],
                 [-12.0, 132.0, -660.0, 1980.0, -3960.0, 5544.0, -5544.0, 3960.0, -1980.0, 660.0, -132.0, 12.0, 0.0],
                 [66.0, -660.0, 2970.0, -7920.0, 13860.0, -16632.0, 13860.0, -7920.0, 2970.0, -660.0, 66.0, 0.0, 0.0],
                 [-220.0, 1980.0, -7920.0, 18480.0, -27720.0, 27720.0, -18480.0, 7920.0, -1980.0, 220.0, 0.0, 0.0, 0.0],
                 [495.0, -3960.0, 13860.0, -27720.0, 34650.0, -27720.0, 13860.0, -3960.0, 495.0, 0.0, 0.0, 0.0, 0.0],
                 [-792.0, 5544.0, -16632.0, 27720.0, -27720.0, 16632.0, -5544.0, 792.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [924.0, -5544.0, 13860.0, -18480.0, 13860.0, -5544.0, 924.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [-792.0, 3960.0, -7920.0, 7920.0, -3960.0, 792.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [495.0, -1980.0, 2970.0, -1980.0, 495.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [-220.0, 660.0, -660.0, 220.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [66.0, -132.0, 66.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [-12.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        return bezier_matrix_for_degree


    def bezier_interpolation(self, control_points, start_range, end_range):
        """
        Interpolate a set of control points using the Bezier curve algorithm.
        
        Parameters
        ----------
        control_points : list of tuples
            A list of tuples, each tuple representing a control point in the format (x, y).
        start_range : float
            The start of the sampling range.
        end_range : float
            The end of the sampling range.
            
        Returns
        -------
        callable
            A function that takes an x value and returns the interpolated y value 
            using Bezier curve interpolation.
        """
        num_control_points = len(control_points) - 1
        x_coordinates, y_coordinates = zip(*control_points)

        def bernstein_polynomial(control_point_index, degree, parameter_value):
            """
            Compute the Bernstein polynomial basis function.
            
            Parameters
            ----------
            control_point_index : int
                Index of the control point.
            degree : int
                Degree of the Bezier curve.
            parameter_value : float
                Parameter value in [0, 1].
                
            Returns
            -------
            float
                Value of the Bernstein polynomial basis function.
            """
            return math.comb(degree, control_point_index) * (parameter_value ** control_point_index) * ((1 - parameter_value) ** (degree - control_point_index))

        def bezier_curve(parameter_value):
            """
            Evaluate the Bezier curve at the given parameter value.
            
            Parameters
            ----------
            parameter_value : float
                Parameter value in [0, 1].
                
            Returns
            -------
            float
                The y value of the Bezier curve at the parameter value.
            """
            return sum(bernstein_polynomial(i, num_control_points, parameter_value) * y_coordinates[i] 
                      for i in range(num_control_points + 1))

        return lambda x: bezier_curve((x - start_range) / (end_range - start_range))



    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        f : callable
            A function which returns an approximate (noisy) Y value given X.
        a : float
            Start of the fitting range.
        b : float
            End of the fitting range.
        d : int
            The expected degree of a polynomial matching f.
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        callable
            A function that takes a float x and returns a float y that fits 
            f between a and b.
        """
        num_sample_points = 80
        if b == a:
            constant_value = f(a)
            return lambda x: constant_value

        if d > 12:
            d = 12
        elif d < 0:
            d = 1
        sample_points_matrix = self.sample_points(f, a, b, num_sample_points)
        parameter_matrix = self.build_parameter_matrix(sample_points_matrix, a, b, d)
        bezier_matrix = self.compute_bezier_matrix(d)

        control_points = np.linalg.inv(bezier_matrix).dot(
            (np.linalg.inv(parameter_matrix.transpose().dot(parameter_matrix)))
        ).dot(parameter_matrix.transpose()).dot(sample_points_matrix)

        fitted_bezier_function = self.bezier_interpolation(control_points, a, b)

        return fitted_bezier_function


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    # def test_return(self):
    #     f = NOISY(0.01)(poly(1,1,1))
    #     ass4 = Assignment4()
    #     T = time.time()
    #     shape = ass4.fit(f=f, a=0, b=1, d=0, maxtime=5)
    #     T = time.time() - T
    #     self.assertLessEqual(T, 5)

    # def test_delay(self):
    #     f = DELAYED(0)(NOISY(0.01)(poly(1,1,1)))
    #
    #     ass4 = Assignment4()
    #     T = time.time()
    #     shape = ass4.fit(f=f, a=0, b=1, d=1, maxtime=5)
    #
    #     T = time.time() - T
    #     self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1,1,1)
        # f = (lambda x: 6)
        nf =(NOISY(1)(f))
        ass4 = Assignment4()
        T = time.time()
        # for i in range(0,15):
        #     print(i)
        ff = ass4.fit(f=nf, a=0, b=1, d=3, maxtime=5)
        # print()
        T = time.time() - T
        mse=0
        for x in np.linspace(0,1,1000):            
            self.assertNotEqual(f(x), nf(x))
            mse+= (f(x)-ff(x))**2
        mse = mse/1000
        print("mse d=0", mse)

        
        



if __name__ == "__main__":
    unittest.main()
