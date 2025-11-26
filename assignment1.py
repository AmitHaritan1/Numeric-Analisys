"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random
import operator
from functools import reduce


class Assignment1:
    def __init__(self):
        """
        Initialize the interpolation class with storage for interpolation points.
        
        Here goes any one time calculation that need to be made before
        starting to interpolate arbitrary functions.
        """
        self.x_values = []
        self.y_values = []

    def find_closest_index(self, sorted_array, target_value):
        """
        Find the index of the value in the sorted array that is closest to the target value.
        Uses binary search to efficiently locate the approximate position.
        
        Parameters
        ----------
        sorted_array : array-like
            A sorted array of values.
        target_value : float
            The value to find the closest match for in the array.
            
        Returns
        -------
        int
            Index in sorted_array of the value closest to target_value.
            Returns 0 if target_value is less than the first element.
            Returns -1 if target_value is greater than the last element.
        """
        low_index = 0
        high_index = len(sorted_array) - 1
        mid_index = 0

        while low_index <= high_index:
            mid_index = (high_index + low_index) // 2
            if sorted_array[mid_index] < target_value:
                low_index = mid_index + 1
            elif sorted_array[mid_index] > target_value:
                high_index = mid_index - 1
            else:
                return mid_index

        if low_index > 0 and high_index < len(sorted_array) - 1:
            return min(low_index, high_index)
        elif target_value < sorted_array[0]:
            return 0
        elif target_value > sorted_array[-1]:
            return -1



    def build_lagrange_interpolator(self, start_index, end_index) -> callable:
        """
        Build a Lagrange interpolation polynomial for the specified range of points.
        
        Parameters
        ----------
        start_index : int
            Starting index in the x_values and y_values arrays.
        end_index : int
            Ending index (exclusive) in the x_values and y_values arrays.
            
        Returns
        -------
        callable
            A function that takes an x value and returns the interpolated y value
            using Lagrange interpolation over the specified range of points.
        """
        def lagrange_interpolator(x):
            """
            Evaluate the Lagrange interpolation polynomial at point x.
            
            Parameters
            ----------
            x : float
                The x value at which to evaluate the interpolated function.
                
            Returns
            -------
            float
                The interpolated y value at x.
            """
            def lagrange_basis_polynomial(point_index):
                """
                Compute the Lagrange basis polynomial L_i(x) for point at point_index.
                
                Parameters
                ----------
                point_index : int
                    Index of the point for which to compute the basis polynomial.
                    
                Returns
                -------
                float
                    The value of the basis polynomial L_i(x).
                """
                basis_terms = [
                    (x - self.x_values[other_index]) / (self.x_values[point_index] - self.x_values[other_index])
                    for other_index in range(start_index, end_index) 
                    if other_index != point_index
                ]
                if len(basis_terms) < 1:
                    return 1
                return reduce(operator.mul, basis_terms)

            return sum(
                lagrange_basis_polynomial(i) * self.y_values[i] 
                for i in range(start_index, end_index)
            )

        return lagrange_interpolator

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time.
        The assignment will be tested on variety of different functions with
        large n values.

        Interpolation error will be measured as the average absolute error at
        2*n random points between a and b. See test_with_poly() below.

        Note: It is forbidden to call f more than n times.

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.**

        Note: sometimes you can get very accurate solutions with only few points,
        significantly less than n.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """



        local_interpolation_degree = 5
        x_values = np.linspace(a, b, n)
        y_values = [f(x) for x in x_values]
        self.x_values = x_values
        self.y_values = y_values

        total_points = n
        interpolator_cache = [0 for _ in range(n)]

        def interpolated_function(x):
            """
            Interpolate the function at point x using cached local Lagrange interpolators.
            
            Parameters
            ----------
            x : float
                The x value at which to interpolate.
                
            Returns
            -------
            float
                The interpolated y value at x.
            """
            closest_index = self.find_closest_index(x_values, x)

            if interpolator_cache[closest_index] != 0:
                return interpolator_cache[closest_index](x)

            start_index = closest_index - local_interpolation_degree if closest_index >= local_interpolation_degree else 0
            end_index = closest_index + local_interpolation_degree if closest_index + local_interpolation_degree < total_points else total_points

            local_interpolator = self.build_lagrange_interpolator(start_index, end_index)
            interpolator_cache[closest_index] = local_interpolator
            return local_interpolator(x)

        return interpolated_function



##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200) * 20 - 10
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    #
    def test_with_poly2(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 1)

            xs = np.random.random(200) * 20 - 10
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print("test 2: ", mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)


if __name__ == "__main__":
    unittest.main()
