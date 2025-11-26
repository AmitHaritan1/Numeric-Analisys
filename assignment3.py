"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""
import math

import numpy as np
import time
import random


class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def newton_raphson(self, func, x0, max_iter, start, part, maxerr):
        """
        Find a root of the function using Newton-Raphson method.
        
        Parameters
        ----------
        func : callable
            The function to find its roots.
        x0 : float
            The initial guess for the root.
        max_iter : int
            Maximum number of iterations to perform.
        start : float
            The starting boundary of the search interval.
        part : float
            The size of the search interval.
        maxerr : float
            An upper bound on the difference between the function values 
            at the approximate intersection points.
            
        Returns
        -------
        float or None
            The root of func in the interval, or None if not found.
        """
        current_estimate = x0
        for iteration in range(max_iter):
            function_value = func(current_estimate)
            if function_value is None:
                return None
            function_value_at_perturbed_x = func(current_estimate + 1e-8)
            if function_value_at_perturbed_x is None:
                return None
            numerical_derivative = ((function_value_at_perturbed_x - function_value) / 1e-8)
            if numerical_derivative == 0:
                return None
            next_estimate = current_estimate - function_value / numerical_derivative
            if next_estimate < start or next_estimate > start + part:
                break
            # Update the root estimate using the derivative by definition

            if abs(function_value) < maxerr:  # Check if the tolerance is satisfied
                return next_estimate

            current_estimate = next_estimate

        return None  # Return None if the maximum number of iterations is reached

    def bisection(self, func, a, b, maxerr):
        """
        Find a root of the function using the bisection method.
        
        Parameters
        ----------
        func : callable
            The function to find its roots.
        a : float
            Minimum value of the search interval.
        b : float
            Maximum value of the search interval.
        maxerr : float
            An upper bound on the difference between the function values 
            at the approximate intersection points.
            
        Returns
        -------
        float or None
            The root of func in [a, b], or None if no root exists in the interval.
        """
        if func(a) * func(b) >= 0:
            return None

        midpoint = (a + b) / 2
        while abs(func(midpoint)) >= maxerr:
            if func(midpoint) == 0:
                return midpoint
            elif func(midpoint) * func(a) < 0:
                b = midpoint
            else:
                a = midpoint
            midpoint = (a + b) / 2

        return midpoint

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001):
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.

        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        # Define the difference function: f1(x) - f2(x)
        def difference_function(x):
            f1_value = f1(x)
            if f1_value is None:
                return None
            return f1(x) - f2(x)

        segment_size = abs(b - a) / 500
        intersection_points = []

        if abs(difference_function(a)) < maxerr:
            intersection_points.append(a)

        segment_start = a
        segment_end = a
        while segment_end < b:
            segment_end += segment_size
            intersection_point = self.newton_raphson(
                difference_function, segment_start, 9, segment_end, segment_size, maxerr
            )
            if intersection_point is not None:
                intersection_points.append(intersection_point)
            if intersection_point is None and difference_function(segment_start) * difference_function(segment_end) <= 0:
                bisection_root = self.bisection(difference_function, segment_start, segment_end, maxerr)
                if bisection_root is not None:
                    intersection_points.append(bisection_root)

            segment_start = segment_end

        # Remove duplicate intersection points that are too close together
        unique_intersection_points = []
        if len(intersection_points) > 0:
            unique_intersection_points.append(intersection_points[0])
        if len(intersection_points) > 1:
            for i in range(1, len(intersection_points)):
                if (intersection_points[i] is not None and 
                    intersection_points[i-1] is not None and 
                    intersection_points[i] - intersection_points[i - 1] > maxerr):
                    unique_intersection_points.append(intersection_points[i])
                else:
                    intersection_points[i] = intersection_points[i - 1]
        return unique_intersection_points


    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the integration error. 
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions. 
        
        Integration error will be measured compared to the actual value of the 
        definite integral. 
        
        Note: It is forbidden to call f more than n times. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """
        # if n == 1:
        #     return np.float32((b-a) * f((b-a)/2))
        # elif n == 2:
        #     return np.float32((b-a) * (f(a)+f(b))/2)
        #
        #
        # return self.simpson_integrate(f,a,b,n)
        quadrature_points, quadrature_weights = self.gauss_legendre(a, b, n)
        # Evaluate the integral using the Gauss-Legendre quadrature method
        return np.float32(sum([quadrature_weights[i] * f(quadrature_points[i]) for i in range(len(quadrature_weights))]))




    def gauss_legendre(self, a, b, n):
        """
        Compute Gauss-Legendre quadrature points and weights for the given interval.
        
        Parameters
        ----------
        a : float
            Lower bound of the integration interval.
        b : float
            Upper bound of the integration interval.
        n : int
            Number of quadrature points to use.
            
        Returns
        -------
        tuple
            A tuple (scaled_points, scaled_weights) where:
            - scaled_points: array of quadrature points scaled to [a, b]
            - scaled_weights: array of corresponding quadrature weights
        """
        standard_points, standard_weights = np.polynomial.legendre.leggauss(n)
        scaled_points = (b - a) * 0.5 * standard_points + (b + a) * 0.5
        scaled_weights = (b - a) * 0.5 * standard_weights
        return scaled_points, scaled_weights


    def simpson_integrate(self, f, a, b, n):
        """
        Compute the definite integral using Simpson's rule.
        
        Parameters
        ----------
        f : callable
            The function to integrate.
        a : float
            Lower bound of the integration interval.
        b : float
            Upper bound of the integration interval.
        n : int
            Number of points to use (will be adjusted to be even).
            
        Returns
        -------
        np.float32
            The computed integral value.
        """
        sign_multiplier = 1
        if a > b:
            swap_variable = a
            a = b
            b = swap_variable
            sign_multiplier = -1

        if n % 2 == 1:
            n = n - 1
        else:
            n = n - 2
        step_size = (b - a) / n
        current_x = a

        integral_sum = f(a) + f(b)
        for i in range(1, n):
            current_x = current_x + step_size
            if i % 2 == 1:
                integral_sum += 4 * f(current_x)
            else:
                integral_sum += 2 * f(current_x)

        result = np.float32((step_size / 3) * integral_sum * sign_multiplier)
        if result is None:
            return np.float32(0)
        return result

    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds
        all intersection points between the two functions to work correctly.

        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area.

        In order to find the enclosed area the given functions must intersect
        in at least two points. If the functions do not intersect or intersect
        in less than two points this function returns NaN.
        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """

        intersection_points = self.intersections(f1, f2, 1, 100, maxerr=0.001)
        # Check if there are not enough intersection points between the 2 functions
        if intersection_points is None:
            return None
        if len(intersection_points) < 2:
            return None
        total_area = 0
        for i in range(len(intersection_points) - 1):
            midpoint_x = (float(intersection_points[i]) + float(intersection_points[i + 1])) / 2
            if f1(float(midpoint_x)) > f2(float(midpoint_x)):
                # f1 is above f2 in this section
                total_area = total_area + self.simpson_integrate(f1, intersection_points[i], intersection_points[i + 1], 100)
                total_area = total_area - self.simpson_integrate(f2, intersection_points[i], intersection_points[i + 1], 100)
            else:
                # f2 is above f1 in this section
                total_area = total_area + self.simpson_integrate(f2, intersection_points[i], intersection_points[i + 1], 100)
                total_area = total_area - self.simpson_integrate(f1, intersection_points[i], intersection_points[i + 1], 100)

        final_area = np.float32(total_area)
        return final_area


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        r = ass3.integrate(f1, -1, 1, 10)

        self.assertEqual(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 20)
        true_result = -7.78662 * 10 ** 33

        print(f1(0.1))
        print(abs((r - true_result) / true_result))
        print("area is: ", r)
        # self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_areabetween_func(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1,3,7])
        f2 = np.poly1d([1, -16, 24])
        r = ass3.areabetween(f1, f2)
        true_result = 140.625
        print(abs((r - true_result)))
        print("area is ", r)
        self.assertGreaterEqual(0.001, abs((r - true_result)))

    def test_areabetween_func_None(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 0])
        # f1 = lambda x: None
        f2 = np.poly1d([1, 0, ])
        r = ass3.areabetween(f1, f2)
        true_result = None
        print(r)
        print("area is ", r)
        self.assertEqual(r, true_result)


if __name__ == "__main__":
    unittest.main()
