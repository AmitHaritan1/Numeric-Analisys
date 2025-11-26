"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable


class Assignment2:
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
            numerical_derivative = ((func(current_estimate + 1e-8) - function_value) / 1e-8)
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
            return None  # No root exists between [a, b]

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

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
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
            return f1(x) - f2(x)

        segment_size = 1 / 50
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

        return intersection_points


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))



    def test_Grader(self):

        ass2 = Assignment2()

        def f10(x):
             return np.sin(np.log(x))

        def f3(x):
            return np.sin(x**2)
        f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f3, f10, 1, 10, maxerr=0.001)
        # print(X)
        for x in X:
            self.assertGreaterEqual(0.001, abs(f3(x) - f10(x)))


if __name__ == "__main__":
    unittest.main()
