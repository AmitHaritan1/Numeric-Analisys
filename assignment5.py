"""
In this assignment you should fit a model function of your choice to data
that you sample from a contour of given shape. Then you should calculate
the area of that shape.

The sampled data is very noisy so you should minimize the mean least squares
between the model you fit and the data points you sample.

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You
must make sure that the fitting function returns at most 5 seconds after the
allowed running time elapses. If you know that your iterations may take more
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment.
Note: !!!Despite previous note, using reflection to check for the parameters
of the sampled function is considered cheating!!! You are only allowed to
get (x,y) points from the given shape by calling sample().
"""
import numpy as np
import time
import random
from functionUtils import AbstractShape
from sklearn.cluster import KMeans
import math


class MyShape(AbstractShape):
    """
    A shape implementation that uses K-means clustering and the shoelace formula
    to compute the area of a shape from noisy sample points.
    """
    def __init__(self, sampled_points, sampling_function):
        """
        Initialize the shape with sampled points and a sampling function.
        
        Parameters
        ----------
        sampled_points : list
            Initial list of sampled points (may be empty).
        sampling_function : callable
            Function that returns a (x, y) point when called.
        """
        self.points = sampled_points
        self.sample = sampling_function
        self.sorted_contour_points = []

    def contour(self, num_points):
        """
        Get contour points for the shape.
        
        Parameters
        ----------
        num_points : int
            Number of contour points to return.
            
        Returns
        -------
        list
            List of (x, y) contour points.
        """
        if len(self.sorted_contour_points) < num_points:
            contour_points = []
            for i in range(num_points):
                contour_points.append(self.sample())
            return contour_points
        else:
            return self.sorted_contour_points[:num_points]

    def area(self):
        """
        Compute the area of the shape using K-means clustering and the shoelace formula.
        
        Returns
        -------
        np.float32
            The computed area of the shape.
        """
        def sort_by_angle(points, center_point=(0, 0)):
            """
            Sort points by their angle relative to a center point.
            
            Parameters
            ----------
            points : array-like
                Points to sort.
            center_point : tuple, optional
                Reference point for angle calculation. Defaults to (0, 0).
                
            Returns
            -------
            list
                Points sorted by angle.
            """
            def calculate_angle(point):
                x, y = point
                center_x, center_y = center_point
                angle = math.atan2(y - center_y, x - center_x)
                return angle

            return sorted(points, key=calculate_angle)

        initial_sample_points = []
        while len(initial_sample_points) < 10000:
            initial_sample_points.append(self.sample())

        kmeans = KMeans(n_clusters=36, n_init=12)
        kmeans.fit(initial_sample_points)
        cluster_centers = np.array(kmeans.cluster_centers_, dtype=np.float32)

        x_coordinates, y_coordinates = zip(*cluster_centers)
        center_x = np.mean(x_coordinates)
        center_y = np.mean(y_coordinates)
        centroid = (center_x, center_y)
        self.sorted_contour_points = sort_by_angle(cluster_centers, centroid)
        num_vertices = len(self.sorted_contour_points)
        computed_area = 0.0
        for i in range(num_vertices - 1):
            computed_area += (self.sorted_contour_points[i][0] * self.sorted_contour_points[i + 1][1] - 
                            self.sorted_contour_points[i][1] * self.sorted_contour_points[i + 1][0])
        computed_area += (self.sorted_contour_points[num_vertices - 1][0] * self.sorted_contour_points[0][1] - 
                         self.sorted_contour_points[num_vertices - 1][1] * self.sorted_contour_points[0][0])
        return np.float32(abs(computed_area) / 2)



class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def shoelace_formula(self, contour_coordinates):
        """
        Compute the area of a polygon using the shoelace formula.
        
        Parameters
        ----------
        contour_coordinates : list of tuples
            List of (x, y) coordinates representing the polygon vertices.
            
        Returns
        -------
        float
            The area of the polygon.
        """
        num_vertices = len(contour_coordinates)
        polygon_area = 0.0
        for i in range(num_vertices - 1):
            polygon_area += (contour_coordinates[i][0] * contour_coordinates[i + 1][1] - 
                           contour_coordinates[i][1] * contour_coordinates[i + 1][0])
        polygon_area += (contour_coordinates[num_vertices - 1][0] * contour_coordinates[0][1] - 
                        contour_coordinates[num_vertices - 1][1] * contour_coordinates[0][0])
        return abs(polygon_area) / 2



    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        """
        Compute the area of the shape with the given contour.

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour - a function that takes an integer n
            and returns n contour points.
        maxerr : float, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        np.float32
            The area of the shape.
        """
        num_contour_points = 600
        contour_coordinates = contour(num_contour_points)
        if num_contour_points <= 2:
            return np.float32(0)
        computed_area = self.shoelace_formula(contour_coordinates)
        return np.float32(computed_area)





    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a shape object that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        sample : callable
            A function which returns a data point (x, y) that is near the shape contour
            when called.
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        AbstractShape
            An object extending AbstractShape that represents the fitted shape.
        """
        sampled_points = []
        fitted_shape = MyShape(sampled_points, sample)
        return fitted_shape


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_circle_area_from_contour(self):
        circ = Circle(cx=1, cy=1, radius=1, noise=0.0)
        ass5 = Assignment5()
        T = time.time()
        a_computed = ass5.area(contour=circ.contour, maxerr=0.1)
        T = time.time() - T
        a_true = circ.area()
        self.assertLess(abs((a_true - a_computed) / a_true), 0.1)


if __name__ == "__main__":
    unittest.main()
