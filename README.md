# Numerical Analysis and Optimization

This repository contains implementations of various numerical analysis and optimization algorithms for a course on Numerical Analysis and Optimization.

## Overview

This project consists of five assignments, each focusing on different aspects of numerical computation:

1. **Assignment 1**: Function Interpolation using Lagrange Polynomials
2. **Assignment 2**: Finding Intersection Points of Two Functions
3. **Assignment 3**: Computing Area Between Two Functions
4. **Assignment 4**: Fitting Noisy Data with Bezier Curves
5. **Assignment 5**: Shape Fitting and Area Calculation from Noisy Contour Points

## Project Structure

```
.
├── assignment1.py      # Lagrange interpolation implementation
├── assignment2.py     # Function intersection finding
├── assignment3.py     # Area between curves calculation
├── assignment4.py     # Bezier curve fitting for noisy data
├── assignment5.py     # Shape fitting and area computation
├── commons.py         # Common utility functions and decorators
├── functionUtils.py   # Function utilities and abstract classes
├── sampleFunctions.py # Sample functions for testing
├── grader.py          # Grading utilities
└── README.md          # This file
```

## Dependencies

- Python 3.x
- NumPy
- scikit-learn (for K-means clustering in Assignment 5)
- tqdm (for progress bars in tests)

Install dependencies with:
```bash
pip install numpy scikit-learn tqdm
```

## Assignment Details

### Assignment 1: Function Interpolation

**Objective**: Interpolate a given function in a closed range using at most n points, minimizing interpolation error and running time.

**Key Features**:
- Uses Lagrange polynomial interpolation
- Implements local interpolation with caching for efficiency
- Binary search to find closest interpolation points
- Optimized for O(n) performance with preprocessing

**Main Method**: `interpolate(f, a, b, n)`
- `f`: The function to interpolate
- `a`: Beginning of interpolation range
- `b`: End of interpolation range
- `n`: Maximum number of points to use

**Example Usage**:
```python
from assignment1 import Assignment1
import numpy as np

ass1 = Assignment1()
f = np.poly1d([1, 0, -1])  # x^2 - 1
interpolated = ass1.interpolate(f, -10, 10, 100)
result = interpolated(5.0)  # Evaluate at x=5
```

### Assignment 2: Finding Intersection Points

**Objective**: Find all intersection points between two functions in a given range.

**Key Features**:
- Combines Newton-Raphson and bisection methods
- Handles multiple intersection points
- Efficiently searches through the interval

**Main Method**: `intersections(f1, f2, a, b, maxerr=0.001)`
- `f1`, `f2`: The two functions to find intersections between
- `a`, `b`: Search range boundaries
- `maxerr`: Maximum allowed error in intersection points

**Example Usage**:
```python
from assignment2 import Assignment2
import numpy as np

ass2 = Assignment2()
f1 = np.poly1d([-1, 0, 1])  # -x^2 + 1
f2 = np.poly1d([1, 0, -1])  # x^2 - 1
intersections = ass2.intersections(f1, f2, -1, 1)
```

### Assignment 3: Area Between Curves

**Objective**: Calculate the area enclosed between two functions using numerical integration.

**Key Features**:
- Finds all intersection points between functions
- Uses Simpson's rule and Gauss-Legendre quadrature for integration
- Works with float32 precision to minimize floating-point errors
- Handles multiple intersection points correctly

**Main Methods**:
- `intersections(f1, f2, a, b, maxerr)`: Find intersection points
- `integrate(f, a, b, n)`: Numerical integration using Gauss-Legendre quadrature
- `areabetween(f1, f2)`: Calculate total area between two functions

**Example Usage**:
```python
from assignment3 import Assignment3
import numpy as np

ass3 = Assignment3()
f1 = np.poly1d([-1, 3, 7])
f2 = np.poly1d([1, -16, 24])
area = ass3.areabetween(f1, f2)
```

### Assignment 4: Bezier Curve Fitting

**Objective**: Fit a Bezier curve model to noisy data points, minimizing mean least squares error.

**Key Features**:
- Uses Bezier curve interpolation for smooth fitting
- Handles noisy data effectively
- Respects time constraints (returns within max_time seconds)
- Supports polynomial degrees 0-12

**Main Method**: `fit(f, a, b, d, maxtime)`
- `f`: Noisy function to fit
- `a`, `b`: Fitting range
- `d`: Expected polynomial degree
- `maxtime`: Maximum allowed time in seconds

**Example Usage**:
```python
from assignment4 import Assignment4
from sampleFunctions import poly, NOISY

ass4 = Assignment4()
noisy_function = NOISY(0.01)(poly(1, 1, 1))
fitted = ass4.fit(noisy_function, 0, 1, 3, maxtime=5)
result = fitted(0.5)  # Evaluate fitted function at x=0.5
```

### Assignment 5: Shape Fitting and Area Calculation

**Objective**: Fit a shape model to noisy contour points and calculate its area.

**Key Features**:
- Uses K-means clustering to reduce noise in sampled points
- Implements shoelace formula for polygon area calculation
- Sorts points by angle for proper contour reconstruction
- Handles noisy sampling functions

**Main Methods**:
- `fit_shape(sample, maxtime)`: Fit a shape from noisy samples
- `area(contour, maxerr)`: Calculate area from contour points

**Example Usage**:
```python
from assignment5 import Assignment5
from sampleFunctions import noisy_circle

ass5 = Assignment5()
circle_sampler = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
shape = ass5.fit_shape(sample=circle_sampler, maxtime=5)
area = shape.area()
```

## Testing

Each assignment includes unit tests. Run tests with:

```bash
python assignment1.py
python assignment2.py
python assignment3.py
python assignment4.py
python assignment5.py
```

## Key Algorithms and Techniques

### Numerical Methods Used

1. **Lagrange Interpolation**: Polynomial interpolation using basis polynomials
2. **Newton-Raphson Method**: Root finding using iterative approximation
3. **Bisection Method**: Robust root finding for bracketed intervals
4. **Simpson's Rule**: Numerical integration using parabolic approximations
5. **Gauss-Legendre Quadrature**: High-accuracy numerical integration
6. **Bezier Curves**: Smooth curve fitting using Bernstein polynomials
7. **K-means Clustering**: Noise reduction in sampled data points
8. **Shoelace Formula**: Polygon area calculation from vertex coordinates

### Performance Optimizations

- **Caching**: Assignment 1 caches interpolators for frequently accessed regions
- **Local Interpolation**: Uses small-degree polynomials locally instead of global high-degree polynomials
- **Binary Search**: Efficient point location in sorted arrays
- **Segment-based Search**: Divides intervals into segments for efficient root finding

## Code Quality

The codebase follows these principles:
- **Meaningful Variable Names**: All variables use descriptive names
- **Comprehensive Docstrings**: All functions include detailed documentation
- **Type Hints**: Function signatures include type information
- **Modular Design**: Each assignment is self-contained with clear interfaces

## Notes

- Assignment 3 requires float32 precision for all calculations
- Assignment 4 must respect time constraints and return within the specified time limit
- Assignment 5 uses K-means clustering with 36 clusters for noise reduction
- All assignments are designed to handle edge cases and noisy input data

## Author

This project was developed as part of a Numerical Analysis and Optimization course.

## License

This project is for educational purposes.
