"""
Copyright 2017, Andrew Lin
All rights reserved.

This software is licensed under the BSD 3-Clause License.
See LICENSE.txt at the root of the project or
https://opensource.org/licenses/BSD-3-Clause
"""
import matplotlib.pyplot as plt
import numpy as np


def swap(a: np.ndarray, li: int, ri: int):
    """Swap elements of numpy array.

    Args:
        a: numpy array.
        li: left index.
        ri: right index.
    """
    tmp = np.copy(a[li])
    a[li] = a[ri]
    a[ri] = tmp


def ccw(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> int:
    """Determine if points turn counter-clockwise.

    Args:
        p1: first point.
        p2: second point.
        p3: third point.

    Returns:
        -1 => counter-clockwise turn.
        1 => clockwise turn.
        0 => collinear.
    """
    dx1 = p2[0] - p1[0]
    dy1 = p2[1] - p1[1]
    dx2 = p3[0] - p1[0]
    dy2 = p3[1] - p1[1]

    dx1dy2 = dx1 * dy2
    dy1dx2 = dy1 * dx2

    if dx1dy2 > dy1dx2:
        return 1
    if dx1dy2 < dy1dx2:
        return -1
    if dx1 * dx2 < 0 or dy1 * dy2 < 0:
        return -1
    if dx1 * dx1 + dy1 * dy1 < dx2 * dx2 + dy2 * dy2:
        return 1

    return 0


def extract_primary(points: np.ndarray) -> tuple:
    """Kind of self-explanatory.

    Args:
        points:

    Returns:
        primary (start) point, remaining points.
    """
    min_point_idx = np.argmin(points)
    primary = np.copy(points[min_point_idx])
    remaining_points = np.concatenate(
        (points[:min_point_idx], points[min_point_idx + 1:])
    )
    return primary, remaining_points


def sort_for_graham_scan(points: np.ndarray, primary: np.ndarray) -> np.ndarray:
    """Sort points for graham scan.

    Args:
        points: points to sort .
        primary: primary (start) point (not in array).

    Returns:
        sorted points.
    """
    point_slopes = np.array([v[1] / v[0] for v in points])
    sorted_indexes = np.argsort(point_slopes)
    sorted_points = np.array(points)[sorted_indexes]
    hull = np.concatenate(
        (sorted_points[-1:], [primary], sorted_points)
    )
    return hull


def find_hull_vertices(points: np.ndarray) -> np.ndarray:
    """Finds points that don't require clockwise turns to connect.

    Args:
        points: sorted points. Will be modified by this process.

    Returns:
        hull vertices.
    """
    M = 3
    N = points.shape[0]
    for i in range(4, N):
        while ccw(points[M], points[M - 1], points[i]) >= 0:
            M -= 1

        M += 1
        swap(points, M, i)

    return points[1:M + 1]


def graham_scan(points: np.ndarray) -> np.ndarray:
    """Find vertices of convex hull around points.

    Args:
        points: (x, y) coordinates of points.

    Returns:
        hull: points that are vertices on convex hull.
    """
    primary, remaining_points = extract_primary(points)
    sorted_points = sort_for_graham_scan(remaining_points, primary)
    hull = find_hull_vertices(sorted_points)
    return hull


def main():
    """Convex Hull

    Implements a mish-mash of the Graham Scan algorithms presented in Robert
    Sedgewick's _Algorithms in C++_ (1992), and the Wikipedia entry
    (https://en.wikipedia.org/wiki/Graham_scan) as of March 2017, which itself
    is supposedly based off of Sedgewick's _Algorithms_, 4th Edition.
    """
    points = np.array(
        [[1, 1], [2, 5], [3, 2], [4, 4], [5, 2], [6, 3], [2, 3], [3, 4], [5, 3]]
    )
    hull = graham_scan(points)
    hull = np.concatenate((hull, [hull[0]]))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(points[:, 0], points[:, 1])
    ax.plot(hull[:, 0], hull[:, 1], 'r')
    ax.set_title('Convex Hull using Graham Scan')
    plt.show()

if __name__ == '__main__':
    main()
