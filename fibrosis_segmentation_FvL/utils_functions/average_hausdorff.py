import numpy as np
import math
from scipy.spatial import cKDTree

def reg_hausdorff_distance(image0, image1):
    a_points = np.transpose(np.nonzero(image0))
    b_points = np.transpose(np.nonzero(image1))
    # print(a_points)
    # print(b_points)

    # Handle empty sets properly:
    # - if both sets are empty, return zero
    # - if only one set is empty, return infinity
    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.sqrt(image0.shape[0]**2 + image0.shape[1]**2)
    elif len(b_points) == 0:
        # return np.inf
        return np.sqrt(image0.shape[0]**2 + image0.shape[1]**2)
    
    max_ab = max(cKDTree(a_points).query(b_points, k=1)[0])
    max_ba = max(cKDTree(b_points).query(a_points, k=1)[0])

    return max(max_ab, max_ba)


def avg_hausdorff_distance(image0, image1):
    """Calculate the Hausdorff distance between nonzero elements of given images.
    The Hausdorff distance [1]_ is the maximum distance between any point on
    ``image0`` and its nearest point on ``image1``, and vice-versa.
    Parameters
    ----------
    image0, image1 : ndarray
        Arrays where ``True`` represents a point that is included in a
        set of points. Both arrays must have the same shape.
    Returns
    -------
    distance : float
        The Hausdorff distance between coordinates of nonzero pixels in
        ``image0`` and ``image1``, using the Euclidian distance.
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Hausdorff_distance
    Examples
    --------
    >>> points_a = (3, 0)
    >>> points_b = (6, 0)
    >>> shape = (7, 1)
    >>> image_a = np.zeros(shape, dtype=bool)
    >>> image_b = np.zeros(shape, dtype=bool)
    >>> image_a[points_a] = True
    >>> image_b[points_b] = True
    >>> hausdorff_distance(image_a, image_b)
    3.0
    """
    a_points = np.transpose(np.nonzero(image0))
    b_points = np.transpose(np.nonzero(image1))
    # print(a_points)
    # print(b_points)

    # Handle empty sets properly:
    # - if both sets are empty, return zero
    # - if only one set is empty, return infinity
    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.sqrt(image0.shape[0]**2 + image0.shape[1]**2)
    elif len(b_points) == 0:
        # return np.inf
        return np.sqrt(image0.shape[0]**2 + image0.shape[1]**2)
    
    hausdorff_distances_ab = cKDTree(a_points).query(b_points, k=1)[0]
    avg_hausdorff_distances_ab = np.average(hausdorff_distances_ab)
    hausdorff_distances_ba = cKDTree(b_points).query(a_points, k=1)[0]
    avg_hausdorff_distances_ba = np.average(hausdorff_distances_ba)
    avg_hausdorff = max(avg_hausdorff_distances_ab, avg_hausdorff_distances_ba)

    return avg_hausdorff