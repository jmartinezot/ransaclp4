import numpy as np
from typing import List, Optional, Union
import open3d as o3d


def sampling_np_arrays_from_enumerable(source_list: Union[List, np.ndarray], cardinality_of_np_arrays: int, number_of_np_arrays: int=1, num_source_elems: Optional[int] = None, seed: Optional[int] = None) -> List[np.ndarray]:
    """
    Returns a list with **number_of_np_arrays** numpy arrays of size **cardinality_of_np_arrays** with random elements from a list or numpy array.

    :param source_list: The list or numpy array to sample from.
    :type source_list: Union[list, np.ndarray]
    :param cardinality_of_np_arrays: The cardinality of the numpy arrays to generate.
    :type cardinality_of_np_arrays: int
    :param number_of_np_arrays: The number of numpy arrays to generate. Default is 1.
    :type number_of_np_arrays: int
    :param num_source_elems: The number of elements in the source list or numpy array. Default is None.
    :type num_source_elems: int
    :param seed: The seed value for the random number generator. Default is None.
    :type seed: int
    :return: List of numpy arrays containing the sampled arrays.
    :rtype: List[np.ndarray]

    :Example:

    ::

        >>> import sampling
        >>> import numpy as np
        >>> sampling.sampling_np_arrays_from_enumerable(np.array([1,2,3,4,5,6,7,8,9,10]), 3, 2, seed=42)
        >>> [array([9, 2, 6]), array([1, 8, 3])]
        >>> np.random.seed(42)
        >>> random_3d_points = np.random.rand(100, 3)
        >>> random_np_arrays_of_points = sampling.sampling_np_arrays_from_enumerable(random_3d_points, cardinality_of_np_arrays=3, number_of_np_arrays=4, seed=42)
        >>> random_np_arrays_of_points
        >>> [array([[0.85300946, 0.29444889, 0.38509773],
        >>> [0.72821635, 0.36778313, 0.63230583],
        >>> [0.54873379, 0.6918952 , 0.65196126]]),
        >>> array([[0.32320293, 0.51879062, 0.70301896],
        >>> [0.11986537, 0.33761517, 0.9429097 ],
        >>> [0.18657006, 0.892559  , 0.53934224]]),
        >>> array([[0.14092422, 0.80219698, 0.07455064],
        >>> [0.94045858, 0.95392858, 0.91486439],
        >>> [0.60754485, 0.17052412, 0.06505159]]),
        >>> array([[0.37454012, 0.95071431, 0.73199394],
        >>> [0.59789998, 0.92187424, 0.0884925 ],
        >>> [0.11959425, 0.71324479, 0.76078505]])]

    """
    if num_source_elems is None:
        num_source_elems = len(source_list)
    if seed is not None:
        np.random.seed(seed)
    random_elems_indices = np.random.choice(range(num_source_elems), size= cardinality_of_np_arrays * number_of_np_arrays, replace=False)
    random_elems = np.array(source_list)[random_elems_indices]
    # Split random_elems into sets
    sampled_np_arrays = [random_elems[i * cardinality_of_np_arrays:(i + 1) * cardinality_of_np_arrays] for i in range(number_of_np_arrays)]
    return sampled_np_arrays

