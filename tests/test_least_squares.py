import unittest

import numpy as np
from mirror_descent import least_squares

class TestMirrorDescent(unittest.TestCase):

    def setUp(self):
        pass

    def test_least_squares_one_block(self):
        one_block_soln = least_squares(
                np.array([[1, 0], [0, 1]]),
                np.array([1, 1.5]),
                np.array([2]))
        np.testing.assert_almost_equal(np.array([0.25, 0.75]), one_block_soln)

    def test_least_squares_two_blocks(self):
        two_block_simple_soln = least_squares(
                np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                np.array([1, 1.5, 1, 1.5]),
                np.array([2, 2]))
        np.testing.assert_almost_equal(np.array([0.25, 0.75, 0.25, 0.75]),
                two_block_simple_soln)

if __name__=='__main__':
    unittest.main()
