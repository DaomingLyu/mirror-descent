import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg

def least_squares(A, b, blocks, iters=1000, tolerance=1e-9):
    n_vector = np.array(
            [([block_size]*block_size) for block_size in blocks])
    n_vector = np.squeeze(n_vector.reshape((-1, 1)))
    n_vector = n_vector.astype(float)
    if sps.issparse(A):
        Lf = sps.linalg.svds(A, 1, return_singular_vectors=False)[:]
        Lf = Lf[0]
    else:
        Lf = np.linalg.svd(A, compute_uv=False)
        Lf = Lf[0]

    def t_(k):
        return np.squeeze(np.asarray(
            np.sqrt(2*np.log(n_vector))/(np.sqrt(k)*Lf)))

    def compute_gradient(x):
        inside = np.squeeze(np.asarray(A.dot(x))) - b
        return np.squeeze(
                np.asarray(A.T.dot(inside)))

    x = np.divide(1.0, n_vector)
    x = np.squeeze(np.asarray(x))
    for _iter in xrange(1, iters+1):
        x_prev = x
        up = compute_gradient(x) 
        up *= t_(_iter)
        x = x * np.exp(-up)

        beginning = 0
        for block in blocks:
            x_section = x[beginning:block+beginning]
            x[beginning:block+beginning] = x_section/np.sum(x_section)
            beginning += block

        x = np.squeeze(np.asarray(x))
        if np.linalg.norm(x - x_prev, np.inf) < tolerance:
            break

    return x

def test_least_squares():
    one_block_soln = least_squares(
            np.array([[1, 0], [0, 1]]),
            np.array([1, 1.5]),
            np.array([2]))
    np.testing.assert_almost_equal(np.array([0.25, 0.75]), one_block_soln)

    two_block_simple_soln = least_squares(
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
            np.array([1, 1.5, 1, 1.5]),
            np.array([2, 2]))
    np.testing.assert_almost_equal(np.array([0.25, 0.75, 0.25, 0.75]),
            two_block_simple_soln)

if __name__ == '__main__':
    test_least_squares()
