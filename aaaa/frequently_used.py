import numpy as np
import numba as nb


def select_top(scr, count):
    top_val = np.partition(scr, -count)[-count]
    high_scored_cell = np.where(scr > top_val)[0]
    low_scored_cell = np.where(scr == top_val)[0]
    top_ind = np.empty(count, dtype='int32')
    top_ind[:high_scored_cell.size] = high_scored_cell
    top_ind[high_scored_cell.size:] = np.random.choice(low_scored_cell, count-high_scored_cell.size, False)
    return top_ind


def select_bottom(scr, count):
    top_val = np.partition(scr, count)[count]
    low_scored_cell = np.where(scr < top_val)[0]
    high_scored_cell = np.where(scr == top_val)[0]
    bottom_ind = np.empty(count, dtype='int32')
    bottom_ind[:low_scored_cell.size] = low_scored_cell
    bottom_ind[low_scored_cell.size:] = np.random.choice(high_scored_cell, count-low_scored_cell.size, False)
    return bottom_ind


@nb.jit(parallel=True, nopython=True)  # Equivalent with: np.isin(matrix_in, search_for) but ~60 times faster
def nb_isin(matrix, search):
    flat_mat = matrix.reshape(-1)
    out = np.empty(flat_mat.shape[0], dtype=nb.boolean)
    search = set(search)
    for i in nb.prange(flat_mat.shape[0]):
        if flat_mat[i] in search:
            out[i] = True
        else:
            out[i] = False
    return out.reshape(matrix.shape)


@nb.jit(parallel=True, nopython=True)
def sum_isin(matrix, search):  # Equivalent with: np.sum(np.isin(matrix_in, search_for), axis=1) but ~50 times faster
    out = np.zeros(matrix.shape[0], dtype=nb.int32)
    search = set(search)
    for i in nb.prange(matrix.shape[0]):
        for j in nb.prange(matrix.shape[1]):
            if matrix[i][j] in search:
                out[i] += 1
    return out


@nb.jit(parallel=True, nopython=True)
def sum_isin2d(matrix, search, num):
    out = np.zeros(matrix.shape[0], dtype=nb.int32)
    for i in nb.prange(matrix.shape[0]):
        search_set = set(np.where(search[i] == num)[0])
        for j in nb.prange(matrix.shape[1]):
            if matrix[i][j] in search_set:
                out[i] += 1
            else:
                out[i] += 0
    return out


@nb.jit(parallel=True, nopython=True)
def sum_isin2d_col(matrix, search, num, cpc):
    out = np.zeros(matrix.shape[0], dtype=nb.int32)
    for i in nb.prange(matrix.shape[0]):
        search_set = set(np.where(search[i//cpc] == num)[0])
        for j in nb.prange(matrix.shape[1]):
            if matrix[i][j] in search_set:
                out[i] += 1
            else:
                out[i] += 0
    return out


@nb.jit(parallel=True, nopython=True)
def small_neg(matrix, mul=0.5, pos=True):
    out = np.copy(matrix).reshape(-1)
    if pos:
        for i in nb.prange(matrix.size):
            if out[i] < 0:
                out[i] = out[i]*mul
    else:
        for i in nb.prange(matrix.size):
            if out[i] > 0:
                out[i] = -out[i]*mul
            else:
                out[i] = -out[i]
    return out.reshape(matrix.shape)


@nb.jit(parallel=True, nopython=True, fastmath=True)
def nb_clip(mat, lo, hi):  # Equivalent with: np.clip(mat, lo, hi)
    out = np.copy(mat).reshape(-1)
    for i in nb.prange(out.size):
        if out[i] <= lo:
            out[i] = lo
        elif out[i] >= hi:
            out[i] = hi
    return out.reshape(mat.shape)


@nb.jit(parallel=True, nopython=True, fastmath=True)
def nb_add(mat1, mat2):  # Equivalent with: mat1 + mat2
    for i in nb.prange(mat1.size):
        mat1[i] += mat2[i]
    return mat1


@nb.jit(parallel=True, nopython=True, fastmath=True)
def nb_add3(mat1, mat2, mat3):  # Equivalent with: mat1 + mat2
    for i in nb.prange(mat1.size):
        mat1[i] += mat2[i] + mat3[i]
    return mat1


@nb.jit(parallel=True, nopython=True, fastmath=True)
def nb_sub(mat1, mat2):  # Equivalent with: mat1 - mat2
    for i in nb.prange(mat1.size):
        mat1[i] -= mat2[i]
    return mat1


@nb.jit(parallel=True, nopython=True)
def nb_eq(mat, val):  # Equivalent with: mat == val
    mat1d = mat.reshape(-1)
    out = np.empty(mat1d.size, dtype=nb.boolean)
    for i in nb.prange(mat1d.size):
        if mat1d[i] == val:
            out[i] = True
        else:
            out[i] = False
    return out.reshape(mat.shape)


@nb.jit(parallel=True, nopython=True)
def nb_ge(mat, val):  # Equivalent with: mat >= val
    mat1d = mat.reshape(-1)
    out = np.empty(mat1d.size, dtype=nb.boolean)
    for i in nb.prange(mat1d.size):
        if mat1d[i] >= val:
            out[i] = True
        else:
            out[i] = False
    return out.reshape(mat.shape)


@nb.jit(parallel=True, nopython=True)
def nb_gt(mat, val):  # Equivalent with: mat > val
    mat1d = mat.reshape(-1)
    out = np.empty(mat1d.size, dtype=nb.boolean)
    for i in nb.prange(mat1d.size):
        if mat1d[i] > val:
            out[i] = True
        else:
            out[i] = False
    return out.reshape(mat.shape)
