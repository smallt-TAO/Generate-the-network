# -*- coding: utf-8 -*-
"""

Data load.
"""

import numpy as np
import copy
import random


def pre_handle(matrix):
    """
    For the nn networks better work.
    :param matrix:
    :return:
    """
    iter_num = 0
    m = len(matrix)
    matrix_d = [0] * m
    for i in range(m):
        for j in range(m):
            if matrix[i][j] == 1:
                matrix_d[i] += 1
    dict_d = {}
    for i in range(m):
        dict_d[i] = matrix_d[i]
    if max(matrix_d) == min(matrix_d):
        for i in range(m):
            for j in range(m):
                if matrix[i][j] == 0:
                    matrix[i][i] = 1
                    matrix[j][i] = 1
                    return matrix
    return matrix


def matrix_change(matrix):
    m = len(matrix)
    matrix_d = [0] * m
    for i in range(m):
        for j in range(m):
            if matrix[i][j] == 1:
                matrix_d[i] += 1
    dict_d = {}
    for i in range(m):
        dict_d[i] = matrix_d[i]
    max_d = max(matrix_d)
    matrix_d_new = []
    for num_v in range(max_d, 0, -1):
        for (k, v) in dict_d.items():
            if v == num_v:
                matrix_d_new.append(k)
    array_new = matrix_alter1(matrix, max_d, matrix_d)
    matrix_new = matrix_alter(array_new, matrix)

    return matrix_new


def matrix_random(matrix):
    m = len(matrix)
    array = [i for i in range(m)]
    random.shuffle(array)
    matrix_new = [([0] * m) for si in range(m)]
    for i in range(m):
        for j in range(m):
            matrix_new[i][j] = matrix[array[i]][array[j]]

    return matrix_new


def matrix_alter(array_d, matrix):
    m = len(matrix)
    matrix_v = [array_d[0]]
    flag = True
    for i in range(1, m):
        if flag:
            matrix_v.append(array_d[i])
            flag = False
        else:
            matrix_v.insert(0, array_d[i])
            flag = True
    matrix_new = [([0] * m) for si in range(m)]
    for i in range(m):
        for j in range(m):
            matrix_new[i][j] = matrix[matrix_v[i]][matrix_v[j]]

    return matrix_new


def matrix_alter1(matrix, max_num, matrix_d):
    m = len(matrix)
    matrix_d_new = []
    # behind the Point we talked.
    array_behind = [0 for si in range(m)]
    for behind_j in range(1, max_num + 1):
        for i in range(m):
            for j in range(m):
                if matrix_d[i] == behind_j and matrix[i][j] == 1 and matrix_d[j] == behind_j + 1:
                    array_behind[j] += 1

    for i in range(max_num, -1, -1):
        array_new = []
        size_new = len(matrix_d_new)
        loss_array = [0] * m
        loss = m
        for j in range(size_new):
            loss_array[matrix_d_new[j]] = loss
            loss -= 1
        for j in range(m):
            if matrix_d[j] == i:
                array_new.append(j)
        if len(array_new) != 0 and i != max_num:
            iter_num = len(array_new)
            iter_array = []
            iter_flag = [True] * iter_num
            for kk in range(iter_num):
                mat_1 = matrix[array_new[kk]]
                sum_num = 0
                for run in range(size_new):
                    sum_num += mat_1[matrix_d_new[run]] * loss_array[matrix_d_new[run]]
                sum_num += array_behind[array_new[kk]]
                iter_array.append(sum_num)
            iter_array_or = copy.deepcopy(iter_array)
            iter_array.sort()
            iter_array.reverse()
            for k in iter_array:
                for l in range(iter_num):
                    if iter_array_or[l] == k and iter_flag[l] is True:
                        matrix_d_new.append(array_new[l])
                        iter_flag[l] = False
                        break
        if i == max_num:  # handle the same d.
            new_s = array_new[0]
            for start_i in range(len(array_new) - 1):
                matrix_d_new.append(new_s)
                array_new.remove(new_s)
                for start_j in range(m):
                    if matrix[new_s][start_j] == 1 and (start_j in array_new):
                        new_s = start_j
                        break
                    if matrix[new_s][start_j] == 0 and (start_j in array_new):
                        new_s = start_j
                        break

            matrix_d_new.extend(array_new)

    return matrix_d_new


def small_word(size, k, p):
    """
    :param size: size of the matrix
    :param k: size of the link limit
    :param p: size of the possibly
    :return: a small word network
    """
    # init ncn
    matrix_b = [([0] * size) for si in range(size)]
    for i in range(size):
        for j in range(i + 1, i + k / 2 + 1):
            if j < size:
                matrix_b[i][j] = 1
                matrix_b[j][i] = 1
            else:
                matrix_b[i][j - size] = 1
                matrix_b[j - size][i] = 1
    # create the small word
    for i in range(size):
        for j in range(i + 1, size):
            if matrix_b[i][j] != 0:
                if random.random() < p:
                    matrix_b[i][j] = 0
                    matrix_b[j][i] = 0
                    s = random.randint(i + 1, size - 1)
                    matrix_b[i][s] = 1
                    matrix_b[s][i] = 1

    return matrix_b


def scale_free(size, num, m, key_p=0):
    """
    :param size: size of the matrix.
    :param num: size of the matrix begin.
    :param m: size of limit.
    :return: matrix_b
    """
    # init the matrix
    matrix_b = [([0] * size) for si in range(size)]
    for i in range(num):
        for j in range(i):
            matrix_b[i][j] = 1
            matrix_b[j][i] = 1
    # init the degree of vector.
    degree_b = [0] * size
    for i in range(num):
        degree_b[i] = num - 1
    # add the vector
    for i in range(num, size):
        mm = 0
        dd = degree_b[:i]
        while mm < m:
            mm += 1
            degree_n = 0
            for j in range(i):
                degree_n += dd[j]
            p = [0] * i
            p[0] = dd[0] / (degree_n + 0.000001)
            for j in range(1, i):
                p[j] = p[j - 1] + dd[j] / (degree_n + 0.000001)
            for j in range(i):
                if random.random() < p[j]:
                    if random.random() > key_p:
                        matrix_b[i][j] = 1
                        matrix_b[j][i] = 1
                        degree_b[i] += 1
                        degree_b[j] += 1
                        dd[j] = 0
                        break
    return matrix_b


def load_data(size=400, n=56, y_dim=10):
    y_label = np.zeros((size * 10,))
    x_image = np.zeros((size * 10, 1, n, n))
    echo = int(size / 50)
    iter_num = 0
    print("start to make matrix>>>>>>")
    print("small_word>>>>>>>>>")
    for k in [10, 12, 15, 16, 17]:
        for i in range(10):
            for ii in range(echo):
                p = 0.1 + float(0 / 100)
                ba = small_word(n, k, p)
                ba = matrix_random(ba)
                ba = matrix_change(ba)
                y_label[iter_num] = i
                x_image[iter_num, :, :, :] = ba
                iter_num += 1
        print("iter_num = " + str(iter_num))

    x_image = x_image.reshape((size * 10, n, n, 1)).astype(np.float)
    x_image = np.asarray(x_image)
    y_label = np.asarray(y_label).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(x_image)
    np.random.seed(seed)
    np.random.shuffle(y_label)

    y_vec = np.zeros((len(y_label), y_dim), dtype=np.float)
    for i, label in enumerate(y_label):
        y_vec[i, y_label[i]] = 1.0

    return x_image, y_vec


if __name__ == '__main__':
    x_image, y_label = load_data(50)
    print x_image
