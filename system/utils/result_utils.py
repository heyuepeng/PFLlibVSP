# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import h5py
import numpy as np
import os


def average_data(algorithm="", dataset="", goal="", times=10):
    test_mae, test_rmse = get_all_results_for_one_algo(algorithm, dataset, goal, times)

    min_mae = []
    min_rmse = []
    for i in range(times):
        min_mae.append(test_mae[i].min())
        min_rmse.append(test_rmse[i].min())

    print("std for lowest mae:", np.std(min_mae))
    print("mean for lowest mae:", np.mean(min_mae))
    print("std for lowest rmse:", np.std(min_rmse))
    print("mean for lowest rmse:", np.mean(min_rmse))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10):
    test_mae = []
    test_rmse = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        mae, rmse = read_data_then_delete(file_name, delete=False)
        test_mae.append(np.array(mae))
        test_rmse.append(np.array(rmse))

    return test_mae, test_rmse


def read_data_then_delete(file_name, delete=False):
    file_path = "../results/" + file_name + ".h5"

    with h5py.File(file_path, 'r') as hf:
        rs_test_mae = np.array(hf.get('rs_test_mae'))
        rs_test_rmse = np.array(hf.get('rs_test_rmse'))

    if delete:
        os.remove(file_path)

    print("Length MAE: ", len(rs_test_mae))
    print("Length RMSE: ", len(rs_test_rmse))

    return rs_test_mae, rs_test_rmse