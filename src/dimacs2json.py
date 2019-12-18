#!/usr/bin/env python3
"""
Auxiliary script for converting sets of DIMACS files into PDP's compact JSON format.
"""

# Copyright (c) Microsoft. All rights reserved.
#
# Licensed under the MIT license. See LICENSE.md file
# in the project root for full license information.

import sys
import argparse
import random

from os import listdir
from os.path import isfile, join, split, splitext

import numpy as np


class CompactDimacs:
    "Encapsulates a CNF file given in the DIMACS format."

    def __init__(self, dimacs_file, output, propagate):

        self.propagate = propagate
        self.file_name = split(dimacs_file)[1]
        self.max_len = 0

        with open(dimacs_file, 'r') as f:
            j = 0
            for line in f:
                seg = line.split(" ")
                if seg[0] == 'c':
                    continue

                if seg[0] == 'p':
                    var_num = int(seg[2])
                    clause_num = int(seg[3])
                    self._clause_mat = np.zeros((clause_num, var_num), dtype=np.int32)

                elif len(seg) <= 1:
                    continue
                else:
                    temp = np.array(seg[:-1], dtype=np.int32)
                    self.max_len = max(self.max_len, len(temp))
                    self._clause_mat[j, np.abs(temp) - 1] = np.sign(temp)
                    j += 1

        ind = np.where(np.sum(np.abs(self._clause_mat), 1) > 0)[0]
        # self.diff = np.array([self.max_len - np.sum(self._clause_mat[i, :] != 0) for i in range(len(ind))], dtype = np.int)
        # self._clause_mat = np.concatenate((self._clause_mat, np.zeros([self._clause_mat.shape[0], max(self.diff)])), axis = 1)
        # for i in range(len(self.diff)):
        #     if self.diff[i] > 0:
        #         self._clause_mat[i, -self.diff[i]:] = 1

        self._clause_mat = self._clause_mat[ind, :]

        if propagate:
            self._clause_mat = self._propagate_constraints(self._clause_mat)

        self._output = output


    def _propagate_constraints(self, clause_mat):
        n = clause_mat.shape[0]
        if n < 2:
            return clause_mat

        length = np.tile(np.sum(np.abs(clause_mat), 1), (n, 1))
        intersection_len = np.matmul(clause_mat, np.transpose(clause_mat))

        temp = intersection_len == np.transpose(length)
        temp *= np.tri(*temp.shape, k=-1, dtype=bool)
        flags = np.logical_not(np.any(temp, 0))

        clause_mat = clause_mat[flags, :]

        n = clause_mat.shape[0]
        if n < 2:
            return clause_mat

        length = np.tile(np.sum(np.abs(clause_mat), 1), (n, 1))
        intersection_len = np.matmul(clause_mat, np.transpose(clause_mat))

        temp = intersection_len == length
        temp *= np.tri(*temp.shape, k=-1, dtype=bool)
        flags = np.logical_not(np.any(temp, 1))

        return clause_mat[flags, :]

    def to_json(self, base = 0):
        clause_num, var_num = self._clause_mat.shape
        var_num = var_num + base
        ind = np.nonzero(self._clause_mat)
        negative_sample_set = np.argwhere(self._clause_mat == 0)
        relations = ''
        node_type = ''
        test_str = ''
        validate_str = ''
        type_1 = set()
        type_2 = set()
        for i, item in enumerate(ind[1]):
            # if i % 20 == 0 and i > 0:
            #     print(max(type_2), max(type_1))
            var_item = item + base
            temp = str(self._clause_mat[ind[0][i]][item]) + ' ' + str(var_item) + ' ' + str(ind[0][i] + var_num)
            # relations += temp + '\n'
            seed = random.random()
            if seed > 0.95:
                validate_str += temp + ' 1 \n'
                negative_relation = negative_sample_set[random.randint(0, len(negative_sample_set) - 1)]
                edge_type = '1 ' if random.random() > 0.5 else '-1 '
                validate_str += edge_type + str(negative_relation[1] + base) + ' ' + str(negative_relation[0] + var_num) + ' 0\n'
            elif seed > 0.85:
                test_str += temp + ' 1 \n'
                negative_relation = negative_sample_set[random.randint(0, len(negative_sample_set) - 1)]
                edge_type = '1 ' if random.random() > 0.5 else '-1 '
                test_str += edge_type + str(negative_relation[1] + base) + ' ' + str(negative_relation[0] + var_num) + ' 0\n'
            else:
                relations += temp + '\n'

            if var_item in type_1:
                pass
            else:
                node_type += str(var_item) + ' 0\n'
                type_1.add(var_item)
            if (ind[0][i] + var_num) in type_2:
                pass
            else:
                node_type += str(ind[0][i] + var_num) + ' 1\n'
                type_2.add(ind[0][i] + var_num)
        # return [[var_num, clause_num], list(((ind[1] + 1) * self._clause_mat[ind]).astype(np.int)), list(ind[0] + 1), self._output]
        return node_type, relations, test_str, validate_str, max(type_2)


def convert_directory(dimacs_dir, output_file, propagate, only_positive=False):
    file_list = [join(dimacs_dir, f) for f in listdir(dimacs_dir) if isfile(join(dimacs_dir, f))]
    try:
        base = 0
        temp = []
        output_file_list = output_file.split('$')
        if len(output_file_list) < 4:
            raise Exception('Invalid argument output_file', output_file)
        node_type_file = open(output_file_list[0], 'w')
        train_file = open(output_file_list[1], 'w')
        test_file = open(output_file_list[2], 'w')
        validate_file = open(output_file_list[3], 'w')
        for i in range(len(file_list)):
            name, ext = splitext(file_list[i])
            ext = ext.lower()

            if ext != '.dimacs' and ext != '.cnf':
                continue

            # label = float(name[-1]) if name[-1].isdigit() else -1
            # label = 1 if name.split('=')[-1] == 'True' else -1
            label = -1

            if only_positive and label == 0:
                continue

            bc = CompactDimacs(file_list[i], label, propagate)
            # f.write(str(bc.to_json()).replace("'", '"') + '\n')
            # print("Generating JSON input file: %6.2f%% complete..." % (
            #     (i + 1) * 100.0 / len(file_list)), end='\r', file=sys.stderr)
            node_type, relations, test_str, validate_str, base = bc.to_json(base)
            temp.append(base)
            train_file.write(relations)
            node_type_file.write(node_type)
            test_file.write(test_str)
            validate_file.write(validate_str)

        node_type_file.close()
        train_file.close()
        test_file.close()
        validate_file.close()
        # print(temp, len(temp))
    except Exception as e:
        print(e)


def convert_file(file_name, output_file, propagate):
    with open(output_file, 'w') as f:
        if len(file_name) < 8:
            label = -1
        else:
            temp = file_name[-8]
            label = float(temp) if temp.isdigit() else -1

        bc = CompactDimacs(file_name, label, propagate)
        f.write(str(bc.to_json()).replace("'", '"') + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', action='store', type=str)
    parser.add_argument('out_file', action='store', type=str)
    parser.add_argument('-s', '--simplify', help='Propagate binary constraints', required=False, action='store_true', default=False)
    parser.add_argument('-p', '--positive', help='Output only positive examples', required=False, action='store_true', default=False)
    args = vars(parser.parse_args())

    convert_directory(args['in_dir'], args['out_file'], args['simplify'], args['positive'])
