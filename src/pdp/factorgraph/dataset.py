# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

# factor_graph_input_pipeline.py : Defines the input pipeline for the PDP framework.

import linecache, json
import collections
from itertools import combinations
import numpy as np

import torch
import torch.utils.data as data
import scipy.sparse as sp


class DynamicBatchDivider(object):
    "Implements the dynamic batching process."

    def __init__(self, limit, hidden_dim):
        self.limit = limit
        self.hidden_dim = hidden_dim

    def divide(self, variable_num, function_num, graph_map, edge_feature, graph_feature, label, misc_data):
        batch_size = len(variable_num)
        edge_num = [len(n) for n in edge_feature]

        graph_map_list = []
        edge_feature_list = []
        graph_feature_list = []
        variable_num_list = []
        function_num_list = []
        label_list = []
        adj_list = []
        misc_data_list = []

        if (self.limit // (max(edge_num) * self.hidden_dim)) >= batch_size:
            if graph_feature[0] is None:
                graph_feature_list = [[None]]
            else:
                graph_feature_list = [graph_feature]

            graph_map_list = [graph_map]
            edge_feature_list = [edge_feature]
            variable_num_list = [variable_num]
            function_num_list = [function_num]
            label_list = [label]
            misc_data_list = [misc_data]

        else:

            indices = sorted(range(len(edge_num)), reverse=True, key=lambda k: edge_num[k])
            sorted_edge_num = sorted(edge_num, reverse=True)

            i = 0

            while i < batch_size:
                allowed_batch_size = self.limit // (sorted_edge_num[i] * self.hidden_dim)
                # allowed_batch_size = min(allowed_batch_size, 1)
                ind = indices[i:min(i + allowed_batch_size, batch_size)]

                if graph_feature[0] is None:
                    graph_feature_list += [[None]]
                else:
                    graph_feature_list += [[graph_feature[j] for j in ind]]

                edge_feature_list += [[edge_feature[j] for j in ind]]
                variable_num_list += [[variable_num[j] for j in ind]]
                function_num_list += [[function_num[j] for j in ind]]
                graph_map_list += [[graph_map[j] for j in ind]]
                label_list += [[label[j] for j in ind]]
                misc_data_list += [[misc_data[j] for j in ind]]

                i += allowed_batch_size

        return variable_num_list, function_num_list, graph_map_list, edge_feature_list, graph_feature_list, label_list, misc_data_list


###############################################################


class FactorGraphDataset(data.Dataset):
    """Implements a PyTorch Dataset lass for reading and parsing CNFs in the JSON format from disk."""

    def __init__(self, input_file, limit, hidden_dim, max_cache_size=100000, generator=None, epoch_size=0, batch_replication=1):

        self._cache = collections.OrderedDict()
        self._generator = generator
        self._epoch_size = epoch_size
        self._input_file = input_file
        self._max_cache_size = max_cache_size

        if self._generator is None:
            with open(self._input_file, 'r') as fh_input:
                self._row_num = len(fh_input.readlines())
        # print(batch_replication)
        self.batch_divider = DynamicBatchDivider(limit // batch_replication, hidden_dim)

    def __len__(self):
        if self._generator is not None:
            return self._epoch_size
        else:
            return self._row_num

    def __getitem__(self, idx):
        if self._generator is not None:
            print('d\d')
            return self._generator.generate()

        else:
            if idx in self._cache:
                return self._cache[idx]

            line = linecache.getline(self._input_file, idx + 1)
            # line = linecache.getline(self._input_file, idx + 1)[1:-2]

            result = self._convert_line(line)

            if len(self._cache) >= self._max_cache_size:
                self._cache.popitem(last=False)

            self._cache[idx] = result
            return result

    def _convert_line(self, json_str):
        # print(json_str)
        input_data = json.loads(json_str)
        variable_num, function_num = input_data[0]

        variable_ind = np.array(input_data[1], dtype=np.int32)
        variable_ind[np.argwhere(variable_ind > variable_num)] = 0
        variable_num += 1
        edge_feature = np.sign(variable_ind)
        variable_ind = np.abs(variable_ind)
        # var_count = list(set(variable_ind))
        function_ind = np.abs(np.array(input_data[2], dtype=np.int32)) - 1
        # func_index = np.array([function_ind[j] for j in [np.argwhere(variable_ind == i) for i in var_count]])
        # iters = np.array([c for c in [list(combinations(range(len(func_index[i])), 2)) for i in range(len(func_index))]])
        # # edge_relations = []
        # edge_idx_relations = []
        #
        # try:
        #     for i in range(len(iters)):
        #         if len(iters[i]) == 0:
        #             continue
        #         temp = np.squeeze(func_index[i][np.array(iters[i])], axis = 2)
        #         # edge_relations.append(temp)
        #         edge_idx_relations += ['+'.join(map(str, list(temp[j]))) for j in range(len(temp))]
        # except:
        #     print(len(edge_idx_relations))
        # edge_idx_relations = Counter(edge_idx_relations)
        # edge_count = list(dict(edge_idx_relations).values())
        # row_col_s = [i for i in map(lambda x: x.split('+'), list(dict(edge_idx_relations).keys()))]
        # rows = np.array(np.array(row_col_s)[:,0], dtype = np.int)
        # cols = np.array(np.array(row_col_s)[:,1], dtype = np.int)
        # adj = sp.coo_matrix((edge_count, (rows.tolist(), cols.tolist())), shape = (function_num, function_num))
        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # adj = self._normalize_adj(adj + sp.eye(adj.shape[0]))
        # adj = torch.FloatTensor(np.array(adj.todense()))
        graph_map = np.stack((variable_ind, function_ind))
        alpha = float(function_num) / variable_num

        misc_data = []
        if len(input_data) > 4:
            misc_data = input_data[4]
        # result = True
        result = input_data[3] if len(input_data) > 3 else False


        return (variable_num, function_num, graph_map, edge_feature, None, float(result), misc_data)


    def dag_collate_fn(self, input_data):
        "Torch dataset loader collation function for factor graph input."
        vn, fn, gm, ef, gf, l, md = zip(*input_data)

        variable_num, function_num, graph_map, edge_feature, graph_feat, label, misc_data = \
            self.batch_divider.divide(vn, fn, gm, ef, gf, l, md)
        segment_num = len(variable_num)
        graph_feat_batch = []
        graph_map_batch = []
        batch_variable_map_batch = []
        batch_function_map_batch = []
        edge_feature_batch = []
        label_batch = []

        for i in range(segment_num):

            # Create the graph features batch
            graph_feat_batch += [None if graph_feat[i][0] is None else torch.from_numpy(np.stack(graph_feat[i])).float()]

            # Create the edge feature batch
            edge_feature_batch += [torch.from_numpy(np.expand_dims(np.concatenate(edge_feature[i]), 1)).float()]

            # Create the label batch
            label_batch += [torch.from_numpy(np.expand_dims(np.array(label[i]), 1)).float()]

            # Create the graph map, variable map and function map batches
            g_map_b = np.zeros((2, 0), dtype=np.int32)
            v_map_b = np.zeros(0, dtype=np.int32)
            f_map_b = np.zeros(0, dtype=np.int32)
            variable_ind = 0
            function_ind = 0

            for j in range(len(graph_map[i])):
                graph_map[i][j][0, :] += variable_ind
                graph_map[i][j][1, :] += function_ind
                g_map_b = np.concatenate((g_map_b, graph_map[i][j]), axis=1)

                v_map_b = np.concatenate((v_map_b, np.tile(j, variable_num[i][j])))
                f_map_b = np.concatenate((f_map_b, np.tile(j, function_num[i][j])))

                variable_ind += variable_num[i][j]
                function_ind += function_num[i][j]

                # index = torch.LongTensor([adj[i][j].row, adj[i][j].col])
                # value = torch.IntTensor(adj[i][j].data)
                # adj_obj = torch.sparse.FloatTensor(index, value).to_dense()
                # adj_batch.append(adj_obj)

            graph_map_batch += [torch.from_numpy(g_map_b).int()]
            batch_variable_map_batch += [torch.from_numpy(v_map_b).int()]
            batch_function_map_batch += [torch.from_numpy(f_map_b).int()]

        return graph_map_batch, batch_variable_map_batch, batch_function_map_batch, edge_feature_batch, graph_feat_batch, label_batch, misc_data, variable_num, function_num
    
    @staticmethod
    def get_loader(input_file, limit, hidden_dim, batch_size, shuffle, num_workers,
                    max_cache_size=100000, use_cuda=True, generator=None, epoch_size=0, batch_replication=1):
        """"Return the torch dataset loader object for the input."""

        dataset = FactorGraphDataset(
            input_file=input_file,
            limit=limit,
            hidden_dim=hidden_dim,
            max_cache_size=max_cache_size,
            generator=generator, 
            epoch_size=epoch_size, 
            batch_replication=batch_replication)
        # print('dataset', dataset, input_file)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=dataset.dag_collate_fn,
            pin_memory=use_cuda)

        return data_loader





