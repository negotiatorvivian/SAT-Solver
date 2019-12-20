# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.

# PDP_solver_trainer.py : Implements a factor graph trainer for various types of PDP SAT solvers.

import numpy as np
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from pdp.factorgraph import FactorGraphTrainerBase
from pdp.nn import solver, util
from pdp.transformer import *

import warnings

warnings.filterwarnings('ignore')


##########################################################################################################################


class Perceptron(nn.Module):
    """Implements a 1-layer perceptron."""

    def __init__(self, input_dimension, hidden_dimension, output_dimension):
        super(Perceptron, self).__init__()
        self._layer1 = nn.Linear(input_dimension, hidden_dimension)
        self._layer2 = nn.Linear(hidden_dimension, output_dimension, bias = False)

    def forward(self, inp):
        return F.sigmoid(self._layer2(F.relu(self._layer1(inp), inplace = True)))


##########################################################################################################################

class SatFactorGraphTrainer(FactorGraphTrainerBase):
    """Implements a factor graph trainer for various types of PDP SAT solvers."""

    def __init__(self, config, use_cuda, logger):
        super(SatFactorGraphTrainer, self).__init__(config = config,
                                                    has_meta_data = False, error_dim = config['error_dim'], loss = None,
                                                    evaluator = nn.L1Loss(), use_cuda = use_cuda, logger = logger)

        self._eps = 1e-8 * torch.ones(1, device = self._device)
        self._loss_evaluator = util.SatLossEvaluator(alpha = self._config['exploration'], device = self._device)
        self._cnf_evaluator = util.SatCNFEvaluator(device = self._device)
        self._counter = 0
        self._max_coeff = 10.0
        self._config = config

    def _build_graph(self, config):
        model_list = []

        if config['model_type'] == 'np-nd-np':
            model_list += [solver.NeuralPropagatorDecimatorSolver(device = self._device, name = config['model_name'],
                                                                  edge_dimension = config['edge_feature_dim'],
                                                                  meta_data_dimension = config['meta_feature_dim'],
                                                                  propagator_dimension = config['hidden_dim'],
                                                                  decimator_dimension = config['hidden_dim'],
                                                                  mem_hidden_dimension = config['mem_hidden_dim'],
                                                                  agg_hidden_dimension = config['agg_hidden_dim'],
                                                                  mem_agg_hidden_dimension = config['mem_agg_hidden_dim'],
                                                                  prediction_dimension = config['prediction_dim'],
                                                                  variable_classifier = Perceptron(config['hidden_dim'],
                                                                                                   config['classifier_dim'],
                                                                                                   config['prediction_dim']),
                                                                  function_classifier = None,
                                                                  dropout = config['dropout'],
                                                                  local_search_iterations = config['local_search_iteration'],
                                                                  epsilon = config['epsilon'])]

        elif config['model_type'] == 'p-nd-np':
            model_list += [solver.NeuralSurveyPropagatorSolver(device = self._device, name = config['model_name'],
                                                               edge_dimension = config['edge_feature_dim'],
                                                               meta_data_dimension = config['meta_feature_dim'],
                                                               decimator_dimension = config['hidden_dim'],
                                                               mem_hidden_dimension = config['mem_hidden_dim'],
                                                               agg_hidden_dimension = config['agg_hidden_dim'],
                                                               mem_agg_hidden_dimension = config['mem_agg_hidden_dim'],
                                                               prediction_dimension = config['prediction_dim'],
                                                               variable_classifier = Perceptron(config['hidden_dim'],
                                                                                                config[
                                                                                                    'classifier_dim'],
                                                                                                config[
                                                                                                    'prediction_dim']),
                                                               function_classifier = None, dropout = config['dropout'],
                                                               local_search_iterations = config[
                                                                   'local_search_iteration'],
                                                               epsilon = config['epsilon'])]

        elif config['model_type'] == 'np-d-np':
            model_list += [solver.NeuralSequentialDecimatorSolver(device = self._device, name = config['model_name'],
                                                                  edge_dimension = config['edge_feature_dim'],
                                                                  meta_data_dimension = config['meta_feature_dim'],
                                                                  propagator_dimension = config['hidden_dim'],
                                                                  decimator_dimension = config['hidden_dim'],
                                                                  mem_hidden_dimension = config['mem_hidden_dim'],
                                                                  agg_hidden_dimension = config['agg_hidden_dim'],
                                                                  mem_agg_hidden_dimension = config[
                                                                      'mem_agg_hidden_dim'],
                                                                  classifier_dimension = config['classifier_dim'],
                                                                  dropout = config['dropout'],
                                                                  tolerance = config['tolerance'],
                                                                  t_max = config['t_max'],
                                                                  local_search_iterations = config[
                                                                      'local_search_iteration'],
                                                                  epsilon = config['epsilon'])]

        elif config['model_type'] == 'p-d-p':
            model_list += [solver.SurveyPropagatorSolver(device = self._device, name = config['model_name'],
                                                         tolerance = config['tolerance'], t_max = config['t_max'],
                                                         local_search_iterations = config['local_search_iteration'],
                                                         epsilon = config['epsilon'])]

        elif config['model_type'] == 'walk-sat':
            model_list += [solver.WalkSATSolver(device = self._device, name = config['model_name'],
                                                iteration_num = config['local_search_iteration'],
                                                epsilon = config['epsilon'])]

        elif config['model_type'] == 'reinforce':
            model_list += [solver.ReinforceSurveyPropagatorSolver(device = self._device, name = config['model_name'],
                                                                  pi = config['pi'], decimation_probability = config[
                    'decimation_probability'],
                                                                  local_search_iterations = config[
                                                                      'local_search_iteration'],
                                                                  epsilon = config['epsilon'])]

        # if config['verbose']:
        #     self._logger.info("The model parameter count is %d." % model_list[0].parameter_count())
        #     self._logger.info("The model list is %s." % model_list)

        return model_list

    def _compute_loss(self, model, loss, prediction, label, graph_map, batch_variable_map,
                      batch_function_map, edge_feature, meta_data):

        return self._loss_evaluator(variable_prediction = prediction[0], label = label, graph_map = graph_map,
                                    batch_variable_map = batch_variable_map, batch_function_map = batch_function_map,
                                    edge_feature = edge_feature, meta_data = meta_data,
                                    global_step = model._global_step,
                                    eps = self._eps, max_coeff = self._max_coeff,
                                    loss_sharpness = self._config['loss_sharpness'])

    def _compute_evaluation_metrics(self, model, evaluator, prediction, label, graph_map,
                                    batch_variable_map, batch_function_map, edge_feature, meta_data):

        output, _, _, _, _ = self._cnf_evaluator(variable_prediction = prediction[0], graph_map = graph_map,
                                                 batch_variable_map = batch_variable_map,
                                                 batch_function_map = batch_function_map,
                                                 edge_feature = edge_feature, meta_data = meta_data)

        recall = torch.sum(label * ((output > 0.5).float() - label).abs()) / torch.max(torch.sum(label), self._eps)
        accuracy = evaluator((output > 0.5).float(), label).unsqueeze(0)
        loss_value = self._loss_evaluator(variable_prediction = prediction[0], label = label, graph_map = graph_map,
                                          batch_variable_map = batch_variable_map,
                                          batch_function_map = batch_function_map,
                                          edge_feature = edge_feature, meta_data = meta_data,
                                          global_step = model._global_step,
                                          eps = self._eps, max_coeff = self._max_coeff,
                                          loss_sharpness = self._config['loss_sharpness']).unsqueeze(0)

        return torch.cat([accuracy, recall, loss_value], 0)

    def _post_process_predictions(self, model, prediction, graph_map,
                                  batch_variable_map, batch_function_map, edge_feature, edge_feature_, graph_feat,
                                  label, misc_data, variable_num, function_num):
        """Formats the prediction and the output solution into JSON format."""
        message = ""
        labs = label.detach().cpu().numpy()

        res = self._cnf_evaluator(variable_prediction = prediction[0], graph_map = graph_map,
                                  batch_variable_map = batch_variable_map, batch_function_map = batch_function_map,
                                  edge_feature = edge_feature, meta_data = graph_feat)
        output, unsat_clause_num, graph_map, clause_values, edge_values = [a.detach().cpu().numpy() for a in res]
        # clause_values = [(k, int(v)) for k, v in enumerate(clause_values)]
        unsat_clause_index = [k for k, v in enumerate(clause_values) if int(v) < 1]
        clauses = {}
        j = 0
        clauses[j] = []
        var_base = 0
        func_base = function_num[j]
        for k in unsat_clause_index:
            if k >= func_base:
                var_base += variable_num[j]
                func_base += function_num[j + 1] if len(function_num) > (j + 1) else 0
                j += 1
                clauses[j] = []
            i = k
            index_list = []
            clause_list = []
            while True:
                start = index_list[-1] if len(index_list) > 0 else 0
                try:
                    index = graph_map[1].tolist().index(i, start + 1)
                except:
                    break
                index_list.append(index)
                clause_list.append(
                    (graph_map[0].tolist()[index] + 1 - var_base) * int(edge_feature_.tolist()[index][0]))
            clauses[j].append(clause_list)
        # print(clauses)
        # input_maps = []
        # for i in range(output.shape[0]):
        #     if unsat_clause_num[i] > 0:
        #         j = i
        #         input_map = []
        #         s = function_num[j - 1] if j > 0 else 0
        #         t = s + function_num[j]
        #         for k in clause_values[s:t]:
        #             j = k[0]
        #             index_list = []
        #             variable_list = []
        #             while True:
        #                 start = index_list[-1] if len(index_list) > 0 else 0
        #                 try:
        #                     index = graph_map[1].tolist().index(j, start + 1)
        #                 except:
        #                     break
        #                 variable_list.append(str(graph_map[0].tolist()[index] + 1
        #                 if int(edge_values.tolist()[index][0]) > 0 else -(graph_map[0].tolist()[index] + 1)))
        #                 index_list.append(index)
        #             if len(input_map) == 0 and i == 0:
        #                 # print(index_list, graph_map[1].tolist())
        #                 m = index_list[0] - 1 if len(index_list) > 0 else 0
        #                 variable_list.append(str(graph_map[0].tolist()[m] + 1 if int(edge_values.tolist()[m][0]) > 0
        #                                                           else -(graph_map[0].tolist()[m] + 1)))
        #             variable_list.append('0')
        #             input_map.extend(variable_list)
        #         input_map.insert(0, 'p cnf ' + str(variable_num[i]) + ' ' + str(function_num[i]))
        #         input_maps.append((' ').join(input_map))

        # unsat_clause_num = self.search_solution(input_maps, unsat_clause_num)

        for i in range(output.shape[0]):
            instance = {
                # 'ID': misc_data[i][0] if len(misc_data[i]) > 0 else "",
                # 'label': int(labs[i, 0]),
                # 'solved': int(output[i].flatten()[0] == 1),
                'unsat_clauses': int(unsat_clause_num[i].flatten()[0]),
                # 'solution': (prediction[0][batch_variable_map == i, 0].detach().cpu().numpy().flatten() > 0.5).astype(
                #     int).tolist()
            }
            # message += str(self._config['test_recurrence_num']) + '\t' + str(self._config['local_search_iteration'])+ '\t' + str(self._config['epsilon']) + '\ngraph_map\n'
            # message += str([int(v) for v in graph_map[1]]) + '\nlength of graph_map:'+ str(len(graph_map[1])) + str(graph_map[1][-10:]) + '\nclause_values\n'
            # message += str([int(v) for v in clause_values]) + '\nlength of clause_values:'+ str(len(clause_values)) + '\n'
            # message += str([(k, graph_map[1][k], graph_map[0][k], int(v)) for k, v in enumerate(edge_values)]) + '\nlength of edge_values:'+ str(len(edge_values)) + '\n'
            # message += 'length of edge_values:'+ str(len(edge_values)) + '\n'

            message += (str(instance).replace("'", '"') + "\n")
            self._counter += 1

        return message

    def _check_recurrence_termination(self, active, prediction, sat_problem):
        "De-actives the CNF examples which the model has already found a SAT solution for."

        output, _, _, _, _ = self._cnf_evaluator(variable_prediction = prediction[0],
                                                 graph_map = sat_problem._graph_map,
                                                 batch_variable_map = sat_problem._batch_variable_map,
                                                 batch_function_map = sat_problem._batch_function_map,
                                                 edge_feature = sat_problem._edge_feature,
                                                 meta_data = sat_problem._meta_data)  # .detach().cpu().numpy()

        if sat_problem._batch_replication > 1:
            real_batch = torch.mm(sat_problem._replication_mask_tuple[1], (output > 0.5).float())
            dup_batch = torch.mm(sat_problem._replication_mask_tuple[0], (real_batch == 0).float())
            active[active[:, 0], 0] = (dup_batch[active[:, 0], 0] > 0).byte()
        else:
            active[active[:, 0], 0] = (output[active[:, 0], 0] <= 0.5).byte()

    def search_solution(self, input_maps, unsat_clause_num):
        "Use other solver to find a solution"

        # input_stream = (' ').join(input_map)
        for (i, input_stream) in enumerate(input_maps):
            with open('temp_file' + str(i), 'w') as f:
                f.write(input_stream)
                # process = subprocess.run(['/home/ziwei/Downloads/MapleLCMDistChronoBT/bin/glucose_static', '-cpu-lim=120'], stdin = f, stdout = subprocess.PIPE, encoding = 'utf-8',timeout = 120)
                process = subprocess.run(['/home/ziwei/Downloads/topoSAT2/bin/glucose', '-cpu-lim=120'], stdin = f,
                                         stdout = subprocess.PIPE, encoding = 'utf-8', timeout = 120)
                if process.stdout.find('UNSATISFIABLE') >= 0:
                    # print('UNSATISFIABLE')
                    continue
                elif process.stdout.find('SATISFIABLE') >= 0:
                    # print('SATISFIABLE')
                    unsat_clause_num[i] = 0
                else:
                    continue

        return unsat_clause_num
