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

import numpy as np
import time
from flcore.clients.clientditto import clientDitto
from flcore.servers.serverbase import Server
from threading import Thread


class Ditto(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientDitto)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global models")
                self.evaluate()

            if i%self.eval_gap == 0:
                print("\nEvaluate personalized models")
                self.evaluate_personalized()

            for client in self.selected_clients:
                client.ptrain()
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(mae_lss=self.rs_test_mae, patience=self.patience,
                                                   improvement_threshold=self.improvement_threshold):
                break

        print("\nLowest MAE.")
        print(min(self.rs_test_mae))
        print("\nLowest RMSE.")
        print(min(self.rs_test_rmse))
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientDitto)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


    def test_metrics_personalized(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_mae = []
        tot_rmse = []
        for c in self.clients:
            total_mae, total_rmse, total_samples = c.test_metrics()
            tot_mae.append(total_mae)
            tot_rmse.append(total_rmse)
            num_samples.append(total_samples)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_mae, tot_rmse

    def train_metrics_personalized(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics_personalized()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate_personalized(self, mae=None, loss=None):
        stats = self.test_metrics_personalized()
        stats_train = self.train_metrics_personalized()

        test_mae = sum(stats[2]) * 1.0 / sum(stats[1])
        test_rmse = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        maes = [a / n for a, n in zip(stats[2], stats[1])]
        rmses = [a / n for a, n in zip(stats[3], stats[1])]

        if mae == None:
            self.rs_test_mae.append(test_mae)
        else:
            mae.append(test_mae)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test MAE: {:.4f}".format(test_mae))
        print("Averaged Test RMSE: {:.4f}".format(test_rmse))
        # self.print_(test_mae, train_mae, train_loss)
        print("Std Test MAE: {:.4f}".format(np.std(maes)))
        print("Std Test RMSE: {:.4f}".format(np.std(rmses)))