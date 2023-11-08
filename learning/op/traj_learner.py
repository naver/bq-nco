"""
BQ-NCO
Copyright (c) 2023-present NAVER Corp.
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license
"""

import time
import torch
from torch import nn
from learning.op.decoding import decode
from utils.misc import do_lr_decay, EpochMetrics

DEBUG_NUM_BATCHES = 3

class TrajectoryLearner:

    def __init__(self, args, net, module, device, data_iterator, optimizer=None, checkpointer=None):
        # same supervisor is used for training and testing, during testing we do not have optimizer, mlflow etc.

        self.net = net
        self.module = module
        self.device = device
        self.data_iterator = data_iterator
        self.optimizer = optimizer
        self.checkpointer = checkpointer
        self.beam_size = args.beam_size
        self.knns = args.knns

        self.output_dir = args.output_dir
        self.test_only = args.test_only

        self.debug = args.debug

        if not args.test_only:
            try:
                self.test_every = args.test_every if args.test_every > 0 else None
            except AttributeError:
                self.test_every = None
            self.decay_rate = args.decay_rate
            self.decay_every = args.decay_every
            self.loss = nn.CrossEntropyLoss()
            self.best_current_val_metric = float('inf')
            self.epoch_done = 0
            self.nb_epochs = args.nb_total_epochs

    def train(self):
        assert not self.test_only

        for _ in range(self.nb_epochs):
            # Train one epoch
            start = time.time()
            self.net.train()
            epoch_metrics_train = EpochMetrics()

            batch_num = 0
            for data in self.data_iterator.train_trajectories:

                self.optimizer.zero_grad()
                node_coords, node_values, upper_bounds, dist_matrices, _ = self.prepare_batch(data)
                inputs = torch.cat([node_coords, node_values.unsqueeze(-1),
                                    upper_bounds.unsqueeze(-1).repeat(1, node_coords.shape[1]).unsqueeze(-1)],
                                   dim=-1)
                output_scores = self.net(inputs, dist_matrices=dist_matrices)
                loss = self.loss(output_scores,
                                 torch.ones((output_scores.shape[0]), dtype=torch.long,
                                            device=output_scores.device))

                loss.backward()
                epoch_metrics_train.update({"training loss": loss})
                self.optimizer.step()

                if batch_num == DEBUG_NUM_BATCHES and self.debug:
                    break
                batch_num += 1

            metrics = {f'{k}_train': v for k, v in epoch_metrics_train.get_means().items()}
            print("[EPOCH {:03d}] Time: {:.3f}s ".format(self.epoch_done, time.time() - start))
            for k, v in metrics.items():
                print(k, f"{v:.5f}")

            # Val test
            val_metrics = self.val_test("val")

            if val_metrics["opt_gap_val"] < self.best_current_val_metric:
                # monitoring on current trajectories, in order to see if we are training enough on them or not
                self.best_current_val_metric = val_metrics["opt_gap_val"]
                self.checkpointer.save(self.module, None, 'best')  # only model

            print("[EPOCH {:03d}] Time: {:.3f}s ".format(self.epoch_done, time.time() - start))

            # test
            if self.test_every is not None:
                if self.epoch_done % self.test_every == 0:
                    self.save_model("current")
                    self.load_model("best")
                    self.val_test("test")
                    self.load_model("current")
                    self.remove_model("current")

            self.epoch_done += 1

            # lr decay
            if self.epoch_done % self.decay_every == 0:
                do_lr_decay(self.optimizer, self.decay_rate)

    def val_test(self, what="test"):
        if what == "test":
            dataloader = self.data_iterator.test_trajectories
        else:
            dataloader = self.data_iterator.val_trajectories

        self.net.eval()
        epoch_metrics = EpochMetrics()
        with torch.no_grad():

            for batch_num, data in enumerate(dataloader):
                val_test_metrics = self.get_minibatch_val_test_metrics(data)
                epoch_metrics.update(val_test_metrics)
                if batch_num == DEBUG_NUM_BATCHES and self.debug:
                    break

        res = {f'{k}_{what}': v for k, v in epoch_metrics.get_means().items()}

        for k, v in res.items():
            print(k, f"{v:.3f}")

        return res

    def get_minibatch_val_test_metrics(self, data):
        metrics = {}
        # autoregressive decoding
        node_coords, node_values, upper_bounds, dist_matrices, opt_values = self.prepare_batch(data, sample=False)
        decoding_metrics = {}
        predicted_values, _ = decode(node_coords, node_values, upper_bounds, dist_matrices, self.net,
                                     self.beam_size, self.knns)

        opt_gap = 100 * ((opt_values - predicted_values) / opt_values).mean().item()

        decoding_metrics.update({"opt_gap": opt_gap}) 
        return {**metrics, **decoding_metrics}

    def load_model(self, label, allow_not_exist=False):
        assert label in ["current", "current_FULL", "best"]
        self.checkpointer.load(self.module, None, label, allow_not_exist=allow_not_exist)

    def save_model(self, label, complete=False):
        assert label in ["current", "best"]
        args = {"module": self.module}
        if not complete:
            args_ = {"optimizer": None, 'label': label}
        else:
            assert not self.eval_only
            assert label == "current"
            args_ = {"optimizer": self.optimizer,
                     "label": label+"_FULL",
                     "other": {
                         "epoch_done": self.epoch_done,
                         "best_current_val_metric": self.best_current_val_metric,
                         "data_iterator": self.data_iterator}
                     }
        self.checkpointer.save(**args, **args_)

    def remove_model(self, label):
        assert label in ["current", "best"]
        self.checkpointer.delete(label)

    def prepare_batch(self, data, sample=True):
        ks = "_s" if sample else ""
        node_coords = data[f"node_coords{ks}"].to(self.device)
        node_values = data[f"node_values{ks}"].to(self.device)
        upper_bounds = data[f"upper_bounds{ks}"].to(self.device)

        dist_matrices = data[f"dist_matrices{ks}"].to(self.device)
        collected_rewards = data[f"collected_rewards{ks}"].to(self.device)

        return node_coords, node_values, upper_bounds, dist_matrices, collected_rewards
