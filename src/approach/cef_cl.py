import copy
import math
import itertools
import random
from torch.utils.data import Subset
import torch
import warnings
import time
import numpy as np
from torchvision.utils import save_image
from torch import nn
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader
from .LAS_utils import mixup_data, mixup_criterion, H2T, LearnableWeightScaling, iFFM,iFFM_avg
from .cmo import rand_bbox, cutmix_data, getbox_new
import datasets.data_loader as stage2_utils
import itertools
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the Learning a Unified Classifier Incrementally via Rebalancing (LUCI) approach
    described in http://dahua.me/publications/dhl19_increclass.pdf
    Original code available at https://github.com/hshustc/CVPR19_Incremental_Learning
    """

    # Sec. 4.1: "we used the method proposed in [29] based on herd selection" and "first one stores a constant number of
    # samples for each old class (e.g. R_per=20) (...) we adopt the first strategy"
    def __init__(self, model, device, nepochs=160, lr=0.1, lr_min=1e-4, lr_factor=10, lr_patience=8, clipgrad=10000,
                 momentum=0.9, wd=5e-4, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger=None, exemplars_dataset=None, lamb=10., lamb_mr=1., dist=0.5, K=2,
                 remove_less_forget=False, remove_margin_ranking=False, remove_adapt_lamda=False):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)

        if nepochs == 100:
            self.lamb = 10.
        else:
            self.lamb = 5.
        self.lamb_mr = lamb_mr
        self.dist = dist
        self.K = K
        self.less_forget = not remove_less_forget
        self.margin_ranking = not remove_margin_ranking
        self.adapt_lamda = not remove_adapt_lamda
        self.lws_models = torch.nn.ModuleList()
        self.stage2epoch = 30
        self.lamda = self.lamb
        self.ref_model = None
        self.warmup_loss = self.warmup_luci_loss
        # cmo parameters
        self.weighted_alpha = 1
        self.mixup_prob = 0.5
        self.beta = 0.8
        self.alpha = 0.1
        # stage2
        self.avgpool = self.model.model.avgpool
        self.layer4 = self.model.model.layer4
        self.iffm = None
        # LUCIR is expected to be used with exemplars. If needed to be used without exemplars, overwrite here the
        # `_get_optimizer` function with the one in LwF and update the criterion
        have_exemplars = self.exemplars_dataset.max_num_exemplars + self.exemplars_dataset.max_num_exemplars_per_class
        if not have_exemplars:
            warnings.warn("Warning: LUCIR is expected to use exemplars. Check documentation.")

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        # Sec. 4.1: "lambda base is set to 5 for CIFAR100 and 10 for ImageNet"
        parser.add_argument('--lamb', default=5., type=float, required=False,
                            help='Trade-off for distillation loss (default=%(default)s)')
        # Loss weight for the Inter-Class separation loss constraint, set to 1 in the original code
        parser.add_argument('--lamb-mr', default=1., type=float, required=False,
                            help='Trade-off for the MR loss (default=%(default)s)')
        # Sec 4.1: "m is set to 0.5 for all experiments"
        parser.add_argument('--dist', default=.5, type=float, required=False,
                            help='Margin threshold for the MR loss (default=%(default)s)')
        # Sec 4.1: "K is set to 2"
        parser.add_argument('--K', default=2, type=int, required=False,
                            help='Number of "new class embeddings chosen as hard negatives '
                                 'for MR loss (default=%(default)s)')
        # Flags for ablating the approach
        parser.add_argument('--remove-less-forget', action='store_true', required=False,
                            help='Deactivate Less-Forget loss constraint(default=%(default)s)')
        parser.add_argument('--remove-margin-ranking', action='store_true', required=False,
                            help='Deactivate Inter-Class separation loss constraint (default=%(default)s)')
        parser.add_argument('--remove-adapt-lamda', action='store_true', required=False,
                            help='Deactivate adapting lambda according to the number of classes (default=%(default)s)')
        return parser.parse_known_args(args)


    def _get_optimizer(self, lr=0, stage2=False):
        """Returns the optimizer"""
        if stage2:
            if lr == 0:
                lr = self.lr
            if self.less_forget:
                # Don't update heads when Less-Forgetting constraint is activated (from original code)
                params = list(self.model.heads[-1].parameters())
            else:
                params = self.model.heads.parameters()
            optimizer = torch.optim.SGD([{"params": params}, {"params": self.lws_models.parameters()},
                                    {"params": self.iffm.parameters()}],
                                   lr, weight_decay=self.wd, momentum=self.momentum)
            print(f"Stage 2 Trainable Parameters: {self.count_trainable_parameters(optimizer):,}")
            return optimizer
        else:
            if lr == 0:
                lr = self.lr
            if self.less_forget:
                # Don't update heads when Less-Forgetting constraint is activated (from original code)
                params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
            else:
                params = self.model.parameters()
            # return torch.optim.SGD(params,lr, weight_decay=self.wd, momentum=self.momentum)
            optimizer = torch.optim.SGD([{"params": params}, {"params": self.lws_models.parameters()}],
                                   lr, weight_decay=self.wd, momentum=self.momentum)
            # 打印第一阶段参数量
            print(f"Stage 1 Trainable Parameters: {self.count_trainable_parameters(optimizer):,}")
            return optimizer

    def count_trainable_parameters(self,optimizer):
        """统计优化器中所有可训练参数的总量"""
        total_params = 0
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                if param.requires_grad:
                    total_params += param.numel()
        return total_params

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""
        lws_model = LearnableWeightScaling(num_classes=self.model.task_cls[t]).to(self.device)
        self.lws_models.append(lws_model)
        # self.iffm = iFFM(channels=512).to(self.device)
        self.iffm = iFFM_avg(channels=512).to(self.device)
        if t == 0:
            # Sec. 4.1: "the ReLU in the penultimate layer is removed to allow the features to take both positive and
            # negative values"
            if self.model.model.__class__.__name__ == 'ResNet':
                old_block = self.model.model.layer3[-1]
                self.model.model.layer3[-1] = BasicBlockNoRelu(old_block.conv1, old_block.bn1, old_block.relu,
                                                               old_block.conv2, old_block.bn2, old_block.downsample)
            else:
                warnings.warn("Warning: ReLU not removed from last block.")
        # Changes the new head to a CosineLinear
        self.model.heads[-1] = CosineLinear(self.model.heads[-1].in_features, self.model.heads[-1].out_features)
        self.model.to(self.device)
        if t > 0:
            # Share sigma (Eta in paper) between all the heads
            self.model.heads[-1].sigma = self.model.heads[-2].sigma
            # Fix previous heads when Less-Forgetting constraint is activated (from original code)
            if self.less_forget:
                for h in self.model.heads[:-1]:
                    for param in h.parameters():
                        param.requires_grad = False
                self.model.heads[-1].sigma.requires_grad = True
            # Eq. 7: Adaptive lambda
            if self.adapt_lamda:
                self.lamda = self.lamb * math.sqrt(sum([h.out_features for h in self.model.heads[:-1]])
                                                   / self.model.heads[-1].out_features)
        # The original code has an option called "imprint weights" that seems to initialize the new head.
        # However, this is not mentioned in the paper and doesn't seem to make a significant difference.
        super().pre_train_process(t, trn_loader)

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)
        cls_num_dict = self.get_cls_num_list(trn_loader)
        print('cls num dict:')
        print(cls_num_dict)
        out_size = len(cls_num_dict)
        #self.fc = nn.Linear(1024, out_size).to(self.device)
        cls_num_list = [cls_num_dict.get(i, 0) for i in range(max(cls_num_dict.keys()) + 1)]
        # get weight trn_lodaer
        cls_weight = 1.0 / (np.array(cls_num_list) ** self.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        samples_weight_loader = torch.utils.data.DataLoader(trn_loader.dataset,
                                                 batch_size=trn_loader.batch_size,
                                                 shuffle =False,
                                                 num_workers=trn_loader.num_workers,
                                                 pin_memory=trn_loader.pin_memory)
        samples_weight = self.get_samples_weight(cls_weight, samples_weight_loader)
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        # print(samples_weight)
        weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight),
                                                                  replacement=True)
        weighted_train_loader = torch.utils.data.DataLoader(trn_loader.dataset,
                                                            batch_size=trn_loader.batch_size,
                                                            num_workers=trn_loader.num_workers,
                                                            pin_memory=trn_loader.pin_memory,
                                                            sampler=weighted_sampler)

        if len(trn_loader.dataset) > 10 * self.model.task_cls[t]:
            # add exemplars to train_loader

            lr = self.lr
            best_loss = np.inf
            patience = self.lr_patience

            self.optimizer = self._get_optimizer(lr)

            # for idx, h in enumerate(self.model.heads):
            #     print(f"Classifier {idx + 1}:")
            #     for name, param in h.named_parameters():
            #         print(f"Parameter name: {name}, Shape: {param.shape}")
            #         print(f"Parameter requires_grad:")
            #         print(param.requires_grad)
            #         print(f"Parameter value:")
            #         print(param)
            #
            #         print("\n")

            # Loop epochs
            for e in range(self.nepochs):
                # Train
                clock0 = time.time()
                self.train_epoch(t, trn_loader, weighted_train_loader)
                clock1 = time.time()
                if self.eval_on_train:
                    train_loss, train_acc, _ = self.eval(t, trn_loader)
                    clock2 = time.time()
                    print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                        e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                    self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                    self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
                else:
                    print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

                # Valid
                clock3 = time.time()
                valid_loss, valid_acc, _ = self.eval(t, val_loader)
                clock4 = time.time()
                print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    clock4 - clock3, valid_loss, 100 * valid_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

                # Adapt learning rate - patience scheme - early stopping regularization
                if valid_loss < best_loss:
                    # if the loss goes down, keep it as the best model and end line with a star ( * )
                    best_loss = valid_loss
                    patience = self.lr_patience
                    print(' *', end='')
                if e + 1 in self.model.schedule_step:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    self.optimizer.param_groups[0]['lr'] = lr
                    # self.model.set_state_dict(best_model)
                self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
                print()
            self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)
            balance_sampler = stage2_utils.ClassAwareSampler(trn_loader.dataset)
            balanced_trn_loader = DataLoader(trn_loader.dataset,
                                             batch_size=trn_loader.batch_size,
                                             shuffle=False,
                                             num_workers=trn_loader.num_workers,
                                             pin_memory=trn_loader.pin_memory,
                                             sampler=balance_sampler)

            # for idx, h in enumerate(self.model.heads):
            #     print(f"Classifier {idx + 1}:")
            #     for name, param in h.named_parameters():
            #         print(f"Parameter name: {name}, Shape: {param.shape}")
            #         print(f"Parameter requires_grad:")
            #         print(param.requires_grad)
            #         print(f"Parameter value:")
            #         print(param)
            #
            #         print("\n")

            """stage2"""
            lr = 0.1
            best_loss = np.inf
            patience = self.lr_patience
            self.optimizer = self._get_optimizer(lr, stage2=True)
            for e in range(self.stage2epoch):
                # Train
                clock0 = time.time()
                self.train_epoch_stage2(t, balanced_trn_loader, trn_loader)
                # self.train_epoch_stage2(t, trn_loader)
                clock1 = time.time()
                if self.eval_on_train:
                    train_loss, train_acc, _ = self.eval(t, trn_loader)
                    clock2 = time.time()
                    print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |stage2'.format(
                        e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                    self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                    self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
                else:
                    print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |stage2'.format(e + 1, clock1 - clock0),
                          end='')

                # Valid
                clock3 = time.time()
                valid_loss, valid_acc, _ = self.eval(t, val_loader)
                clock4 = time.time()
                print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    clock4 - clock3, valid_loss, 100 * valid_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

                # Adapt learning rate - patience scheme - early stopping regularization
                if valid_loss < best_loss:
                    # if the loss goes down, keep it as the best model and end line with a star ( * )
                    best_loss = valid_loss
                    patience = 10
                    print(' *', end='')
                self.adjust_learning_rate_stage_2(e + 1)
                self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
                print()

            # for idx, h in enumerate(self.model.heads):
            #     print(f"Classifier {idx + 1}:")
            #     for name, param in h.named_parameters():
            #         print(f"Parameter name: {name}, Shape: {param.shape}")
            #         print(f"Parameter requires_grad:")
            #         print(param.requires_grad)
            #         print(f"Parameter value:")
            #         print(param)
            #         print("\n")

        else:
            balance_sampler = stage2_utils.ClassAwareSampler(trn_loader.dataset)
            balanced_trn_loader = DataLoader(trn_loader.dataset,
                                             batch_size=trn_loader.batch_size,
                                             shuffle=False,
                                             num_workers=trn_loader.num_workers,
                                             pin_memory=trn_loader.pin_memory,
                                             sampler=balance_sampler)
            lr = 0.1
            best_loss = np.inf
            patience = self.lr_patience
            self.optimizer = self._get_optimizer(lr, stage2=True)
            for e in range(self.stage2epoch):
                # Train
                clock0 = time.time()
                self.train_epoch_stage2(t, balanced_trn_loader, trn_loader)
                # self.train_epoch_stage2(t, trn_loader)
                clock1 = time.time()
                if self.eval_on_train:
                    train_loss, train_acc, _ = self.eval(t, trn_loader)
                    clock2 = time.time()
                    print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |stage2'.format(
                        e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                    self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                    self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
                else:
                    print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |stage2'.format(e + 1, clock1 - clock0),
                          end='')

                # Valid
                clock3 = time.time()
                valid_loss, valid_acc, _ = self.eval(t, val_loader)
                clock4 = time.time()
                print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    clock4 - clock3, valid_loss, 100 * valid_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

                # Adapt learning rate - patience scheme - early stopping regularization
                if valid_loss < best_loss:
                    # if the loss goes down, keep it as the best model and end line with a star ( * )
                    best_loss = valid_loss
                    patience = 10
                    print(' *', end='')
                self.adjust_learning_rate_stage_2(e + 1)
                self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
                print()
            self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

        # EXEMPLAR MANAGEMENT -- select training subset

    def get_data_distribution(self, dataset):
        targets = []
        num_samples = []
        for x, y in dataset:
            targets.append(y)
        num_classes = len(np.unique(targets))
        cls_data_list = [list() for _ in range(num_classes)]
        for i, label in enumerate(targets):
            cls_data_list[label].append(i)
        for a in cls_data_list:
            num_samples.append(len(a))
        # print(num_samples)
        return num_samples

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward current model
                outputs = self.model(images.to(self.device))
                # if t!=0:
                for idx in range(len(outputs)):
                    # output.append(self.model.heads[idx](feat.detach()))
                    outputs[idx] = self.lws_models[idx](outputs[idx])
                loss = self.criterion(t, outputs, targets.to(self.device))
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        # Make the old model return outputs without the sigma (eta in paper) factor
        for h in self.ref_model.heads:
            h.train()
        self.ref_model.freeze_all()

    def train_epoch(self, t, trn_loader, weighted_train_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        weighted_train_loader = iter(weighted_train_loader)
        for images, targets in trn_loader:
            images, targets = images.to(self.device), targets.to(self.device)
            try:
                images2, targets2 = next(weighted_train_loader)
            except:
                weighted_train_loader = iter(weighted_train_loader)
                images2, targets2 = next(weighted_train_loader)
            images2 = images2[:images.size()[0]]
            targets2 = targets2[:targets.size()[0]]
            images2, targets2 = images2.to(self.device), targets2.to(self.device)
            r = np.random.rand(1)
            lwsoutputs = []
            ref_outputs = None
            ref_features = None
            if r < self.mixup_prob:  # Resize mix
                rate = random.uniform(self.alpha, self.beta)
                data_small = F.interpolate(images2, scale_factor=rate)
                m1, n1, m2, n2 = getbox_new(images.size(), data_small.shape[2:])
                for i in range(images.shape[0]):
                    images[i, :, m1[i]:m2[i], n1[i]:n2[i]] = data_small[i]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - (data_small.size()[-1] * data_small.size()[-2]) / (images.size()[-1] * images.size()[-2])

                # Forward current model
                outputs, features = self.model(images, return_features=True)
                if t > 0:
                    ref_outputs, ref_features = self.ref_model(images, return_features=True)
                loss = self.criterion(t, outputs, targets, ref_outputs, features, ref_features, lwsoutputs) * lam + \
                       self.criterion(t, outputs, targets2, ref_outputs, features, ref_features, lwsoutputs) * (
                               1. - lam)
            else:
                outputs, features = self.model(images, return_features=True)
                if t > 0:
                    ref_outputs, ref_features = self.ref_model(images, return_features=True)
                loss = self.criterion(t, outputs, targets, ref_outputs, features, ref_features, lwsoutputs)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train_epoch_stage2(self, t, train_loader, trn_loader):
        training_data_num = len(train_loader.dataset)
        end_steps = int(np.ceil(float(training_data_num) / float(train_loader.batch_size)))
        self.model.model.eval()
        self.model.heads.train()
        fuse_loader = iter(trn_loader)
        cyclic_fuse_loader = itertools.cycle(fuse_loader)
        for i, (images, target) in enumerate(train_loader):
            if i > end_steps:
                break
            try:
                images2, _ = next(cyclic_fuse_loader)
            except:
                cyclic_fuse_loader = itertools.cycle(fuse_loader)
                images2, _ = next(cyclic_fuse_loader)
            # H2T
            with torch.no_grad():
                _, feat1, feat1_ = self.model.model(images.to(self.device), return_feat=True)
                _, feat2, feat2_ = self.model.model(images2.to(self.device), return_feat=True)
                if feat1.shape[0] != feat2.shape[0]:
                    feat = feat1
                    feat_ = feat1_
                else:
                    feat = H2T(feat1, feat2)
                    feat_ = H2T(feat1_, feat2_)
            feat = self.layer4(feat)
            if (feat + feat_).shape[0] != 1:
                feat = self.iffm(feat, feat_)
            else:
                feat = feat + feat_
            feat = self.avgpool(feat)
            feat = torch.flatten(feat, 1)
            output = []
            for idx in range(len(self.model.heads)):
                output.append(self.model.heads[idx](feat.detach()))
                output[idx] = self.lws_models[idx](output[idx]['wsigma'])
            loss = self.criterion(t, output, target.to(self.device), stage2=True)
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()


    def adjust_learning_rate_stage_2(self, epoch):
        """Sets the learning rate"""
        lr_min = 0
        lr_max = 0.5
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(epoch / self.stage2epoch * 3.1415926535))
        print(' lr={:.1e}'.format(lr), end='')
        for idx, param_group in enumerate(self.optimizer.param_groups):
            if idx == 1:
                param_group['lr'] = (1 / self.lr_factor) * lr
            else:
                param_group['lr'] = 1.00 * lr

    def criterion(self, t, outputs, targets, ref_outputs=None, features=None, ref_features=None, stage2=False,
                  lwsoutputs=[]):
        """Returns the loss value"""
        if ref_outputs is None or ref_features is None or stage2:
            # print('stage2')
            if type(outputs[0]) == dict:
                outputs = torch.cat([o['wsigma'] for o in outputs], dim=1)
            else:
                outputs = torch.cat(outputs, dim=1)
            # Eq. 1: regular cross entropy
            loss = nn.CrossEntropyLoss(None)(outputs, targets)
        else:
            if self.less_forget:
                # Eq. 6: Less-Forgetting constraint
                loss_dist = nn.CosineEmbeddingLoss()(features, ref_features.detach(),
                                                     torch.ones(targets.shape[0]).to(self.device)) * self.lamda
            else:
                # Scores before scale, [-1, 1]
                ref_outputs = torch.cat([ro['wosigma'] for ro in ref_outputs], dim=1).detach()
                old_scores = torch.cat([o['wosigma'] for o in outputs[:-1]], dim=1)
                num_old_classes = ref_outputs.shape[1]

                # Eq. 5: Modified distillation loss for cosine normalization
                loss_dist = nn.MSELoss()(old_scores, ref_outputs) * self.lamda * num_old_classes

            loss_mr = torch.zeros(1).to(self.device)
            if self.margin_ranking:
                # Scores before scale, [-1, 1]
                outputs_wos = torch.cat([o['wosigma'] for o in outputs], dim=1)
                num_old_classes = outputs_wos.shape[1] - outputs[-1]['wosigma'].shape[1]

                # Sec 3.4: "We select those new classes that yield highest responses to x (...)"
                # The index of hard samples, i.e., samples from old classes
                hard_index = targets < num_old_classes
                hard_num = hard_index.sum()

                if hard_num > 0:
                    # Get "ground truth" scores
                    gt_scores = outputs_wos.gather(1, targets.unsqueeze(1))[hard_index]
                    gt_scores = gt_scores.repeat(1, self.K)

                    # Get top-K scores on novel classes
                    max_novel_scores = outputs_wos[hard_index, num_old_classes:].topk(self.K, dim=1)[0]

                    assert (gt_scores.size() == max_novel_scores.size())
                    assert (gt_scores.size(0) == hard_num)
                    # Eq. 8: margin ranking loss
                    loss_mr = nn.MarginRankingLoss(margin=self.dist)(gt_scores.view(-1, 1),
                                                                     max_novel_scores.view(-1, 1),
                                                                     torch.ones(hard_num * self.K).to(self.device))
                    loss_mr *= self.lamb_mr

            # Eq. 1: regular cross entropy
            loss_ce = nn.CrossEntropyLoss()(torch.cat([o['wsigma'] for o in outputs], dim=1), targets)
            # Eq. 9: integrated objective
            loss = loss_dist + loss_ce + loss_mr
        return loss

    @staticmethod
    def warmup_luci_loss(outputs, targets):
        if type(outputs) == dict:
            # needed during train
            return torch.nn.functional.cross_entropy(outputs['wosigma'], targets)
        else:
            # needed during eval()
            return torch.nn.functional.cross_entropy(outputs, targets)

    def get_cls_num_list(self, trn_loader):
        cls_num_dict = {}
        for images, targets in trn_loader:
            images, targets = images.to(self.device), targets.to(self.device)
            for target in targets:
                target = target.item()
                cls_num_dict[target] = cls_num_dict.get(target, 0) + 1
        return cls_num_dict

    def get_samples_weight(self, cls_weight, trn_loader):
        samples_weight = []
        for images, targets in trn_loader:
            images, targets = images.to(self.device), targets.to(self.device)
            for target in targets:
                target = target.item()
                samples_weight.append(cls_weight[target])
        return np.array(samples_weight)
    '''
    def get_new_indices(self, cls_num_dict, num_per_cls, samples_weight_loader):
        head_indices = []
        new_indices = []
        for batch_index, (inputs, targets) in enumerate(samples_weight_loader):
            targets = targets.to(self.device)
            batch_size = targets.size(0)
            for i in range(batch_size):
                target = targets[i].item()
                if cls_num_dict[target] > 100 or (cls_num_dict[target] == 20 and num_per_cls[target] > 111):
                    total_index = batch_index * batch_size + i
                    head_indices.append(total_index)
                if cls_num_dict[target] != 20:
                    total_index = batch_index * batch_size + i
                    new_indices.append(total_index)
        return head_indices, new_indices
    '''

# Sec 3.2: This class implements the cosine normalizing linear layer module using Eq. 4
class CosineLinear(Module):
    def __init__(self, in_features, out_features, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)  # for initializaiton of sigma

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out_s = self.sigma * out
        else:
            out_s = out
        if self.training:
            return {'wsigma': out_s, 'wosigma': out}
        else:
            return out_s


# This class implements a ResNet Basic Block without the final ReLu in the forward
class BasicBlockNoRelu(nn.Module):
    expansion = 1

    def __init__(self, conv1, bn1, relu, conv2, bn2, downsample):
        super(BasicBlockNoRelu, self).__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.relu = relu
        self.conv2 = conv2
        self.bn2 = bn2
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        # Removed final ReLU
        return out