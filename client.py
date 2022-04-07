import time
import torch
from utils import get_optimizer, get_model
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torch.autograd import Variable
import copy
from optimization import Optimization
from torch.utils.tensorboard import SummaryWriter


class Client:
    def __init__(self, cid, data, device, project_dir, model_name, local_epoch, lr, batch_size, drop_rate, stride):
        self.cid = cid
        self.project_dir = project_dir
        self.model_name = model_name
        self.data = data
        self.device = device
        self.local_epoch = local_epoch
        self.lr = lr
        self.batch_size = batch_size

        self.dataset_sizes = self.data.train_dataset_sizes[cid]
        self.train_loader = self.data.train_loaders[cid]

        self.full_model = get_model(self.data.train_class_sizes[cid], drop_rate, stride)
        # full model contains add_block & classifier
        # backup origin classifier
        self.classifier = self.full_model.classifier.classifier  # full connect layer
        # replace the full connection layer of the model with null
        self.full_model.classifier.classifier = nn.Sequential()
        # model without full connection layer
        self.model = self.full_model
        # print(self.model)
        self.distance = 0
        self.optimization = Optimization(self.train_loader, self.device)
        # print("class name size",class_names_size[cid])

    # training based on federated_model's params
    def train(self, federated_model, use_cuda):
        self.y_err = []
        self.y_loss = []

        # federated_model.state_dict() contains model params
        self.model.load_state_dict(federated_model.state_dict())
        # return the classifier to the self.model.classifier.classifier
        # ensure the correct load of params
        self.model.classifier.classifier = self.classifier
        self.old_classifier = copy.deepcopy(self.classifier)
        self.model = self.model.to(self.device)

        optimizer = get_optimizer(self.model, self.lr)
        # drop the learning rate(lr * gamma) every 40 epoch
        scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

        # 分类问题使用交叉熵损失函数
        # torch.nn.TripletMarginLoss
        criterion = nn.CrossEntropyLoss()

        since = time.time()

        print('Client', self.cid, 'start training')
        for epoch in range(self.local_epoch):
            print('Epoch {}/{}'.format(epoch, self.local_epoch - 1))
            print('-' * 10)
            self.model.train(True)
            running_loss = 0.0
            running_corrects = 0.0

            for data in self.train_loader:
                inputs, labels = data
                b, c, h, w = inputs.shape
                if b < self.batch_size:
                    # skipping last data batch
                    continue
                if use_cuda:
                    # .detach() method remove grad from tensor
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                # writer = SummaryWriter('logs')
                # writer.add_graph(self.model, inputs)
                # print(inputs.shape)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                # calc grad
                loss.backward()
                # back propagation
                optimizer.step()

                running_loss += loss.item() * b
                running_corrects += float(torch.sum(preds == labels.data))

            scheduler.step()
            # calc data_size after drop_last, although drop_last is False, last batch is discarded
            used_data_sizes = (self.dataset_sizes - self.dataset_sizes % self.batch_size)
            epoch_loss = running_loss / used_data_sizes
            epoch_acc = running_corrects / used_data_sizes

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                'train', epoch_loss, epoch_acc))

            self.y_loss.append(epoch_loss)
            self.y_err.append(1.0 - epoch_acc)

            time_elapsed = time.time() - since
            print('Client', self.cid, ' Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

        time_elapsed = time.time() - since
        print('Client', self.cid, 'Epoch {}/{}'.format(epoch, self.local_epoch - 1),
              ' Training complete in {:.0f}m {:.0f}s'.format(
                  time_elapsed // 60, time_elapsed % 60))

        # save_network(self.model, self.cid, 'last', self.project_dir, self.model_name, gpu_ids)

        # store the trained classifier for next training
        self.classifier = self.model.classifier.classifier
        self.distance = self.optimization.cdw_feature_distance(federated_model, self.old_classifier, self.model)
        self.model.classifier.classifier = nn.Sequential()

    def generate_soft_label(self, x, regularization):
        return self.optimization.kd_generate_soft_label(self.model, x, regularization)

    def get_model(self):
        return self.model

    def get_data_sizes(self):
        return self.dataset_sizes

    # get last training loss
    def get_train_loss(self):
        return self.y_loss[-1]

    def get_cos_distance_weight(self):
        return self.distance
