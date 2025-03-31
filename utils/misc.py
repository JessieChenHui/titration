#!/usr/bin/python
# -*- coding: UTF-8 -*-
# create date: 2024/7/25
# __author__: 'Alex Lu'
import copy
import os
import torch
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from datetime import datetime
from timm.utils import accuracy, AverageMeter
import logging
import torch.nn.functional as F
from torch import nn


def train_predict(model, device, train_loader, valid_loader, test_loader, **kwargs):
    best_model, train_results, valid_results = train(model, device, train_loader, valid_loader, **kwargs)
    model = best_model
    test_result, true_list, pred_list = predict(model, device, test_loader, **kwargs)

    results = {}
    results['train'] = train_results, valid_results
    results['test'] = test_result, true_list, pred_list
    model_save_path = kwargs.get("model_save_path", "./outputs/checked")
    if model_save_path:
        os.makedirs(model_save_path, exist_ok=True)
        model_name_prefix = kwargs.get("model_name_prefix", "SXT")
        valic_acc = max([valid_result[0] for valid_result in valid_results], default=0)
        test_acc = test_result[0]

        model_file_name = f'{model_name_prefix}_{datetime.now().strftime("%Y%m%d%H%M%S")}_' \
                          f'{int(valic_acc * 1000)}_{int(test_acc * 1000)}.pth'
        model_file = os.path.join(model_save_path, model_file_name)
        logging.info(f"model_file:{model_file}")
        torch.save(model.state_dict(), model_file)
    return best_model, results


def __run_epoch(model, dataloader, device, epoch, criterion=None, optimizer=None, training=False, **kwargs):
    total_size = 0
    total_loss = 0.0
    total_correct = 0.0
    pred_list = []
    true_list = []
    # grouped = kwargs.get('grouped', False)

    extra_stats = kwargs.get('extra_stats', False)
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    for images, labels in tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch"):
        images, labels = images.to(device), labels.to(device)
        batch_size = labels.size(0)

        # ----S----
        # Alex: normal sequence: zero_grad(), models(), criterion(), backward(), step()
        # Alex: Following code results are correct.
        outputs = model(images)
        loss = criterion(outputs, labels)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ----E----

        total_loss += loss.item() * batch_size
        total_size += batch_size
        if len(outputs.shape) > 2:
            outputs = F.log_softmax(outputs, dim=-1).sum(1)

        _, predicted = torch.max(outputs, 1)   # it is not accurate in train mode for grouped
        total_correct += (predicted == labels).sum().item()

        if extra_stats:
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            acc1_meter.update(acc1.item(), n=batch_size)
            acc5_meter.update(acc5.item(), n=batch_size)
            pred_list += predicted.tolist()
            true_list += labels.tolist()

    if extra_stats:
        logging.info(f'acc1:{acc1_meter.avg / 100:.3f}, acc5:{acc5_meter.avg / 100:.3f}.')

    return total_correct, total_loss, total_size, true_list, pred_list


def run_epoch(model, dataloader, device, epoch=0, criterion=torch.nn.CrossEntropyLoss(), optimizer=None,
              training=False, **kwargs):
    if training:
        model.train()
        total_correct, total_loss, total_size, true_list, pred_list = \
            __run_epoch(model, dataloader, device, epoch, criterion=criterion, optimizer=optimizer,
                        training=training, **kwargs)
    else:
        model.eval()
        with torch.no_grad():
            total_correct, total_loss, total_size, true_list, pred_list = \
                __run_epoch(model, dataloader, device, epoch, criterion=criterion, optimizer=optimizer,
                            training=training, **kwargs)
    return total_correct, total_loss, total_size, true_list, pred_list


def __log_epoch(epoch, train_acc, train_loss, valid_acc, valid_loss, patience_count):
    logging.info(f"epoch:{epoch}. Train Accuracy: {train_acc:.4f} Train Loss: {train_loss:.6f}.")
    logging.info(f"epoch:{epoch}. Valid Accuracy: {valid_acc:.4f} Valid Loss: {valid_loss:.6f}. "
                 f"patience count:{patience_count:2d}")


def train(model, device, train_loader, valid_loader, **kwargs):
    # 定义损失函数和优化器
    label_smoothing = kwargs.get('label_smoothing', 0.1)
    criterion = kwargs.pop('criterion', torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing))
    # optimizer = kwargs.pop('optimizer', torch.optim.AdamW(models.parameters(), lr=0.001, weight_decay=1e-4))
    optimizer = kwargs.pop('optimizer', torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4))
    # optimizer = torch.optim.Adam(models.parameters(), lr=0.001)

    # 训练模型
    min_epochs = kwargs.get("min_epochs", 20)
    total_epochs = kwargs.get("total_epochs", 200)  # 定义一个足够大的训练轮次
    patience = kwargs.get("patience", 20)  # 定义耐心值，即验证集准确率不再提高的轮次

    patience_count = 0  # 记录验证集准确率不再提高的轮次
    best_accuracy = 0.0
    best_loss = 0.0
    best_epoch = 0
    best_model = None

    train_results = []
    valid_results = []

    time0 = datetime.now()
    model.to(device)
    for epoch in range(total_epochs):
        total_correct, total_loss, train_size, _, _ = run_epoch(model, train_loader, device, epoch, criterion=criterion,
                                                                optimizer=optimizer, training=True, **kwargs)
        train_acc, train_loss = round(total_correct / train_size, 4), round(total_loss / train_size, 6)
        total_correct, total_loss, valid_size, _, _ = run_epoch(model, valid_loader, device, epoch, criterion=criterion,
                                                                optimizer=optimizer, training=False, **kwargs)
        valid_acc, valid_loss = round(total_correct / valid_size, 4), round(total_loss / valid_size, 6)

        train_results.append((train_acc, train_loss, train_size))
        valid_results.append((valid_acc, valid_loss, valid_size))

        # 如果当前模型性能更好，则保存模型
        if (valid_acc > best_accuracy) or (valid_acc == best_accuracy and valid_loss < best_loss - 0.001):
            best_accuracy = valid_acc
            best_loss = valid_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            if epoch % 10 != 0:
                __log_epoch(epoch, train_acc, train_loss, valid_acc, valid_loss, patience_count)
            patience_count = 0  # 重置耐心计数
        else:
            patience_count += 1

        if epoch % 10 == 0:
            __log_epoch(epoch, train_acc, train_loss, valid_acc, valid_loss, patience_count)

        # 如果耐心计数达到耐心值，则跳出训练
        if epoch >= min(min_epochs, 50, total_epochs // 2):
            if (patience_count >= patience):
                if epoch % 10 != 0:  # epoch % 10 == 0  => already log in previous
                    __log_epoch(epoch, train_acc, train_loss, valid_acc, valid_loss, patience_count)
                logging.info(f"epoch:{epoch}. Early stopping!")
                break

    time1 = datetime.now()
    logging.info(f'In train, best_accuracy is {best_accuracy}, in epoch {best_epoch}, avg loss is {best_loss:.6f}. '
                 f'time cost is {(time1 - time0).seconds} seconds.')

    return best_model, train_results, valid_results


def predict(model, device, test_loader, **kwargs):
    model.to(device)
    criterion = kwargs.pop('criterion', torch.nn.CrossEntropyLoss(label_smoothing=0.1))
    if kwargs.get('extra_stats') is None:
        kwargs['extra_stats'] = True
    time1 = datetime.now()
    total_correct, total_loss, total_size, true_list, pred_list = \
        run_epoch(model, test_loader, device, criterion=criterion, **kwargs)
    time2 = datetime.now()

    test_acc, test_loss = round(total_correct / total_size, 4), round(total_loss / total_size, 6)

    logging.info(f'In predict, accuracy is : {test_acc:.4f}. ={total_correct}/{total_size}. '
                 f'loss is {test_loss:.6f}.  time cost is {(time2 - time1).microseconds} microseconds.')

    precision, recall, f1_score, _ = precision_recall_fscore_support(true_list, pred_list, average='weighted',
                                                                     zero_division=1)
    logging.info(f'precision: {precision:.3f}. recall:{recall:.3f}. F1_score:{f1_score:.3f}. ')

    cm = confusion_matrix(true_list, pred_list)
    logging.info('confusion_matrix:\n')
    logging.info(cm)

    # cr = classification_report(true_list, pred_list, digits=3, zero_division=0)
    test_result = (test_acc, test_loss, total_size)
    return test_result, true_list, pred_list


def init_logger(log_file):
    log_file = log_file if log_file is not None else f"{datetime.now().strftime('%Y%m%d')}.log"
    lgr = logging.getLogger()

    different_log_file = False
    for handler in lgr.handlers:
        if isinstance(handler, logging.FileHandler):
            if not os.path.exists(log_file):
                with open(log_file, 'w') as f:
                    f.write('')
            if not os.path.samefile(handler.baseFilename, log_file):
                different_log_file = True
    if different_log_file:
        lgr.handlers.clear()

    if not lgr.hasHandlers():
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s', datefmt='%H:%M:%S'))
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter('%(message)s'))

        lgr.addHandler(file_handler)
        lgr.addHandler(console_handler)
        lgr.setLevel(logging.INFO)

    return lgr
