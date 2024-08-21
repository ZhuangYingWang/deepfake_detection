import argparse
import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from config import get_config
from dataset import get_augmentation_train_data
from model import get_model
from utils.metric_util import calculate_auc

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'


#######
def plot_curves(train_accs, val_accs, train_losses, val_losses):
    epochs = range(1, len(train_accs) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accs, 'bo-', label='Training Acc')
    plt.plot(epochs, val_accs, 'ro-', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


# 在验证阶段添加计算AUC的代码
def calculate_auc_val(args: argparse.Namespace, model, dataloader):
    device = args.device
    model.eval()
    y_true = []
    y_scores = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)  # 获取概率
            y_scores.append(probs.cpu().numpy())
            y_true.append(labels.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_scores = np.concatenate(y_scores)
    auc_score = roc_auc_score(y_true, y_scores[:, 1])  # 选择正类的概率
    return auc_score


def train_model(args: argparse.Namespace, dataloaders, dataset_sizes, class_to_idx, model, criterion, optimizer,
                scheduler,
                num_epochs):
    device = args.device
    logger = args.logger

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    train_fake_id = class_to_idx["train"]["fake"]
    train_real_id = class_to_idx["train"]["real"]

    for epoch in range(num_epochs):
        logger.info("Epoch {}/{}".format(epoch + 1, num_epochs))
        logger.info("-" * 10)

        # Training phase
        model.train()
        torch.cuda.empty_cache()  # 及时清理内存
        train_loss = 0.0
        train_corrects = 0.0
        train_predict_fake_count = 0
        train_predict_real_count = 0
        train_label_fake_count = 0
        train_label_real_count = 0
        train_predict_prob = []
        train_label = []

        # log
        train_total_step = len(dataloaders["train"])
        print_control = len(str(train_total_step))
        loss_log = []
        start_time = time.time()
        step = 0

        for inputs, labels in dataloaders["train"]:
            start_time = time.time()
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            softmax_output = torch.softmax(outputs, dim=1)
            predict_labels = torch.argmax(softmax_output, dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # for auc
            train_label.append(labels.cpu().detach().numpy())
            train_predict_prob.append(softmax_output.cpu().detach().numpy()[:, train_fake_id])

            train_loss += loss.item() * inputs.size(0)

            train_corrects += torch.sum(predict_labels == labels.data).item()
            train_label_fake_count += torch.sum(labels.data == train_fake_id).item()
            train_predict_fake_count += torch.sum(
                (predict_labels == train_fake_id) & (labels.data == train_fake_id)).item()
            train_label_real_count += torch.sum(labels.data == train_real_id).item()
            train_predict_real_count += torch.sum(
                (predict_labels == train_real_id) & (labels.data == train_real_id)).item()

            loss_log.append(loss.item())

            step += 1
            if step % args.print_step == 0 or step < 10:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Current learning rate: {current_lr:.6f}")
                duration = time.time() - start_time
                over = time.strftime(
                    "%H:%M:%S",
                    time.localtime(time.time() + duration * (train_total_step - step)),
                )
                logger.info(
                    f"Epoch {epoch + 1} | Step {step:>{print_control}d}/{train_total_step} --- Use {duration:0.2f}s, over this epoch at {over} | loss = {loss.item():0.3e}")
            scheduler.step()

        # summary epoch
        train_epoch_loss = train_loss / dataset_sizes["train"]
        train_epoch_acc = train_corrects / dataset_sizes["train"]

        train_label = np.concatenate(train_label)
        train_predict_prob = np.concatenate(train_predict_prob)
        train_auc_score = calculate_auc(train_label, train_predict_prob, train_fake_id)

        # np.save(f"logs/epoch_{epoch + 1}_loss.npy", np.array(loss_log))
        # plt.plot([i for i in range(len(loss_log))], loss_log)
        # plt.xlabel("Step")
        # plt.ylabel("loss")
        # plt.savefig(f"logs/epoch_{epoch + 1}_loss.jpg")
        # plt.cla()
        logger.info(
            "Train Loss: {:.4f} Auc: {:.4f} Acc: {:.4f}% | Fake Acc: {:4f}% | Real Acc: {:.4f}%".format(
                train_epoch_loss, train_auc_score, train_epoch_acc * 100,
                                                   train_predict_fake_count / train_label_fake_count * 100,
                                                   train_predict_real_count / train_label_real_count * 100
            )
        )

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0.0
        val_predict_fake_count = 0
        val_predict_real_count = 0
        val_label_fake_count = 0
        val_label_real_count = 0
        val_fake_id = class_to_idx["val"]["fake"]
        val_real_id = class_to_idx["val"]["real"]
        val_label = []
        val_predict_prob = []

        logger.info("Test...")
        start_time = time.time()

        with torch.no_grad():
            for inputs, labels in dataloaders["val"]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                softmax_output = torch.softmax(outputs, dim=1)
                predict_labels = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                loss = criterion(outputs, labels)

                # for auc
                val_label.append(labels.cpu().detach().numpy())
                val_predict_prob.append(softmax_output.cpu().detach().numpy()[:, train_fake_id])

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(predict_labels == labels.data).item()
                val_label_fake_count += torch.sum(labels.data == val_fake_id).item()
                val_predict_fake_count += torch.sum(
                    (predict_labels == val_fake_id) & (labels.data == val_fake_id)).item()
                val_label_real_count += torch.sum(labels.data == val_real_id).item()
                val_predict_real_count += torch.sum(
                    (predict_labels == val_real_id) & (labels.data == val_real_id)).item()

        val_epoch_loss = val_loss / dataset_sizes["val"]
        val_epoch_acc = val_corrects / dataset_sizes["val"]
        # val_auc = calculate_auc_val(args, model, dataloaders["val"])
        val_label = np.concatenate(val_label)
        val_predict_prob = np.concatenate(val_predict_prob)
        val_auc_score = calculate_auc(val_label, val_predict_prob, val_fake_id)
        logger.info(
            "Val Loss: {:.4f} Acc: {:.4f}% Auc: {:.4f} | Fake Acc: {:4f}% | Real Acc: {:.4f}% | Use {:.2f}s".format(
                val_epoch_loss, val_epoch_acc * 100, val_auc_score, val_predict_fake_count / val_label_fake_count * 100,
                                val_predict_real_count / val_label_real_count * 100, time.time() - start_time
            )
        )

        # Save the model if it has the best accuracy on validation set
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            weight_path = os.path.join(args.weight_dir, f"{best_acc:.4f}_epoch{epoch}.pt")
            logger.info(f"Save model {weight_path}")
            torch.save(model.state_dict(), weight_path)
        else:
            weight_path = os.path.join(args.weight_dir, f"{best_acc:.4f}_epoch{epoch}.pt")
            logger.info(f"Save model {weight_path}")
            torch.save(model.state_dict(), weight_path)

    time_elapsed = time.time() - since
    logger.info(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    logger.info("Best val Acc: {:.4f}".format(best_acc))

    return model


def main(args: argparse.Namespace):
    # get train data
    dataloaders, dataset_sizes, class_to_idx = get_augmentation_train_data(args)
    # get model
    model_dict = get_model(args)
    # train model
    model_ft = train_model(args, dataloaders, dataset_sizes, class_to_idx, **model_dict)


if __name__ == '__main__':
    config = get_config()
    main(config)
