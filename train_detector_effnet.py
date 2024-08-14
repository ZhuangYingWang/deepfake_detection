import copy
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'

device = torch.device("cuda:0")
log_file_tiemstamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime(time.time()))
logging.basicConfig(filename=f"logs/log_{log_file_tiemstamp}.txt", level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

net = EfficientNet.from_name("efficientnet-b7").to(device)
net._fc.out_features = 2
# net = models.res
net = net.to(device)

base_dir = "/home/dell/桌面/kaggle/deepfake_detection/data_phase1/phase1"

# data_preprocessing.main()

batch_size = 16
PRINT_STEP = 100

image_datasets = {
    "train": datasets.ImageFolder(os.path.join(base_dir, "trainset"), data_transform),
    "val": datasets.ImageFolder(os.path.join(base_dir, "valset"), data_transform),
}

dataloaders = {
    "train": torch.utils.data.DataLoader(
        image_datasets["train"], batch_size=batch_size, shuffle=True, num_workers=0  # 原来是8，改为0，减少并行加载数据时的内存使用
    ),
    "val": torch.utils.data.DataLoader(
        image_datasets["val"], batch_size=batch_size, shuffle=False, num_workers=0  #
    ),
}

dataset_sizes = {
    "train": len(image_datasets["train"]),
    "val": len(image_datasets["val"]),
}

class_names = image_datasets["train"].classes


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
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


# 在验证阶段添加计算AUC的代码
def calculate_auc(model, dataloader):
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


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        logger.info("Epoch {}/{}".format(epoch, num_epochs - 1))
        logger.info("-" * 10)

        # Training phase
        model.train()
        torch.cuda.empty_cache()  # 及时清理内存
        train_loss = 0.0
        train_corrects = 0.0
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
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data).item()
            loss_log.append(loss.item())

            step += 1
            if step % PRINT_STEP == 0 or step < 10:
                duration = time.time() - start_time
                over = time.strftime(
                    "%H:%M:%S",
                    time.localtime(time.time() + duration * (train_total_step - step)),
                )
                logger.info(
                    f"Epoch {epoch + 1} | Step {step:>{print_control}d}/{train_total_step} --- Use {duration:0.2f}s, over this epoch at {over} | loss = {loss.item():0.3e}")

        scheduler.step()
        train_epoch_loss = train_loss / dataset_sizes["train"]
        train_epoch_acc = train_corrects / dataset_sizes["train"]
        # np.save(f"logs/epoch_{epoch + 1}_loss.npy", np.array(loss_log))
        # plt.plot([i for i in range(len(loss_log))], loss_log)
        # plt.xlabel("Step")
        # plt.ylabel("loss")
        # plt.savefig(f"logs/epoch_{epoch + 1}_loss.jpg")
        # plt.cla()
        logger.info(
            "Train Loss: {:.4f} Acc: {:.4f}%".format(
                train_epoch_loss, train_epoch_acc * 100
            )
        )

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0.0

        logger.info("Test...")
        start_time = time.time()

        with torch.no_grad():
            for inputs, labels in dataloaders["val"]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()

        val_epoch_loss = val_loss / dataset_sizes["val"]
        val_epoch_acc = val_corrects / dataset_sizes["val"]
        val_auc = calculate_auc(model, dataloaders["val"])
        logger.info(
            "Val Loss: {:.4f} Acc: {:.4f} Auc: {:.4f}% | Use {:.2f}s".format(
                val_epoch_loss, val_epoch_acc * 100, val_auc, time.time() - start_time
            )
        )

        # Save the model if it has the best accuracy on validation set
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            weight_path = f"/home/dell/桌面/kaggle/deepfake_detection/weight/{best_acc:.4f}_epoch{epoch}.pt"
            logger.info(f"Save model {weight_path}")
            torch.save(model.state_dict(), weight_path)

    time_elapsed = time.time() - since
    logger.info(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    logger.info("Best val Acc: {:.4f}".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


criterion = nn.CrossEntropyLoss()
optimizer_ft = torch.optim.Adam(net.parameters(), lr=0.0005, betas=(0.9, 0.999))
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(net, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=20)
