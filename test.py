import os

import torch
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型
model = EfficientNet.from_name("efficientnet-b7").to(device)
model._fc.out_features = 2
model_state_dict = torch.load("/home/dell/桌面/kaggle/deepfake_detection/weight/0.8326_epoch10.pt", map_location=device,
                              weights_only=True)
model.load_state_dict(model_state_dict)
model.eval()  # 设置为评估模式

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

test_dir = "/home/dell/桌面/kaggle/deepfake_detection/phase2/testset1_seen"

predictions = []

for img_name in os.listdir(test_dir):
    if img_name.lower().endswith((".jpg", ".png")):
        img_path = os.path.join(test_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = data_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)  # 应用 softmax 函数ex
            _, preds = torch.max(probabilities, 1)
            score = probabilities[0][1]

        predictions.append(f"{img_name},{score.item()}")

with open("prediction.txt", "w") as f:
    for pred in predictions:
        f.write(pred + "\n")

print("Predictions have been written to prediction.txt")
