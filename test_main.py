import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import math
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from dataset_load import *
from model_zoo import model_create
from set_random_seed import set_seed
from torchsummary import summary
import io
import sys
from test_zoo import *

set_seed(66)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
criterion = nn.CrossEntropyLoss()

def compute_ppd(resolution, diagonal_size_inches, viewing_distance_meters):
    ar = resolution[0] / resolution[1]
    height_mm = math.sqrt((diagonal_size_inches * 25.4) ** 2 / (1 + ar ** 2))
    display_size_m = (ar * height_mm / 1000, height_mm / 1000)
    pix_deg = 2 * math.degrees(math.atan(0.5 * display_size_m[0] / resolution[0] / viewing_distance_meters))
    display_ppd = 1 / pix_deg
    return display_ppd

def test_model(model, testloader, device, test_classes, test_class_list, resolution):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    print(f"✅ Accuracy: {acc:.2f}%")

    for test_name, test_class in enumerate(test_classes, test_class_list):
        test_instance = test_class(sample_num=10)
        loss_test_hvs = test_instance.test_models(model=model, resolution=np.array(resolution) * 2)
        print(f"Testing {test_name} Loss: {loss_test_hvs:.2f}")

    return acc

# ✅ 简写字典
test_special_name_dict = {
    "Contrast_Detection_Area": "CDA",
    "Contrast_Detection_Luminance": "CDL",
    "Contrast_Detection_SpF_Gabor_Ach": "CDSGA",
    "Contrast_Masking_Phase_Coherent": "CMPC",
    "Contrast_Masking_Phase_Incoherent": "CMPI"
}

if __name__ == '__main__':
    # ✅ 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_classes', nargs='+', default=[], help='List of test class names to use')
    args = parser.parse_args()

    test_classes = args.test_classes
    test_class_list = [globals()[name] for name in test_classes]
    suffix = "_".join([test_special_name_dict.get(name, name) for name in test_classes])

    train_dataset_name_list = ['CIFAR-100']
    model_name_list = ['resnet18', 'resnet34']
    resolution = [32, 32]
    batch_size = 128
    skip_trained_model = False

    for dataset_name in train_dataset_name_list:
        for model_name in model_name_list:
            set_seed(66)
            print(f"Dataset: {dataset_name}, Model: {model_name}, Tests: {test_classes}")
            testloader = dataset_load(dataset_name=dataset_name, type='test', batch_size=batch_size)
            model = model_create(model_name=model_name, dataset_name=dataset_name, pretrained=True)
            model.to(device)
            model.eval()
            model_path = f'../HVS_VFM_loss_pth/best_{model_name}_{dataset_name}_{suffix}.pth'
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
                test_model(model, testloader, device, test_classes, test_class_list, resolution)
            else:
                print(f"❌ Model weights not found at {model_path}")

