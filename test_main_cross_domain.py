import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset_load import *
from model_zoo import model_create
from set_random_seed import set_seed
from test_zoo import *
import itertools

set_seed(66)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
criterion = nn.CrossEntropyLoss()

# ✅ 简写字典
test_special_name_dict = {
    "Contrast_Detection_Area": "CDA",
    "Contrast_Detection_Luminance": "CDL",
    "Contrast_Detection_SpF_Gabor_Ach": "CDSGA",
    "Contrast_Masking_Phase_Coherent": "CMPC",
    "Contrast_Masking_Phase_Incoherent": "CMPI"
}

def test_model(model_name, model, dataset_name, testloader, test_classes, test_class_list, suffix, resolution):
    # 分类准确率测试
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(testloader, desc=f"Testing {model_name} on {dataset_name}"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    print(f"🎯 Test Accuracy: {acc:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_classes', nargs='+', default=[], help='List of test class names to use')
    args = parser.parse_args()

    test_classes = args.test_classes
    test_class_list = [globals()[name] for name in test_classes]
    train_skip_iter = 50
    start_epoch = 12
    suffix = f"{train_skip_iter}_{start_epoch}" + "_".join([test_special_name_dict.get(name, name) for name in test_classes])

    base_model_dataset = 'CIFAR-100'
    test_dataset_name_list = ['CIFAR-100-C']
    model_name_list = ['resnet18', 'resnet34']
    resolution = [32, 32]
    batch_size = 128

    corruption_type_list = ['gaussian_noise', 'fog', 'jpeg_compression']
    severity_list = [2]

    for dataset_name in test_dataset_name_list:
        for model_name in model_name_list:
            for corruption_type, severity in itertools.product(corruption_type_list, severity_list):
                set_seed(66)
                print(f"🧪 Testing: Dataset = {dataset_name}, Corruption_type = {corruption_type},"
                      f" Severity = {severity}, Model = {model_name}, Tests = {test_classes}")

                # testloader = dataset_load(dataset_name=dataset_name, type='test', batch_size=batch_size)
                testloader = dataset_load(dataset_name=dataset_name, type='test',
                                          corruption_type=corruption_type, severity=severity)
                model = model_create(model_name=model_name, dataset_name=base_model_dataset, pretrained=False)
                model.to(device)

                # 加载模型权重
                weight_path = f'../HVS_VFM_loss_pth/best_{model_name}_{base_model_dataset}_{suffix}'
                weight_path += '_hvs.pth' if len(test_classes) else '.pth'

                if not os.path.exists(weight_path):
                    print(f"🚫 Model weights not found: {weight_path}")
                    continue

                model.load_state_dict(torch.load(weight_path, map_location=device))
                print(f"✅ Loaded model from {weight_path}")

                # 运行测试
                test_model(model_name, model, dataset_name, testloader, test_classes, test_class_list, suffix,
                           resolution)
