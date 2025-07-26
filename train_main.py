import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from dataset_load import *
from model_zoo import model_create
from set_random_seed import set_seed
set_seed(66)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
from torchsummary import summary
import io
import sys
from test_zoo import *

criterion = nn.CrossEntropyLoss()
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
    def flush(self):
        for f in self.files:
            f.flush()

def compute_ppd(resolution, diagonal_size_inches, viewing_distance_meters):
    ar = resolution[0] / resolution[1]
    height_mm = math.sqrt((diagonal_size_inches * 25.4) ** 2 / (1 + ar ** 2))
    display_size_m = (ar * height_mm / 1000, height_mm / 1000)
    pix_deg = 2 * math.degrees(math.atan(0.5 * display_size_m[0] / resolution[0] / viewing_distance_meters))
    display_ppd = 1 / pix_deg
    return display_ppd

def train_one_epoch(model, trainloader, optimizer, criterion, device, epoch, test_class_list, resolution):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss_test_hvs_list = []
        for test_class in test_class_list:
            test_instance = test_class(sample_num=10)
            loss_test_hvs = test_instance.test_models(model=model, resolution=np.array(resolution)*2)
            loss_test_hvs_list.append(loss_test_hvs)
        loss += sum(loss_test_hvs_list)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()


    print(f"[Epoch {epoch}] Training Loss: {running_loss / len(trainloader):.3f}")

def test_one_epoch(model, testloader, device, epoch):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    print(f"[Epoch {epoch}] Test Accuracy: {acc:.2f}%")
    return acc

def train_model(model, trainloader, testloader, optimizer, scheduler, criterion, device, save_path, log_file_path, resolution, test_class_list, max_epochs=100):
    best_acc = 0.0
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, 'w') as log_file:
        buf = io.StringIO()
        tee = Tee(sys.stdout, buf)
        old_stdout = sys.stdout
        sys.stdout = tee
        try:
            summary(model, input_size=(3, resolution[0], resolution[1]))
        finally:
            sys.stdout = old_stdout
        log_file.write(buf.getvalue())
        log_file.write('\n')
        log_file.write(f"# Model: {model_name}, Dataset: {dataset_name}\n")
        for epoch in tqdm(range(1, max_epochs + 1)):
            train_one_epoch(model, trainloader, optimizer, criterion, device, epoch, test_class_list, resolution)
            acc = test_one_epoch(model, testloader, device, epoch)
            log_file.write(f"[Epoch {epoch}] Test Accuracy: {acc:.2f}%\n")
            log_file.flush()
            scheduler.step()
            # 保存最好的模型
            if acc > best_acc:
                best_acc = acc
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"✅ Saved best model with accuracy {best_acc:.2f}%")
                log_file.write(f"Saved best model with accuracy {best_acc:.2f}%\n")

if __name__ == '__main__':
    train_dataset_name_list = ['CIFAR-100']
    model_name_list = ['resnet18', 'resnet34']
    resolution = [32, 32] #64,64
    batch_size = 128 #* 8
    skip_trained_model = False
    test_classes = ["Contrast_Detection_Area", "Contrast_Detection_Luminance", "Contrast_Detection_SpF_Gabor_Ach", "Contrast_Masking_Phase_Coherent", "Contrast_Masking_Phase_Incoherent"]
    # test_classes = ["Contrast_Masking_Phase_Coherent"]
    test_class_list = [globals()[test_class] for test_class in test_classes]

    for dataset_name in train_dataset_name_list:
        for model_name in model_name_list:
            set_seed(66)
            print(f"Dataset: {dataset_name}, Model: {model_name}")
            trainloader = dataset_load(dataset_name=dataset_name, type='train', batch_size=batch_size)
            testloader = dataset_load(dataset_name=dataset_name, type='test', batch_size=batch_size)
            model = model_create(model_name=model_name, dataset_name=dataset_name, pretrained=True)
            model.to(device)
            optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

            save_path = (f'../HVS_VFM_loss_pth/best_{model_name}_{dataset_name}.pth')
            log_path = (f'../HVS_VFM_loss_logs/log_{model_name}_{dataset_name}.txt')
            # 如果日志文件存在，说明训练已经完成，跳过
            if os.path.exists(log_path) and skip_trained_model:
                print(f"🚫 Skipping: already trained — {log_path}")
                continue
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            os.makedirs(os.path.dirname(log_path), exist_ok=True)

            try:
                train_model(
                    model=model,
                    trainloader=trainloader,
                    testloader=testloader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion,
                    device=device,
                    save_path=save_path,
                    log_file_path=log_path,
                    resolution=resolution,
                    test_class_list=test_class_list,
                    max_epochs=20,
                )
            except Exception as e:
                print(f"Error occurred: {e}")





