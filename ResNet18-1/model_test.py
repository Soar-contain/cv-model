import os

import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import Residual, ResNet18
from tqdm import tqdm
from PIL import Image
import pandas as pd


def test_data_process():
    # 定义数据集的路径
    ROOT_TRAIN = 'gdgoc-telkom-university-2026-mask-classification/Dataset/test'
    normalize = transforms.Normalize([0.198, 0.181, 0.175], [0.085, 0.077, 0.075])
    # 定义数据集处理方法变量
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    # 加载数据集
    test_data = ImageFolder(ROOT_TRAIN, transform=test_transform)


    test_dataloader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=True, num_workers=0)

    return test_dataloader


def test_model_process(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    # 只进行前向传播计算，不计算梯度
    with torch.no_grad():
        for test_data_x, test_data_y in tqdm(test_dataloader):
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            model.eval()

            output = model(test_data_x)

            prelab = torch.argmax(output, dim=1)

            test_corrects += torch.sum(prelab == test_data_y.data)

            test_num += test_data_x.size(0)
            print(prelab)
            break

        test_acc = test_corrects.double().item() / test_num
        print("测试的准确率为：", test_acc)


if __name__ == '__main__':
    # 加载模型
    model = ResNet18(Residual)
    model.load_state_dict(torch.load("best_model.pth"))

    # test_dataloader = test_data_process()
    # test_model_process(model, test_dataloader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    result = []
    name = []
    # #
    # classes = ["猫", "狗"]
    # with torch.no_grad():
    #     for b_x, b_y in test_dataloader:
    #         b_x = b_x.to(device)
    #         b_y = b_y.to(device)
    #
    #         model.eval()
    #         output = model(b_x)
    #
    #         prelab = torch.argmax(output, dim=1)
    #         result = prelab.item()
    #         label = b_y.item()
    #
    #         print("预测值：", classes[result], "----------", "真实值：", classes[label])

    for image_name in tqdm(os.listdir("gdgoc-telkom-university-2026-mask-classification/Dataset/test")):
        image_path = os.path.join("gdgoc-telkom-university-2026-mask-classification/Dataset/test", image_name)
        image = Image.open(image_path)
        normalize = transforms.Normalize([0.198, 0.181, 0.175], [0.085, 0.077, 0.075])
        test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
        image = test_transform(image)

        image = image.unsqueeze(0)


        with torch.no_grad():
            image = image.to(device)
            model.eval()
            output = model(image)
            pre_lab = torch.argmax(output, dim=1)
            result.append(pre_lab.item())
            name.append(image_name)

    df = pd.DataFrame({
        "filename": name,
        'label': result
    })

    # 2. 写入CSV（index=False 表示不写入Pandas自带的行号）
    df.to_csv('submission.csv', index=False, encoding='utf-8')
    print("预测结果已保存！")


