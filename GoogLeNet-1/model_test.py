import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import GoogLeNet, Inception
from PIL import Image


def test_data_process():
    # 定义数据集的路径
    ROOT_TRAIN = './data/test'
    normalize = transforms.Normalize([0.162, 0.151, 0.138], [0.058, 0.052, 0.048])
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
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            model.eval()

            output = model(test_data_x)

            prelab = torch.argmax(output, dim=1)

            test_corrects += torch.sum(prelab == test_data_y.data)

            test_num += test_data_x.size(0)

        test_acc = test_corrects.double().item() / test_num
        print("测试的准确率为：", test_acc)


if __name__ == '__main__':
    # 加载模型
    model = GoogLeNet(Inception)
    model.load_state_dict(torch.load("best_model.pth"))

    test_dataloader = test_data_process()
    test_model_process(model, test_dataloader)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
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

    # image = Image.open("dogs-vs-cats-imint-2020/dogs-vs-cats-imint-2020/test/1.jpg")
    # normalize = transforms.Normalize([0.162, 0.151, 0.138], [0.058, 0.052, 0.048])
    # test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    # image = test_transform(image)
    #
    # image = image.unsqueeze(0)
    #
    #
    # with torch.no_grad():
    #     image = image.to(device)
    #     model.eval()
    #     output = model(image)
    #     pre_lab = torch.argmax(output, dim=1)
    #     result = pre_lab.item()
    # print("预测值：", classes[result])

