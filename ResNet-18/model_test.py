import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from model import ResNet18, Residual


def test_data_process():
    test_data = FashionMNIST(root='./data', train=False,
                              transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                              download=True)


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
    model = ResNet18(Residual)
    model.load_state_dict(torch.load("best_model.pth"))

    test_dataloader = test_data_process()
    test_model_process(model, test_dataloader)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = model.to(device)
    #
    # classes = FashionMNIST.classes
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