import torch


def convert_model(para_model,single_model):
    checkpoint = torch.load(para_model ,map_location=torch.device("cpu"))

    output={}
    for key,value in checkpoint['model'].items():
        output[key.lstrip("module.")]=value

    torch.save({'model':output},single_model)

if __name__=="__main__":
    convert_model(para_model="model_2.pth", single_model="model.pth")




