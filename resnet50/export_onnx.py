import torch
import torchvision
import torch.nn as nn

def export_resnet(model_name="resnet18", img_size=224, batch_size=1):
    """
    导出 torchvision 中的 ResNet 模型为 ONNX
    支持旧版 torchvision（无需 ResNet50_Weights）
    """
    # 1. 加载预训练模型（兼容旧版）
    if model_name == "resnet18":
        model = torchvision.models.resnet18(pretrained=True)
    elif model_name == "resnet50":
        model = torchvision.models.resnet50(pretrained=True)
    else:
        raise ValueError("仅支持 resnet18 或 resnet50")

    model.eval()

    # 2. 【关键】替换 AdaptiveAvgPool → AvgPool（DLA 不支持 adaptive）
    # 计算最后 feature map 尺寸: 对于 224x224 输入，ResNet 输出是 7x7
    # 所以用 kernel=7 的 AvgPool 等价于全局平均池化
    model.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)

    # 3. 构造 dummy input
    dummy_input = torch.randn(batch_size, 3, img_size, img_size)

    # 4. 导出 ONNX
    onnx_path = f"{model_name}.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,          # 兼容性最好（JetPack 5.0+ 支持）
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None          # DLA 要求静态 shape
    )
    print(f"✅ {model_name} 已导出为 {onnx_path}")

if __name__ == "__main__":
    export_resnet("resnet50", img_size=224, batch_size=1)