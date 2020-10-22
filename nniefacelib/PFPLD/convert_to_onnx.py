"""
This code is used to convert the pytorch models into an onnx format models.
"""
import torch.onnx
from pfld.pfld import PFLDInference
import torch.nn as nn
input_img_size = 112  # define input size
from model import parsingNet
# from tensorboardX import SummaryWriter

#model_path = "models/pretrained/checkpoint_epoch_final.pth"
model_path = "models/pretrained/ep049_18_modify.pth"
checkpoint = torch.load(model_path)
#net = PFLDInference()
net = parsingNet(pretrained = False, backbone='18',cls_dim = (201,18, 4),use_aux=False).cuda() # we dont need auxiliary segmentation in testing

state_dict = torch.load(model_path, map_location = 'cpu')['model']
compatible_state_dict = {}
for k, v in state_dict.items():
    if 'module.' in k:
        compatible_state_dict[k[7:]] = v
    else:
        compatible_state_dict[k] = v

net.load_state_dict(compatible_state_dict, strict = False)

# net.load_state_dict(checkpoint,False)
#net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(model_path)['model_path'].items()})
#net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(model_path).items()})
net.eval()
net.to("cuda")

model_name = model_path.split("/")[-1].split(".")[0]
model_path = f"models/onnx/{model_name}.onnx"

dummy_input = torch.randn(1, 3, 288, 800).to("cuda")
#dummy_input = torch.randn(1, 3, 112, 112).to("cuda")

# with SummaryWriter(comment='parsingNet') as w:
#     w.add_graph(net, (dummy_input, ))

torch.onnx.export(net, dummy_input, model_path, export_params=True, verbose=False)#, input_names=['input'], output_names=['pose', 'landms']
# torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
#                                input_names=input_names, output_names=output_names)
