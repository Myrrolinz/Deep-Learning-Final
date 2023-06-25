from thop import profile
from thop import clever_format
from res2net import *


# 换自己的模型
model = res2net50()
input = torch.randn(1, 3, 224, 224)
macs, params = profile(model, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")
print("macs: ", macs)
print("params: ", params)