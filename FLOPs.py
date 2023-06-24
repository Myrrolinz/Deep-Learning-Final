from thop import profile
from thop import clever_format
from resnet import *
from van import *
from van_multibranch import *
from van_replk import *
from van_res2net import *
from replknet import *
from res2net import *
from LKACAT import *
from SE_LKACAT import *
from van_super import *

# 换自己的模型
model = van_b0()
input = torch.randn(1, 3, 224, 224)
macs, params = profile(model, inputs=(input, ))
macs, params = clever_format([macs, params], "%.3f")
print("macs: ", macs)
print("params: ", params)