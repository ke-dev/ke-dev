import numpy as np
 
np.set_printoptions(threshold=np.inf)

from inspect import isgenerator

from torch import is_grad_enabled
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
 
img=Image.open(np.str('./pic_dir/ADE_val_00000025.png'))
img=np.array(img)
print(img)