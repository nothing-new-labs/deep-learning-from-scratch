import numpy as np
import matplotlib.pyplot as plt

# sudo apt install python3-tk

#import matplotlib
#print(matplotlib.rcsetup.interactive_bk)  # 查看支持的交互式后端
#print(matplotlib.get_backend())

# 生成数据
x = np.arange(0, 6, 0.1)
y = np.sin(x)

# 绘制图形
plt.plot(x, y)
plt.show()
