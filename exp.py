from cmath import pi
import numpy as np
from activations import *




# data=np.array([[1,2],
#                [13,2],
#                [1,23],
#                [2,1],
# ])
# # data=np.array([[0.25], [-1], [2.3], [-0.2], [1]]).T
# print(data.shape)
# # data=np.random.rand(1,4)*4

# print("input data:",data)
# # act=relu()
# act=softmax()

# print("sum of output:",np.sum(act.forward(data)))
# print("output:",np.around(act.forward(data),4))
# print("derivatives:",act.backward(data))


color=np.array([[1,2,3],
                [3,2,1]
])
print(color.shape,color)
pixels=2
image=np.tile(color,pixels**2).reshape(2,pixels,pixels,3)
print(image.shape)
print(image)