from agent2 import *
from envi import *

a = Environment()
# a.reset()
fig = plt.figure(figsize = (12, 6))
plt.plot(a.data, color = 'red')
x, y, rate = a.find_peak()
plt.scatter(x, y, color = 'blue')
plt.xlabel('Số lượng mẫu')
plt.ylabel('Giá trị trung bình bình phương RMS');
print(a.rpm)
print(rate)
plt.show()