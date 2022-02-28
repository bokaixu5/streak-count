# 如果需要遍历多张图片，可以在上面加一个for循环
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

i=1
list=[]
pic_path = "./JPEGImages/"   # 图片路径
for file in os.listdir(pic_path):
	file_name = pic_path + str(i)+".jpg"
	#print(file_name)
	i=i+1
	img = cv2.imread(file_name)
	img1=img[555, 715]
	list.append(img1[0])

X=np.linspace(1, 129, num=129) # X轴坐标数据



#plt.plot(X,list,lable="$sin(X)$",color="red",linewidth=2)

plt.figure(figsize=(8,6))  # 定义图的大小
plt.xlabel("time(s)")     # X轴标签
plt.ylabel("Volt")        # Y轴坐标标签
plt.title("Example")      #  曲线图的标题

plt.plot(X,list)            # 绘制曲线图

plt.show()

