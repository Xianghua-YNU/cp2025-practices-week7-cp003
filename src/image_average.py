# 导入必要的库
# Import necessary libraries
import numpy as np
import scipy.ndimage as sim
import matplotlib.pyplot as plt

def create_small_filter():
    """创建3×3平均滤波器
    Create a 3×3 averaging filter"""
    # 创建一个3×3的矩阵，所有元素值为1/9
    # Create a 3×3 matrix with all elements set to 1/9
    return np.ones((3, 3)) / 9

def create_large_filter():
    """创建15×15平均滤波器
    Create a 15×15 averaging filter"""
    # 创建一个15×15的矩阵，所有元素值为1/(15*15)
    # Create a 15×15 matrix with all elements set to 1/(15*15)
    return np.ones((15, 15)) / (15*15)

def process_image(input_file):
    """处理图像并保存结果
    Process the image and save the results
    
    参数/Parameters:
        input_file: 输入图像文件路径/path to the input image file
    """
    # 读取图像
    # Read the image
    img = plt.imread(input_file)
    
    # 创建滤波器
    # Create filters
    small_filter = create_small_filter()
    large_filter = create_large_filter()
    
    # 应用卷积
    # Apply convolution
    small_result = sim.convolve(img, small_filter)  # 使用3×3滤波器卷积/Convolve with 3×3 filter
    large_result = sim.convolve(img, large_filter)   # 使用15×15滤波器卷积/Convolve with 15×15 filter
    
    # 显示结果
    # Display results
    plt.figure(figsize=(15, 5))  # 创建15英寸宽5英寸高的图形/Create a figure 15 inches wide and 5 inches tall
    
    # 显示原始图像
    # Display original image
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    
    # 显示3×3滤波结果
    # Display 3×3 filter result
    plt.subplot(1, 3, 2)
    plt.imshow(small_result, cmap='gray')
    plt.title('3×3 Filter Result')
    
    # 显示15×15滤波结果
    # Display 15×15 filter result
    plt.subplot(1, 3, 3)
    plt.imshow(large_result, cmap='gray')
    plt.title('15×15 Filter Result')
    
    plt.tight_layout()  # 自动调整子图参数/Automatically adjust subplot parameters
    plt.show()          # 显示图形/Show the figure

if __name__ == "__main__":
    # 处理指定图像文件
    # Process the specified image file
    process_image('data/bwCat.tif')
