import numpy as np
import scipy.ndimage as sim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_stress_fibers():
    """
    加载应力纤维数据
    
    返回:
        numpy.ndarray: 包含应力纤维数据的二维数组
    """
    return np.loadtxt("data/stressFibers.txt")

def create_gauss_filter():
    """
    创建高斯滤波器
    
    返回:
        numpy.ndarray: 51x51的高斯滤波器矩阵
    """
    x = np.arange(-25, 26)
    y = np.arange(-25, 26)
    X, Y = np.meshgrid(x, y)
    gauss_filter = np.exp(-0.5 * (X**2 / 5 + Y**2 / 45))
    return gauss_filter

def create_combined_filter(gauss_filter):
    """
    创建高斯-拉普拉斯组合滤波器
    
    参数:
        gauss_filter: 高斯滤波器矩阵
        
    返回:
        numpy.ndarray: 组合后的滤波器矩阵
    """
    laplacian_filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    combined_filter = sim.convolve(gauss_filter, laplacian_filter, mode='constant', cval=0.0)
    return combined_filter

def plot_filter_surface(filter, title):
    """
    绘制滤波器3D表面图
    
    参数:
        filter: 要绘制的滤波器矩阵
        title: 图形标题(使用英文) 
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(filter.shape[0])
    y = np.arange(filter.shape[1])
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, filter, cmap='viridis')
    ax.set_title(title)
    plt.show()

def process_and_display(stressFibers, filter, vmax_ratio=0.5):
    """
    处理图像并显示结果
    
    参数:
        stressFibers: 输入图像数据
        filter: 要应用的滤波器
        vmax_ratio: 显示时的最大强度比例(默认0.5)
        
    返回:
        numpy.ndarray: 处理后的图像数据
    """
    processed_image = sim.convolve(stressFibers, filter, mode='constant', cval=0.0)
    plt.imshow(processed_image, cmap='gray', vmin=0, vmax=vmax_ratio * processed_image.max())
    plt.colorbar()
    plt.title("Processed Image")
    plt.show()
    return processed_image

def main():
    """
    主函数，执行完整的图像特征强调流程
    """
    # 加载数据
    stressFibers = load_stress_fibers()
    
    # 任务(a): 创建并显示高斯滤波器
    gauss_filter = create_gauss_filter()
    plt.imshow(gauss_filter, cmap='gray')
    plt.colorbar()
    plt.title("Gaussian Filter")
    plt.show()
    plot_filter_surface(gauss_filter, "Gaussian Filter Surface")
    
    # 任务(b): 创建组合滤波器并比较
    combined_filter = create_combined_filter(gauss_filter)
    plt.imshow(combined_filter, cmap='gray')
    plt.colorbar()
    plt.title("Combined Filter")
    plt.show()
    plot_filter_surface(combined_filter, "Combined Filter Surface")
    
    # 任务(c): 应用垂直滤波器
    vertical_filter = np.array([[1], [-1]])  # 简单的垂直方向滤波器
    vertical_result = process_and_display(stressFibers, vertical_filter)
    
    # 任务(d): 应用水平滤波器
    horizontal_filter = np.array([[1, -1]])  # 简单的水平方向滤波器
    horizontal_result = process_and_display(stressFibers, horizontal_filter)
    
    # 选做: 45度方向滤波器
    diagonal_filter = np.array([[1, 0], [0, -1]])  # 简单的对角方向滤波器
    diagonal_result = process_and_display(stressFibers, diagonal_filter)

if __name__ == "__main__":
    main()