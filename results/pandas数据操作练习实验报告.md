# 实验报告 - Pandas 数据操作练习

## 一、实验目的
阐述本次实验的主要目的，可参考任务目的部分。

## 二、实验步骤
详细描述你完成每个任务的步骤和方法，可结合代码进行说明。

### 任务 1: 读取数据
说明你使用的读取数据的函数和过程。

### 任务 2: 查看数据基本信息
描述如何查看数据的基本信息。

### 任务 3: 处理缺失值
说明你找出缺失值列和填充缺失值的方法。

### 任务 4: 数据统计分析
说明你计算数值列均值、中位数和标准差的方法。

### 任务 5: 数据可视化
描述你选择的数值列和绘制直方图的过程。

### 任务 6: 数据保存
说明你保存处理后数据的方法。

## 三、实验结果
展示每个任务的结果，可使用表格或图表进行呈现。
![image](https://github.com/user-attachments/assets/b22fed5d-c630-469e-b42a-f4789e8c2869)

### 任务 1: 读取数据
展示读取的数据的基本信息（如列名、行数等）。
数据基本信息：
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5 entries, 0 to 4
### 任务 2: 查看数据基本信息
粘贴数据的基本信息输出。
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   姓名      5 non-null      object 
 1   年龄      4 non-null      float64
 2   成绩      5 non-null      float64
 3   城市      5 non-null      object 
dtypes: float64(2), object(2)
### 任务 3: 处理缺失值
说明处理后缺失值的情况。
memory usage: 292.0+ bytes
### 任务 4: 数据统计分析
列出数值列的均值、中位数和标准差。
年龄 列的均值: 26.25, 中位数: 26.25, 标准差: 3.031088913245535
成绩 列的均值: 86.8, 中位数: 88.0, 标准差: 5.227332015474051
### 任务 5: 数据可视化
插入绘制的直方图。
![image](https://github.com/user-attachments/assets/d977a968-d926-4df8-beac-49cc79d4cb6d)

### 任务 6: 数据保存
说明保存的文件路径和文件名。
processed_data.csv
## 四、总结
总结本次实验的收获和体会，分析遇到的问题及解决方法，对 Pandas 数据操作的理解和认识。
学会了如何使用 Pandas 读取、处理和分析数据；掌握了数据的基本信息展示（如列名、行数、数据类型等）；理解了如何处理缺失值，包括识别缺失值和用均值填充；学会了如何计算数值列的均值、中位数和标准差；学会了如何将处理后的数据保存为 CSV 文件，便于后续使用。
