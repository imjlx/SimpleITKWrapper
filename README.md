# SimpleITK Wrapper

Some helpful functions and classes based on SimpleITK for medical image processing

> 用于医疗图像处理的函数与类，主要基于[SimpleITK](https://simpleitk.org/)。
>
> SimpleITK是一个强大的医疗图像处理包，可以实现医疗图像读取、配准等等处理。虽然官方有详尽的文档，但SimpleITK十分依赖类来处理，几乎把所有的”运算“都包装成了Filter，使用起来比较繁琐。这里把我在SimpleITK中常用的方法，包装成了更简单的函数和类，便于在科研项目间复用代码。
>
> 我写的函数和类自认为逻辑不算复杂，也可以当作SimpleITK的使用示例。
>
> 在处理DICOM文件时，还会利用到pydicom包

## 使用方法：

没有发行过python包，大家就直接clone到本地，然后直接import就行，我暂时就是这么用的（把 `"SimpleITKWrapper 's dir path"`替换成clone下来文件夹的父目录）

```python
import sys
sys.path.append("SimpleITKWrapper 's dir path")
import SimpleITKWrapper as sitkw
```

或者用Git管理仓库是，可以添加子模块：

```git
git submodule add git@github.com:imjlx/SimpleITKWrapper.git
```

## File Structure：

- Basic：数据基础处理
  - Image：医疗图像读取（包括DICOM系列文件的读取、信息提取等）
  - Atlas：分割数据集的一些处理函数
- Registration：有待整理。配准相关
- Resampling：重采样相关
- InfoStat：有待整理。用于在图像中挖掘信息，比如估算体重、器官大小、PET剂量等
- utils：其他相关文件
  - DCMTags：DICOM文件常用MetaData含义
  - OrganDict：分割图像标准器官ID值（暂不知道是哪个标准，反正我都是这么设的）
