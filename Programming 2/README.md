cv21b.programming02  物体检测练习

【数据集】
- 471个类别，共39564张图像
- 下载链接：https://pan.baidu.com/s/1pqSYZdsoU_oPzECQW2svXw （提取码：cx8l）
- 训练集
  - 图像数量：23550
  - 图像位置：train/
  - 标注文件：train.json
  - 用途：训练物体检测模型
- 验证集
  - 图像数量：8007
  - 图像位置：val/
  - 标注文件：val.json
  - 用途：使用eval.py进行本地测试
- 测试集
  - 图像数量：8007
  - 图像位置：test/
  - 用途：用于最终测试，因此没有提供标注文件

【评测指标】
- mAP：所有类别上的平均AP
    - AP：在不同Recall下的平均Precision  
参考资料：https://zhuanlan.zhihu.com/p/48693246

【标注文件格式】
{<image_name>:{'height':<height>,'width':<width>,'depth':<depth>,'objects:'{<object_id>:{'category':<category name>,'bbox':\[\<xmin\>,\<ymin\>,\<xmax\>,\<ymax\>\]}}

【任务说明】
1. 使用训练集中的数据训练模型；
2. 使用验证集中的数据调优模型；
3. 采用模型对测试集中的所有图像进行物体检测，提交zip格式，包括：
   - 结果文件命名为“学号.json”，格式同标注文件
   - 汇报幻灯片，命名为“汇报人学号+姓名”
   - 小组构成：小组成员的学号和姓名
