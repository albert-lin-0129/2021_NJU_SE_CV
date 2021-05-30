cv21b.programming03 语义分割练习

【数据集】
- 20个物体类别+背景，共2600张图像（1600:500:500）
- 下载链接：https://pan.baidu.com/s/1eaAyfFYK8e_nUe09C2pobA （提取码：x1dy）
- 文件内容
  - dataset/images：包含训练集+验证集+测试集的2600张图像
  - dataset/annotations_trainval：包含训练集+验证集的2100张语义分割标注
  - train.txt：训练集1600张图像的编号
  - val.txt：验证集500张图片的编号
  - test.txt：测试集500张图片的编号
  - eval.py：评估代码，请修改以下内容：
	  - 预测结果的png文件的目录
	  - 标注文件的png结果的目录
	  - train.txt/val.txt/test.txt文件的所在地址

【评测指标】
平均交并比(mean Intersection Over Union, mIOU)
- mIOU：所有类别上的平均像素交并比
参考资料：https://arxiv.org/pdf/1704.06857.pdf（17页）

【标注文件和预测结果格式】
- 标注文件和预测结果要求存储为png格式图像
- pixel值为对应类别的id（0为background）
- 生成的预测结果图像大小必须与原图像严格一致

【任务说明】
1. 使用训练集中的数据训练模型
2. 对测试集中的所有图像进行语义分割，提交zip格式，包括：
   - 结果文件存储为png格式图像，格式同标注文件
   - 汇报幻灯片，命名为“汇报人学号+姓名”
   - 小组构成：小组成员的学号和姓名

