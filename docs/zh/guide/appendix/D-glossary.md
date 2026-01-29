---
title: 附录D：术语表
description: 介绍常见术语的英文译名与中文解释。
---

# 附录D：术语表

|英文术语|中文译名|解释|
| --------------------------------| ---------------------| ----------------------------------------------------------------------|
|k-space|k 空间|在 MRI 中指采集的原始频域数据矩阵，通过傅里叶变换生成图像。|
|T1-weighted|T1 加权|MRI 序列一种对比方式，强调纵向弛豫特性，结构清晰。|
|T2-weighted|T2 加权|MRI 序列另一种对比方式，强调横向弛豫特性，液体／水肿显示明显。|
|Windowing (Window Width/Level)|灰度窗／窗宽窗位|在 CT 或 X 光影像显示时，用于调整灰度范围与中点优化对比。|
|Flat-field correction|平场校正|校正影像中探测器／照明不均或响应不一致导致的暗斑或条纹伪影。|
|SENSE (Sensitivity Encoding)|SENSE（灵敏度编码）|MRI 并行成像技术，利用多个线圈敏感度信息实现扫描加速。|
|Parallel Imaging|并行成像|使用多个接收线圈同步采样以减少扫描时间或提高分辨率。|
|Echo Time (TE)|回波时间|MRI 中 RF 脉冲到信号采集起始之间的时间间隔，影响图像对比。|
|Repetition Time (TR)|重复时间|MRI 中脉冲循环间隔时间，影响信号恢复和对比。|
|Flip Angle|翻转角|MRI 射频脉冲使核磁共振倾斜角度，影响信号强度／对比。|
|Field of View (FOV)|视野|成像覆盖的解剖区域或体积大小，通常以 mm 表示。|
|Voxel|体素|三维像素，医学影像数据中的最小单位。|
|Slice Thickness|切片厚度|在断层扫描（CT／MRI）中每张图像的厚度，影响体积重建与部分体积效应。|
|Partial Volume Effect|部分体积效应|当一个体素包含多种组织类型时，信号混合导致精度降低。|
|Artifact|伪影／伪像|图像中由设备、患者运动、信号处理等引入的非真实组织结构。|
|Signal-to-Noise Ratio (SNR)|信噪比|图像中信号强度与噪声强度之比，衡量图像质量。|
|Contrast-to-Noise Ratio (CNR)|对比噪声比|病灶与背景组织之间信号差距与噪声之比，反映可识别性。|
|Reconstruction Kernel|重建滤波核|CT 重建时使用的滤波器类型，会影响锐利度和噪声水平。|
|Radiomics|放射组学|从影像中提取大量定量特征（纹理、形状、强度等）用于建模分析。|
|Segmentation|分割|将影像像素／体素分类为不同结构（如器官、水肿、肿瘤）以进行定量分析。|
|Registration|配准／注册|将来自不同时间点或不同模态的影像对齐至同一空间以方便比较。|
|Deep Learning|深度学习|使用深度神经网络（如 CNN、Transformer）对影像数据进行分析／预测。|
|Transfer Learning|迁移学习|将一个任务训练好的模型应用于另一个相关任务以减少训练成本。|
|Multimodal Learning|多模态学习|融合多种影像模态（如 MRI + PET + 基因数据）进行联合分析。|
|Overfitting|过拟合|模型在训练集上表现很好但在新数据上泛化能力差的现象。|
|Dice Coefficient|Dice 系数|衡量分割任务中预测与真实区域重叠程度的指标。|
|Intersection over Union (IoU)|交并比|用于分割评价：预测与真实区域交集／并集比值。|
|Sensitivity (Recall)|敏感性／召回率|分类任务中真正例被正确识别的比例。|
|Specificity|特异性|真负例被正确识别的比例。|
|ROC Curve|ROC 曲线|显示假阳性率与真阳性率关系的曲线，用于分类评估。|
|Hounsfield Unit (HU)|Hounsfield 单位|CT 中表示组织密度的单位：水为 0，空气约 –1000。|
|Contrast Agent|造影剂|在成像前注入或服用用于增强某些组织或血管对比。|
|Dual-energy CT|双能量 CT|利用两种不同能量的 X-射线对比增强成像材质分辨能力。|
|Ground-glass Opacity (GGO)|磨玻璃影|肺部影像中轻度增密、但血管纹理仍可见的不透明征象。|
|Mass|肿块|较大体积（通常 \>3 cm）可能异常结构或肿瘤，需要观察与鉴别。|
|Nodule|结节|较小体积（通常 \<3 cm）结构，需随访是否增长或恶变。|
|Pleural Effusion|胸腔积液|胸膜腔内液体积聚，在影像中表现为液平面或模糊边界。|
|Cardiomegaly|心脏扩大|心影/胸廓比增大，提示心脏结构异常。|
|Calcification|钙化|组织或病灶内钙沉积，在 CT 或 X 光中高密度可见。|
|Infarct|梗死|血供中断导致组织坏死，在 MRI／CT 中有典型影像表现。|
|Ring Enhancement|环状强化|对比增强影像中病灶边缘强化而中心低信号／低密度，常见转移瘤或脑脓肿。|
|Spiculated Margin|刺状边缘|肿物边缘呈辐射状尖刺突起，提示侵袭性或恶性可能。|
|Ghosting|幽影／重影|MRI 图像中特别在相位编码方向，由患者运动等引起的条纹或重复影。|
|Flip Angle|翻转角|MRI 射频脉冲使核磁共振倾斜角度，影响信号强度／对比。|
|Fourier Transform|傅里叶变换|将空间域信号转换为频率域（如 k-空间到图像域）的数学变换。|
|Coil Sensitivity|线圈敏感度|多通道 MRI 接收线圈不同空间位置的响应差异，用于并行成像。|
|Nyquist Frequency|奈奎斯特频率|采样理论中最低必须采样频率的一半，低于该频率会发生混叠。|
|Aliasing|混叠（重叠伪影）|当采样不足或 FOV 太小时，影像外部信号在图像中被重复或重叠。|
