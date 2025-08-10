# -*- coding: utf-8 -*-
"""
@Auth ： 挂科边缘
@File ：train.py
@IDE ：PyCharm
@Motto: 学习新思想，争做新青年
@Email ：179958974@qq.com
"""
import warnings
import os
import json
from datetime import datetime
from ultralytics import YOLO
# 忽略警告
warnings.filterwarnings('ignore')



def save_training_results(results, save_dir='training_results'):
    """
    保存训练结果到文件
    :param results: 训练结果字典
    :param save_dir: 保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_file = os.path.join(save_dir, f'training_results_{timestamp}.txt')
    json_file = os.path.join(save_dir, f'training_results_{timestamp}.json')

    # 保存为文本文件
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("训练结果汇总:\n")
        f.write("=" * 50 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

    # 保存为JSON文件
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"训练结果已保存到: {save_dir}")


if __name__ == '__main__':
    # 初始化模型
    model = YOLO(model=r'D:\yolov11\ultralytics-8.3.111\ultralytics-8.3.111\ultralytics\cfg\models\11\DCCmodel2.yaml')

    # 训练参数配置
    train_args = {
        'data': r'data.yaml',
        'imgsz': 640,
        'epochs': 100,
        'batch': 4,
        'workers': 0,
        'device': '',
        'optimizer': 'SGD',
        'close_mosaic': 10,
        'resume': True,
        'project': 'runs/train',
        'name': 'exp',
        'single_cls': False,
        'cache': False,
    }

    # 开始训练
    results = model.train(**train_args)

    # 收集重要的训练结果
    training_results = {
        'model': 'yolo11',
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'epochs': train_args['epochs'],
        'batch_size': train_args['batch'],
        'optimizer': train_args['optimizer'],
        'final_loss': results.results_dict.get('train/box_loss', 'N/A'),
        'mAP50': results.results_dict.get('metrics/mAP50', 'N/A'),
        'mAP50-95': results.results_dict.get('metrics/mAP50-95', 'N/A'),
        'precision': results.results_dict.get('metrics/precision', 'N/A'),
        'recall': results.results_dict.get('metrics/recall', 'N/A'),
        'save_dir': results.save_dir
    }

    # 保存训练结果
    save_training_results(training_results)

    print("训练完成！模型和日志已保存到:", results.save_dir)
