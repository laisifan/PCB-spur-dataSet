#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCB缺陷尺寸分析工具 - PyQt5版本
功能：
1. 从CSV文件分析缺陷尺寸并生成图表
2. 从images和labels文件夹生成CSV文件
3. 支持单独或组合输出6种图表
"""

import sys
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QPushButton, QLabel, QFileDialog, QTextEdit,
    QTabWidget, QGridLayout, QScrollArea, QSplitter, QMessageBox,
    QProgressBar, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QImage

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class WorkerThread(QThread):
    """后台工作线程"""
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)

    def __init__(self, task_type, **kwargs):
        super().__init__()
        self.task_type = task_type
        self.kwargs = kwargs

    def run(self):
        try:
            if self.task_type == 'generate_csv':
                self.generate_csv_from_folders()
            elif self.task_type == 'load_csv':
                self.load_csv_data()
        except Exception as e:
            self.error_signal.emit(str(e))

    def generate_csv_from_folders(self):
        """从images和labels文件夹生成CSV"""
        images_folder = self.kwargs['images_folder']
        labels_folder = self.kwargs['labels_folder']
        output_path = self.kwargs['output_path']

        self.log_signal.emit("=" * 60)
        self.log_signal.emit("开始从YOLO标签生成CSV文件...")
        self.log_signal.emit("=" * 60)

        # 支持的图片格式（统一小写比较）
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

        # 获取所有图片文件（遍历一次，避免重复）
        image_files = [
            f for f in Path(images_folder).iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if not image_files:
            self.error_signal.emit("未找到任何图片文件！")
            return

        self.log_signal.emit(f"找到 {len(image_files)} 个图片文件")

        # 存储所有缺陷数据
        all_defects = []
        processed = 0
        skipped = 0

        for idx, img_path in enumerate(image_files):
            # 更新进度
            progress = int((idx + 1) / len(image_files) * 100)
            self.progress_signal.emit(progress)

            # 查找对应的标签文件
            label_path = Path(labels_folder) / (img_path.stem + '.txt')

            if not label_path.exists():
                skipped += 1
                continue

            # 读取图片尺寸
            try:
                from PIL import Image
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                self.log_signal.emit(f"⚠️ 无法读取图片 {img_path.name}: {e}")
                skipped += 1
                continue

            # 读取标签文件
            try:
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                for line_idx, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    # YOLO格式: class x_center y_center width height (归一化值)
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    norm_width = float(parts[3])
                    norm_height = float(parts[4])

                    # 转换为像素值
                    bbox_width = norm_width * img_width
                    bbox_height = norm_height * img_height
                    area = bbox_width * bbox_height

                    # 计算实际的bbox坐标
                    x1 = (x_center - norm_width / 2) * img_width
                    y1 = (y_center - norm_height / 2) * img_height
                    x2 = (x_center + norm_width / 2) * img_width
                    y2 = (y_center + norm_height / 2) * img_height

                    all_defects.append({
                        'Image': img_path.name,
                        'Image_Width': img_width,
                        'Image_Height': img_height,
                        'Class': class_id,
                        'X_Center': x_center * img_width,
                        'Y_Center': y_center * img_height,
                        'BBox_Width': bbox_width,
                        'BBox_Height': bbox_height,
                        'Area': area,
                        'X1': x1,
                        'Y1': y1,
                        'X2': x2,
                        'Y2': y2
                    })

                processed += 1

            except Exception as e:
                self.log_signal.emit(f"⚠️ 处理标签文件失败 {label_path.name}: {e}")
                skipped += 1
                continue

        self.log_signal.emit(f"\n处理完成:")
        self.log_signal.emit(f"  ✓ 成功处理: {processed} 个图片")
        self.log_signal.emit(f"  ⚠️ 跳过: {skipped} 个图片")
        self.log_signal.emit(f"  📊 总缺陷数: {len(all_defects)}")

        if all_defects:
            # 创建DataFrame并保存
            df = pd.DataFrame(all_defects)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            self.log_signal.emit(f"\n✓ CSV文件已保存: {output_path}")
            self.finished_signal.emit({'success': True, 'path': output_path, 'count': len(all_defects)})
        else:
            self.error_signal.emit("未找到任何有效的缺陷数据！")

    def load_csv_data(self):
        """加载CSV数据"""
        csv_path = self.kwargs['csv_path']

        self.log_signal.emit(f"正在加载CSV文件: {csv_path}")

        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312']
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                self.log_signal.emit(f"✓ 成功使用 {encoding} 编码读取文件")
                break
            except:
                continue

        if df is None:
            self.error_signal.emit("无法读取CSV文件，请检查文件格式！")
            return

        self.log_signal.emit(f"总记录数: {len(df)}")
        self.log_signal.emit(f"列名: {df.columns.tolist()}")

        # 清洗列名
        df.columns = df.columns.str.strip()

        # 尝试找到宽度、高度、面积列
        width_col = None
        height_col = None
        area_col = None

        # 优先通过列名查找
        for col in df.columns:
            col_lower = col.lower()
            if 'width' in col_lower and 'image' not in col_lower:
                width_col = col
            elif 'height' in col_lower and 'image' not in col_lower:
                height_col = col
            elif 'area' in col_lower:
                area_col = col

        # 如果通过列名找不到，尝试使用列索引
        if width_col is None or height_col is None or area_col is None:
            try:
                if len(df.columns) >= 8:
                    width_col = df.columns[6] if width_col is None else width_col
                    height_col = df.columns[7] if height_col is None else height_col
                    area_col = df.columns[8] if area_col is None else area_col
                elif len(df.columns) >= 3:
                    # 假设简单格式：宽度、高度、面积
                    width_col = df.columns[0] if width_col is None else width_col
                    height_col = df.columns[1] if height_col is None else height_col
                    area_col = df.columns[2] if area_col is None else area_col
            except:
                pass

        if width_col is None or height_col is None or area_col is None:
            self.error_signal.emit("无法识别CSV文件中的宽度、高度、面积列！")
            return

        self.log_signal.emit(f"识别到的列 - 宽度: {width_col}, 高度: {height_col}, 面积: {area_col}")

        # 提取数据
        widths = pd.to_numeric(df[width_col], errors='coerce').dropna()
        heights = pd.to_numeric(df[height_col], errors='coerce').dropna()
        areas = pd.to_numeric(df[area_col], errors='coerce').dropna()

        self.log_signal.emit(f"有效数据: 宽度 {len(widths)}, 高度 {len(heights)}, 面积 {len(areas)}")

        self.finished_signal.emit({
            'success': True,
            'widths': widths,
            'heights': heights,
            'areas': areas,
            'df': df
        })


class ChartGenerator:
    """图表生成器"""

    @staticmethod
    def create_width_histogram(widths, fig=None, ax=None):
        """创建宽度分布直方图"""
        if fig is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.clear()
        ax.hist(widths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(widths.mean(), color='red', linestyle='--',
                   label=f'均值: {widths.mean():.1f}')
        ax.axvline(widths.median(), color='green', linestyle='--',
                   label=f'中位数: {widths.median():.1f}')
        ax.set_xlabel('宽度 (像素)', fontsize=12)
        ax.set_ylabel('数量', fontsize=12)
        ax.set_title('缺陷宽度分布', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        return fig, ax

    @staticmethod
    def create_height_histogram(heights, fig=None, ax=None):
        """创建高度分布直方图"""
        if fig is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.clear()
        ax.hist(heights, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
        ax.axvline(heights.mean(), color='red', linestyle='--',
                   label=f'均值: {heights.mean():.1f}')
        ax.axvline(heights.median(), color='green', linestyle='--',
                   label=f'中位数: {heights.median():.1f}')
        ax.set_xlabel('高度 (像素)', fontsize=12)
        ax.set_ylabel('数量', fontsize=12)
        ax.set_title('缺陷高度分布', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        return fig, ax

    @staticmethod
    def create_area_histogram(areas, fig=None, ax=None):
        """创建面积分布直方图"""
        if fig is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.clear()
        ax.hist(areas, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
        ax.axvline(areas.mean(), color='red', linestyle='--',
                   label=f'均值: {areas.mean():.1f}')
        ax.axvline(areas.median(), color='green', linestyle='--',
                   label=f'中位数: {areas.median():.1f}')
        ax.set_xlabel('面积 (像素²)', fontsize=12)
        ax.set_ylabel('数量', fontsize=12)
        ax.set_title('缺陷面积分布', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        return fig, ax

    @staticmethod
    def create_scatter_plot(widths, heights, fig=None, ax=None):
        """创建宽高散点图"""
        if fig is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.clear()
        ax.scatter(widths, heights, alpha=0.3, s=10, c='blue')
        ax.set_xlabel('宽度 (像素)', fontsize=12)
        ax.set_ylabel('高度 (像素)', fontsize=12)
        ax.set_title('宽度-高度散点图', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        return fig, ax

    @staticmethod
    def create_pie_chart(widths, heights, fig=None, ax=None):
        """创建尺寸分类饼图"""
        if fig is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.clear()

        max_dim = np.maximum(widths, heights)
        tiny = (max_dim <= 16).sum()
        small = ((max_dim > 16) & (max_dim <= 32)).sum()
        medium = ((max_dim > 32) & (max_dim <= 48)).sum()
        large = (max_dim > 48).sum()

        size_labels = ['极小\n(<16px)', '小\n(16-32px)', '中\n(32-48px)', '大\n(≥48px)']
        size_values = [tiny, small, medium, large]
        colors = ['#ff9999', '#ffcc99', '#99ccff', '#99ff99']

        # 过滤掉为0的值
        non_zero_labels = []
        non_zero_values = []
        non_zero_colors = []
        for i, v in enumerate(size_values):
            if v > 0:
                non_zero_labels.append(size_labels[i])
                non_zero_values.append(v)
                non_zero_colors.append(colors[i])

        if non_zero_values:
            ax.pie(non_zero_values, labels=non_zero_labels, colors=non_zero_colors,
                   autopct='%1.1f%%', startangle=90)
        ax.set_title('按最大边长分类', fontsize=14, fontweight='bold')

        return fig, ax

    @staticmethod
    def create_cumulative_plot(widths, fig=None, ax=None):
        """创建累积分布图"""
        if fig is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.clear()

        sorted_widths = np.sort(widths)
        cumulative = np.arange(1, len(sorted_widths) + 1) / len(sorted_widths) * 100

        ax.plot(sorted_widths, cumulative, linewidth=2, color='blue')
        ax.axvline(32, color='red', linestyle='--', label='YOLO小目标阈值(32px)')
        ax.set_xlabel('宽度 (像素)', fontsize=12)
        ax.set_ylabel('累积百分比 (%)', fontsize=12)
        ax.set_title('缺陷宽度累积分布', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        return fig, ax

    @staticmethod
    def create_combined_figure(widths, heights, areas):
        """创建组合图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PCB毛刺缺陷尺寸分布分析', fontsize=16, fontweight='bold')

        ChartGenerator.create_width_histogram(widths, fig, axes[0, 0])
        ChartGenerator.create_height_histogram(heights, fig, axes[0, 1])
        ChartGenerator.create_area_histogram(areas, fig, axes[0, 2])
        ChartGenerator.create_scatter_plot(widths, heights, fig, axes[1, 0])
        ChartGenerator.create_pie_chart(widths, heights, fig, axes[1, 1])
        ChartGenerator.create_cumulative_plot(widths, fig, axes[1, 2])

        plt.tight_layout()
        return fig


class PCBDefectAnalyzer(QMainWindow):
    """PCB缺陷尺寸分析工具主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCB缺陷尺寸分析工具 v1.0")
        self.setMinimumSize(1200, 800)

        # 数据存储
        self.widths = None
        self.heights = None
        self.areas = None
        self.csv_path = None
        self.images_folder = None
        self.labels_folder = None

        self.init_ui()

    def init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # 左侧控制面板
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)

        # 右侧显示区域
        right_panel = self.create_display_panel()
        splitter.addWidget(right_panel)

        # 设置分割比例
        splitter.setSizes([350, 850])

    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        # ===== CSV文件选择组 =====
        csv_group = QGroupBox("📊 CSV文件分析")
        csv_layout = QVBoxLayout(csv_group)

        self.csv_label = QLabel("未选择CSV文件")
        self.csv_label.setWordWrap(True)
        csv_layout.addWidget(self.csv_label)

        btn_select_csv = QPushButton("📂 选择CSV文件")
        btn_select_csv.clicked.connect(self.select_csv_file)
        btn_select_csv.setMinimumHeight(35)
        csv_layout.addWidget(btn_select_csv)

        layout.addWidget(csv_group)

        # ===== 图表生成组 =====
        chart_group = QGroupBox("📈 图表生成")
        chart_layout = QGridLayout(chart_group)

        chart_buttons = [
            ("📊 宽度分布直方图", self.show_width_histogram),
            ("📊 高度分布直方图", self.show_height_histogram),
            ("📊 面积分布直方图", self.show_area_histogram),
            ("⚪ 宽高散点图", self.show_scatter_plot),
            ("🥧 尺寸分类饼图", self.show_pie_chart),
            ("📈 累积分布图", self.show_cumulative_plot),
        ]

        for i, (text, callback) in enumerate(chart_buttons):
            btn = QPushButton(text)
            btn.clicked.connect(callback)
            btn.setMinimumHeight(35)
            chart_layout.addWidget(btn, i // 2, i % 2)

        # 组合图表按钮
        btn_combined = QPushButton("🖼️ 生成组合图表（6合1）")
        btn_combined.clicked.connect(self.show_combined_chart)
        btn_combined.setMinimumHeight(40)
        btn_combined.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        chart_layout.addWidget(btn_combined, 3, 0, 1, 2)

        # 保存按钮
        btn_save = QPushButton("💾 保存当前图表")
        btn_save.clicked.connect(self.save_current_chart)
        btn_save.setMinimumHeight(35)
        chart_layout.addWidget(btn_save, 4, 0, 1, 2)

        layout.addWidget(chart_group)

        # ===== YOLO标签转CSV组 =====
        yolo_group = QGroupBox("🏷️ YOLO标签生成CSV")
        yolo_layout = QVBoxLayout(yolo_group)

        # Images文件夹
        self.images_label = QLabel("未选择Images文件夹")
        self.images_label.setWordWrap(True)
        yolo_layout.addWidget(self.images_label)

        btn_select_images = QPushButton("📁 选择Images文件夹")
        btn_select_images.clicked.connect(self.select_images_folder)
        btn_select_images.setMinimumHeight(35)
        yolo_layout.addWidget(btn_select_images)

        # Labels文件夹
        self.labels_label = QLabel("未选择Labels文件夹")
        self.labels_label.setWordWrap(True)
        yolo_layout.addWidget(self.labels_label)

        btn_select_labels = QPushButton("📁 选择Labels文件夹")
        btn_select_labels.clicked.connect(self.select_labels_folder)
        btn_select_labels.setMinimumHeight(35)
        yolo_layout.addWidget(btn_select_labels)

        # 生成CSV按钮
        btn_generate_csv = QPushButton("⚙️ 生成CSV文件")
        btn_generate_csv.clicked.connect(self.generate_csv_from_yolo)
        btn_generate_csv.setMinimumHeight(40)
        btn_generate_csv.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        yolo_layout.addWidget(btn_generate_csv)

        layout.addWidget(yolo_group)

        # ===== 进度条 =====
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # ===== 日志区域 =====
        log_group = QGroupBox("📝 日志输出")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setMinimumHeight(150)
        log_layout.addWidget(self.log_text)

        btn_clear_log = QPushButton("🗑️ 清空日志")
        btn_clear_log.clicked.connect(lambda: self.log_text.clear())
        log_layout.addWidget(btn_clear_log)

        layout.addWidget(log_group)

        # 弹性空间
        layout.addStretch()

        return panel

    def create_display_panel(self):
        """创建显示面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 标题
        title_label = QLabel("图表预览区域")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title_label)

        # 图表画布
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.canvas)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        # 统计信息标签
        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        layout.addWidget(self.stats_label)

        return panel

    def log(self, message):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # 滚动到底部
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def select_csv_file(self):
        """选择CSV文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择CSV文件", "", "CSV文件 (*.csv);;所有文件 (*)"
        )

        if file_path:
            self.csv_path = file_path
            self.csv_label.setText(f"已选择: {Path(file_path).name}")
            self.log(f"选择CSV文件: {file_path}")
            self.load_csv_data()

    def load_csv_data(self):
        """加载CSV数据"""
        if not self.csv_path:
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度

        self.worker = WorkerThread('load_csv', csv_path=self.csv_path)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_csv_loaded)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def on_csv_loaded(self, result):
        """CSV加载完成回调"""
        self.progress_bar.setVisible(False)

        if result.get('success'):
            self.widths = result['widths']
            self.heights = result['heights']
            self.areas = result['areas']

            # 更新统计信息
            stats_text = f"""
            <b>数据统计摘要:</b><br>
            • 缺陷总数: {len(self.widths)}<br>
            • 宽度范围: {self.widths.min():.1f} - {self.widths.max():.1f} 像素 (均值: {self.widths.mean():.1f})<br>
            • 高度范围: {self.heights.min():.1f} - {self.heights.max():.1f} 像素 (均值: {self.heights.mean():.1f})<br>
            • 面积范围: {self.areas.min():.1f} - {self.areas.max():.1f} 像素² (均值: {self.areas.mean():.1f})
            """
            self.stats_label.setText(stats_text)
            self.log("✓ CSV数据加载完成，可以生成图表了！")

            # 自动显示组合图表
            self.show_combined_chart()

    def on_error(self, error_msg):
        """错误回调"""
        self.progress_bar.setVisible(False)
        self.log(f"❌ 错误: {error_msg}")
        QMessageBox.critical(self, "错误", error_msg)

    def check_data_loaded(self):
        """检查数据是否已加载"""
        if self.widths is None or self.heights is None or self.areas is None:
            QMessageBox.warning(self, "提示", "请先选择并加载CSV文件！")
            return False
        return True

    def show_width_histogram(self):
        """显示宽度直方图"""
        if not self.check_data_loaded():
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ChartGenerator.create_width_histogram(self.widths, self.figure, ax)
        self.canvas.draw()
        self.log("显示宽度分布直方图")

    def show_height_histogram(self):
        """显示高度直方图"""
        if not self.check_data_loaded():
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ChartGenerator.create_height_histogram(self.heights, self.figure, ax)
        self.canvas.draw()
        self.log("显示高度分布直方图")

    def show_area_histogram(self):
        """显示面积直方图"""
        if not self.check_data_loaded():
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ChartGenerator.create_area_histogram(self.areas, self.figure, ax)
        self.canvas.draw()
        self.log("显示面积分布直方图")

    def show_scatter_plot(self):
        """显示散点图"""
        if not self.check_data_loaded():
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ChartGenerator.create_scatter_plot(self.widths, self.heights, self.figure, ax)
        self.canvas.draw()
        self.log("显示宽高散点图")

    def show_pie_chart(self):
        """显示饼图"""
        if not self.check_data_loaded():
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ChartGenerator.create_pie_chart(self.widths, self.heights, self.figure, ax)
        self.canvas.draw()
        self.log("显示尺寸分类饼图")

    def show_cumulative_plot(self):
        """显示累积分布图"""
        if not self.check_data_loaded():
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ChartGenerator.create_cumulative_plot(self.widths, self.figure, ax)
        self.canvas.draw()
        self.log("显示累积分布图")

    def show_combined_chart(self):
        """显示组合图表"""
        if not self.check_data_loaded():
            return

        self.figure.clear()

        # 创建2x3子图
        axes = self.figure.subplots(2, 3)
        self.figure.suptitle('PCB毛刺缺陷尺寸分布分析', fontsize=14, fontweight='bold')

        ChartGenerator.create_width_histogram(self.widths, self.figure, axes[0, 0])
        ChartGenerator.create_height_histogram(self.heights, self.figure, axes[0, 1])
        ChartGenerator.create_area_histogram(self.areas, self.figure, axes[0, 2])
        ChartGenerator.create_scatter_plot(self.widths, self.heights, self.figure, axes[1, 0])
        ChartGenerator.create_pie_chart(self.widths, self.heights, self.figure, axes[1, 1])
        ChartGenerator.create_cumulative_plot(self.widths, self.figure, axes[1, 2])

        self.figure.tight_layout()
        self.canvas.draw()
        self.log("显示组合图表（6合1）")

    def save_current_chart(self):
        """保存当前图表"""
        if self.widths is None:
            QMessageBox.warning(self, "提示", "没有可保存的图表！")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图表", "defect_analysis.png",
            "PNG图片 (*.png);;JPEG图片 (*.jpg);;PDF文档 (*.pdf);;所有文件 (*)"
        )

        if file_path:
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
            self.log(f"✓ 图表已保存: {file_path}")
            QMessageBox.information(self, "成功", f"图表已保存至:\n{file_path}")

    def select_images_folder(self):
        """选择Images文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择Images文件夹")
        if folder:
            self.images_folder = folder
            self.images_label.setText(f"已选择: {Path(folder).name}")
            self.log(f"选择Images文件夹: {folder}")

    def select_labels_folder(self):
        """选择Labels文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择Labels文件夹")
        if folder:
            self.labels_folder = folder
            self.labels_label.setText(f"已选择: {Path(folder).name}")
            self.log(f"选择Labels文件夹: {folder}")

    def generate_csv_from_yolo(self):
        """从YOLO标签生成CSV"""
        if not self.images_folder:
            QMessageBox.warning(self, "提示", "请先选择Images文件夹！")
            return

        if not self.labels_folder:
            QMessageBox.warning(self, "提示", "请先选择Labels文件夹！")
            return

        # 选择保存位置
        output_path, _ = QFileDialog.getSaveFileName(
            self, "保存CSV文件", "defect_sizes.csv", "CSV文件 (*.csv)"
        )

        if not output_path:
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.worker = WorkerThread(
            'generate_csv',
            images_folder=self.images_folder,
            labels_folder=self.labels_folder,
            output_path=output_path
        )
        self.worker.log_signal.connect(self.log)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.finished_signal.connect(self.on_csv_generated)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def on_csv_generated(self, result):
        """CSV生成完成回调"""
        self.progress_bar.setVisible(False)

        if result.get('success'):
            self.log(f"✓ 成功生成CSV文件，包含 {result['count']} 条缺陷记录")

            reply = QMessageBox.question(
                self, "成功",
                f"CSV文件已生成！\n包含 {result['count']} 条缺陷记录\n\n是否立即加载此文件进行分析？",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.csv_path = result['path']
                self.csv_label.setText(f"已选择: {Path(result['path']).name}")
                self.load_csv_data()


def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置应用样式
    app.setStyle('Fusion')

    # 创建主窗口
    window = PCBDefectAnalyzer()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()