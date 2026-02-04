#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCB Defect Size Analysis Tool - PyQt5 Version
Features:
1. Analyze defect sizes from CSV files and generate charts
2. Generate CSV files from images and labels folders
3. Support individual or combined output of 6 types of charts
"""

import sys
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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

# Set English fonts
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


class WorkerThread(QThread):
    """Background worker thread"""
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
        """Generate CSV from images and labels folders"""
        images_folder = self.kwargs['images_folder']
        labels_folder = self.kwargs['labels_folder']
        output_path = self.kwargs['output_path']

        self.log_signal.emit("=" * 60)
        self.log_signal.emit("Starting CSV generation from YOLO labels...")
        self.log_signal.emit("=" * 60)

        # Supported image formats (unified lowercase comparison)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

        # Get all image files (iterate once, avoid duplication)
        image_files = [
            f for f in Path(images_folder).iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if not image_files:
            self.error_signal.emit("No image files found!")
            return

        self.log_signal.emit(f"Found {len(image_files)} image files")

        # Store all defect data
        all_defects = []
        processed = 0
        skipped = 0

        for idx, img_path in enumerate(image_files):
            # Update progress
            progress = int((idx + 1) / len(image_files) * 100)
            self.progress_signal.emit(progress)

            # Find corresponding label file
            label_path = Path(labels_folder) / (img_path.stem + '.txt')

            if not label_path.exists():
                skipped += 1
                continue

            # Read image dimensions
            try:
                from PIL import Image
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                self.log_signal.emit(f"⚠️ Unable to read image {img_path.name}: {e}")
                skipped += 1
                continue

            # Read label file
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
                    # YOLO format: class x_center y_center width height (normalized values)
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    norm_width = float(parts[3])
                    norm_height = float(parts[4])

                    # Convert to pixel values
                    bbox_width = norm_width * img_width
                    bbox_height = norm_height * img_height
                    area = bbox_width * bbox_height

                    # Calculate actual bbox coordinates
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
                self.log_signal.emit(f"⚠️ Failed to process label file {label_path.name}: {e}")
                skipped += 1
                continue

        self.log_signal.emit(f"\nProcessing complete:")
        self.log_signal.emit(f"  ✓ Successfully processed: {processed} images")
        self.log_signal.emit(f"  ⚠️ Skipped: {skipped} images")
        self.log_signal.emit(f"  📊 Total defects: {len(all_defects)}")

        if all_defects:
            # Create DataFrame and save
            df = pd.DataFrame(all_defects)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            self.log_signal.emit(f"\n✓ CSV file saved: {output_path}")
            self.finished_signal.emit({'success': True, 'path': output_path, 'count': len(all_defects)})
        else:
            self.error_signal.emit("No valid defect data found!")

    def load_csv_data(self):
        """Load CSV data"""
        csv_path = self.kwargs['csv_path']

        self.log_signal.emit(f"Loading CSV file: {csv_path}")

        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312']
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                self.log_signal.emit(f"✓ Successfully read file using {encoding} encoding")
                break
            except:
                continue

        if df is None:
            self.error_signal.emit("Unable to read CSV file, please check file format!")
            return

        self.log_signal.emit(f"Total records: {len(df)}")
        self.log_signal.emit(f"Column names: {df.columns.tolist()}")

        # Clean column names
        df.columns = df.columns.str.strip()

        # Try to find width, height, area columns
        width_col = None
        height_col = None
        area_col = None

        # First try to find by column names
        for col in df.columns:
            col_lower = col.lower()
            if 'width' in col_lower and 'image' not in col_lower:
                width_col = col
            elif 'height' in col_lower and 'image' not in col_lower:
                height_col = col
            elif 'area' in col_lower:
                area_col = col

        # If not found by column names, try using column indices
        if width_col is None or height_col is None or area_col is None:
            try:
                if len(df.columns) >= 8:
                    width_col = df.columns[6] if width_col is None else width_col
                    height_col = df.columns[7] if height_col is None else height_col
                    area_col = df.columns[8] if area_col is None else area_col
                elif len(df.columns) >= 3:
                    # Assume simple format: width, height, area
                    width_col = df.columns[0] if width_col is None else width_col
                    height_col = df.columns[1] if height_col is None else height_col
                    area_col = df.columns[2] if area_col is None else area_col
            except:
                pass

        if width_col is None or height_col is None or area_col is None:
            self.error_signal.emit("Cannot identify width, height, area columns in CSV file!")
            return

        self.log_signal.emit(f"Identified columns - Width: {width_col}, Height: {height_col}, Area: {area_col}")

        # Extract data
        widths = pd.to_numeric(df[width_col], errors='coerce').dropna()
        heights = pd.to_numeric(df[height_col], errors='coerce').dropna()
        areas = pd.to_numeric(df[area_col], errors='coerce').dropna()

        self.log_signal.emit(f"Valid data: Width {len(widths)}, Height {len(heights)}, Area {len(areas)}")

        self.finished_signal.emit({
            'success': True,
            'widths': widths,
            'heights': heights,
            'areas': areas,
            'df': df
        })


class ChartGenerator:
    """Chart generator"""

    @staticmethod
    def create_width_histogram(widths, fig=None, ax=None):
        """Create width distribution histogram"""
        if fig is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.clear()
        ax.hist(widths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(widths.mean(), color='red', linestyle='--',
                   label=f'Mean: {widths.mean():.1f}')
        ax.axvline(widths.median(), color='green', linestyle='--',
                   label=f'Median: {widths.median():.1f}')
        ax.set_xlabel('Width (pixels)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Defect Width Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        return fig, ax

    @staticmethod
    def create_height_histogram(heights, fig=None, ax=None):
        """Create height distribution histogram"""
        if fig is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.clear()
        ax.hist(heights, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
        ax.axvline(heights.mean(), color='red', linestyle='--',
                   label=f'Mean: {heights.mean():.1f}')
        ax.axvline(heights.median(), color='green', linestyle='--',
                   label=f'Median: {heights.median():.1f}')
        ax.set_xlabel('Height (pixels)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Defect Height Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        return fig, ax

    @staticmethod
    def create_area_histogram(areas, fig=None, ax=None):
        """Create area distribution histogram"""
        if fig is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.clear()
        ax.hist(areas, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
        ax.axvline(areas.mean(), color='red', linestyle='--',
                   label=f'Mean: {areas.mean():.1f}')
        ax.axvline(areas.median(), color='green', linestyle='--',
                   label=f'Median: {areas.median():.1f}')
        ax.set_xlabel('Area (pixels²)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Defect Area Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        return fig, ax

    @staticmethod
    def create_scatter_plot(widths, heights, fig=None, ax=None):
        """Create width-height scatter plot"""
        if fig is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.clear()
        ax.scatter(widths, heights, alpha=0.3, s=10, c='blue')
        ax.set_xlabel('Width (pixels)', fontsize=12)
        ax.set_ylabel('Height (pixels)', fontsize=12)
        ax.set_title('Width-Height Scatter Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        return fig, ax

    @staticmethod
    def create_pie_chart(widths, heights, fig=None, ax=None):
        """Create size classification pie chart"""
        if fig is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.clear()

        max_dim = np.maximum(widths, heights)
        tiny = (max_dim <= 16).sum()
        small = ((max_dim > 16) & (max_dim <= 32)).sum()
        medium = ((max_dim > 32) & (max_dim <= 48)).sum()
        large = (max_dim > 48).sum()

        size_labels = ['Tiny\n(<16px)', 'Small\n(16-32px)', 'Medium\n(32-48px)', 'Large\n(≥48px)']
        size_values = [tiny, small, medium, large]
        colors = ['#ff9999', '#ffcc99', '#99ccff', '#99ff99']

        # Filter out zero values
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
                   autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
        ax.set_title('Classification by\nMaximum Dimension', fontsize=11, fontweight='bold')

        return fig, ax

    @staticmethod
    def create_cumulative_plot(widths, fig=None, ax=None):
        """Create cumulative distribution plot"""
        if fig is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.clear()

        sorted_widths = np.sort(widths)
        cumulative = np.arange(1, len(sorted_widths) + 1) / len(sorted_widths) * 100

        ax.plot(sorted_widths, cumulative, linewidth=2, color='blue')
        ax.axvline(32, color='red', linestyle='--', label='YOLO small object\nthreshold (32px)')
        ax.set_xlabel('Width (pixels)', fontsize=11)
        ax.set_ylabel('Cumulative Percentage (%)', fontsize=11)
        ax.set_title('Defect Width\nCumulative Distribution', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

        return fig, ax

    @staticmethod
    def create_combined_figure(widths, heights, areas):
        """Create combined chart"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PCB Defect Size Distribution Analysis', fontsize=16, fontweight='bold')

        ChartGenerator.create_width_histogram(widths, fig, axes[0, 0])
        ChartGenerator.create_height_histogram(heights, fig, axes[0, 1])
        ChartGenerator.create_area_histogram(areas, fig, axes[0, 2])
        ChartGenerator.create_scatter_plot(widths, heights, fig, axes[1, 0])
        ChartGenerator.create_pie_chart(widths, heights, fig, axes[1, 1])
        ChartGenerator.create_cumulative_plot(widths, fig, axes[1, 2])

        plt.tight_layout()
        return fig


class PCBDefectAnalyzer(QMainWindow):
    """PCB Defect Size Analysis Tool Main Window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCB Defect Size Analysis Tool v1.0")
        self.setMinimumSize(1200, 800)

        # Data storage
        self.widths = None
        self.heights = None
        self.areas = None
        self.csv_path = None
        self.images_folder = None
        self.labels_folder = None

        self.init_ui()

    def init_ui(self):
        """Initialize user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left control panel
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)

        # Right display area
        right_panel = self.create_display_panel()
        splitter.addWidget(right_panel)

        # Set split ratio
        splitter.setSizes([350, 850])

    def create_control_panel(self):
        """Create control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        # ===== CSV File Selection Group =====
        csv_group = QGroupBox("📊 CSV File Analysis")
        csv_layout = QVBoxLayout(csv_group)

        self.csv_label = QLabel("No CSV file selected")
        self.csv_label.setWordWrap(True)
        csv_layout.addWidget(self.csv_label)

        btn_select_csv = QPushButton("📂 Select CSV File")
        btn_select_csv.clicked.connect(self.select_csv_file)
        btn_select_csv.setMinimumHeight(35)
        csv_layout.addWidget(btn_select_csv)

        layout.addWidget(csv_group)

        # ===== Chart Generation Group =====
        chart_group = QGroupBox("📈 Chart Generation")
        chart_layout = QGridLayout(chart_group)

        chart_buttons = [
            ("📊 Width Histogram", self.show_width_histogram),
            ("📊 Height Histogram", self.show_height_histogram),
            ("📊 Area Histogram", self.show_area_histogram),
            ("⚪ Width-Height Scatter", self.show_scatter_plot),
            ("🥧 Size Classification Pie", self.show_pie_chart),
            ("📈 Cumulative Distribution", self.show_cumulative_plot),
        ]

        for i, (text, callback) in enumerate(chart_buttons):
            btn = QPushButton(text)
            btn.clicked.connect(callback)
            btn.setMinimumHeight(35)
            chart_layout.addWidget(btn, i // 2, i % 2)

        # Combined chart button
        btn_combined = QPushButton("🖼️ Generate Combined Chart (6-in-1)")
        btn_combined.clicked.connect(self.show_combined_chart)
        btn_combined.setMinimumHeight(40)
        btn_combined.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        chart_layout.addWidget(btn_combined, 3, 0, 1, 2)

        # Save button
        btn_save = QPushButton("💾 Save Current Chart")
        btn_save.clicked.connect(self.save_current_chart)
        btn_save.setMinimumHeight(35)
        chart_layout.addWidget(btn_save, 4, 0, 1, 2)

        layout.addWidget(chart_group)

        # ===== YOLO Label to CSV Group =====
        yolo_group = QGroupBox("🏷️ YOLO Labels to CSV")
        yolo_layout = QVBoxLayout(yolo_group)

        # Images folder
        self.images_label = QLabel("No Images folder selected")
        self.images_label.setWordWrap(True)
        yolo_layout.addWidget(self.images_label)

        btn_select_images = QPushButton("📁 Select Images Folder")
        btn_select_images.clicked.connect(self.select_images_folder)
        btn_select_images.setMinimumHeight(35)
        yolo_layout.addWidget(btn_select_images)

        # Labels folder
        self.labels_label = QLabel("No Labels folder selected")
        self.labels_label.setWordWrap(True)
        yolo_layout.addWidget(self.labels_label)

        btn_select_labels = QPushButton("📁 Select Labels Folder")
        btn_select_labels.clicked.connect(self.select_labels_folder)
        btn_select_labels.setMinimumHeight(35)
        yolo_layout.addWidget(btn_select_labels)

        # Generate CSV button
        btn_generate_csv = QPushButton("⚙️ Generate CSV File")
        btn_generate_csv.clicked.connect(self.generate_csv_from_yolo)
        btn_generate_csv.setMinimumHeight(40)
        btn_generate_csv.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        yolo_layout.addWidget(btn_generate_csv)

        layout.addWidget(yolo_group)

        # ===== Progress Bar =====
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # ===== Log Area =====
        log_group = QGroupBox("📝 Log Output")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setMinimumHeight(150)
        log_layout.addWidget(self.log_text)

        btn_clear_log = QPushButton("🗑️ Clear Log")
        btn_clear_log.clicked.connect(lambda: self.log_text.clear())
        log_layout.addWidget(btn_clear_log)

        layout.addWidget(log_group)

        # Flexible space
        layout.addStretch()

        return panel

    def create_display_panel(self):
        """Create display panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Title
        title_label = QLabel("Chart Preview Area")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title_label)

        # Chart canvas
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.canvas)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        # Statistics info label
        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        layout.addWidget(self.stats_label)

        return panel

    def log(self, message):
        """Add log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # Scroll to bottom
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def select_csv_file(self):
        """Select CSV file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)"
        )

        if file_path:
            self.csv_path = file_path
            self.csv_label.setText(f"Selected: {Path(file_path).name}")
            self.log(f"Selected CSV file: {file_path}")
            self.load_csv_data()

    def load_csv_data(self):
        """Load CSV data"""
        if not self.csv_path:
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress

        self.worker = WorkerThread('load_csv', csv_path=self.csv_path)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_csv_loaded)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def on_csv_loaded(self, result):
        """CSV loading complete callback"""
        self.progress_bar.setVisible(False)

        if result.get('success'):
            self.widths = result['widths']
            self.heights = result['heights']
            self.areas = result['areas']

            # Update statistics info
            stats_text = f"""
            <b>Data Statistics Summary:</b><br>
            • Total Defects: {len(self.widths)}<br>
            • Width Range: {self.widths.min():.1f} - {self.widths.max():.1f} pixels (Mean: {self.widths.mean():.1f})<br>
            • Height Range: {self.heights.min():.1f} - {self.heights.max():.1f} pixels (Mean: {self.heights.mean():.1f})<br>
            • Area Range: {self.areas.min():.1f} - {self.areas.max():.1f} pixels² (Mean: {self.areas.mean():.1f})
            """
            self.stats_label.setText(stats_text)
            self.log("✓ CSV data loaded successfully, ready to generate charts!")

            # Auto-display combined chart
            self.show_combined_chart()

    def on_error(self, error_msg):
        """Error callback"""
        self.progress_bar.setVisible(False)
        self.log(f"❌ Error: {error_msg}")
        QMessageBox.critical(self, "Error", error_msg)

    def check_data_loaded(self):
        """Check if data is loaded"""
        if self.widths is None or self.heights is None or self.areas is None:
            QMessageBox.warning(self, "Notice", "Please select and load a CSV file first!")
            return False
        return True

    def show_width_histogram(self):
        """Show width histogram"""
        if not self.check_data_loaded():
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ChartGenerator.create_width_histogram(self.widths, self.figure, ax)
        self.canvas.draw()
        self.log("Displaying width distribution histogram")

    def show_height_histogram(self):
        """Show height histogram"""
        if not self.check_data_loaded():
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ChartGenerator.create_height_histogram(self.heights, self.figure, ax)
        self.canvas.draw()
        self.log("Displaying height distribution histogram")

    def show_area_histogram(self):
        """Show area histogram"""
        if not self.check_data_loaded():
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ChartGenerator.create_area_histogram(self.areas, self.figure, ax)
        self.canvas.draw()
        self.log("Displaying area distribution histogram")

    def show_scatter_plot(self):
        """Show scatter plot"""
        if not self.check_data_loaded():
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ChartGenerator.create_scatter_plot(self.widths, self.heights, self.figure, ax)
        self.canvas.draw()
        self.log("Displaying width-height scatter plot")

    def show_pie_chart(self):
        """Show pie chart"""
        if not self.check_data_loaded():
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ChartGenerator.create_pie_chart(self.widths, self.heights, self.figure, ax)
        self.canvas.draw()
        self.log("Displaying size classification pie chart")

    def show_cumulative_plot(self):
        """Show cumulative distribution plot"""
        if not self.check_data_loaded():
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ChartGenerator.create_cumulative_plot(self.widths, self.figure, ax)
        self.canvas.draw()
        self.log("Displaying cumulative distribution plot")

    def show_combined_chart(self):
        """Show combined chart"""
        if not self.check_data_loaded():
            return

        self.figure.clear()

        # Create 2x3 subplots
        axes = self.figure.subplots(2, 3)
        self.figure.suptitle('PCB Defect Size Distribution Analysis', fontsize=14, fontweight='bold')

        ChartGenerator.create_width_histogram(self.widths, self.figure, axes[0, 0])
        ChartGenerator.create_height_histogram(self.heights, self.figure, axes[0, 1])
        ChartGenerator.create_area_histogram(self.areas, self.figure, axes[0, 2])
        ChartGenerator.create_scatter_plot(self.widths, self.heights, self.figure, axes[1, 0])
        ChartGenerator.create_pie_chart(self.widths, self.heights, self.figure, axes[1, 1])
        ChartGenerator.create_cumulative_plot(self.widths, self.figure, axes[1, 2])

        self.figure.tight_layout()
        self.canvas.draw()
        self.log("Displaying combined chart (6-in-1)")

    def save_current_chart(self):
        """Save current chart"""
        if self.widths is None:
            QMessageBox.warning(self, "Notice", "No chart to save!")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Chart", "defect_analysis.png",
            "PNG Image (*.png);;JPEG Image (*.jpg);;PDF Document (*.pdf);;All Files (*)"
        )

        if file_path:
            self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
            self.log(f"✓ Chart saved: {file_path}")
            QMessageBox.information(self, "Success", f"Chart saved to:\n{file_path}")

    def select_images_folder(self):
        """Select Images folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Images Folder")
        if folder:
            self.images_folder = folder
            self.images_label.setText(f"Selected: {Path(folder).name}")
            self.log(f"Selected Images folder: {folder}")

    def select_labels_folder(self):
        """Select Labels folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Labels Folder")
        if folder:
            self.labels_folder = folder
            self.labels_label.setText(f"Selected: {Path(folder).name}")
            self.log(f"Selected Labels folder: {folder}")

    def generate_csv_from_yolo(self):
        """Generate CSV from YOLO labels"""
        if not self.images_folder:
            QMessageBox.warning(self, "Notice", "Please select Images folder first!")
            return

        if not self.labels_folder:
            QMessageBox.warning(self, "Notice", "Please select Labels folder first!")
            return

        # Select save location
        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV File", "defect_sizes.csv", "CSV Files (*.csv)"
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
        """CSV generation complete callback"""
        self.progress_bar.setVisible(False)

        if result.get('success'):
            self.log(f"✓ Successfully generated CSV file with {result['count']} defect records")

            reply = QMessageBox.question(
                self, "Success",
                f"CSV file has been generated!\nContains {result['count']} defect records\n\nLoad this file for analysis now?",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.csv_path = result['path']
                self.csv_label.setText(f"Selected: {Path(result['path']).name}")
                self.load_csv_data()


def main():
    """Main function"""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create main window
    window = PCBDefectAnalyzer()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()