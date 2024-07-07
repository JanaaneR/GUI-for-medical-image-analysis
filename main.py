import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QWidget, QMessageBox, QGridLayout, QScrollArea, QSizePolicy, QHBoxLayout
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import QtCore
import cv2
import numpy as np

class ZoomableImageLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pixmap = None
        self.zoom_factor = 1.0
        self.setScaledContents(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def setPixmap(self, pixmap):
        self.pixmap = pixmap
        super().setPixmap(self.getZoomedPixmap())

    def zoomIn(self):
        self.zoom_factor *= 1.2
        self.setPixmap(self.getZoomedPixmap())

    def zoomOut(self):
        self.zoom_factor /= 1.2
        self.setPixmap(self.getZoomedPixmap())

    def getZoomedPixmap(self):
        if self.pixmap:
            width = int(self.pixmap.width() * self.zoom_factor)
            height = int(self.pixmap.height() * self.zoom_factor)
            return self.pixmap.scaled(width, height, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        return self.pixmap

class WoundImageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Wound Image Analyzer')
        self.setGeometry(100, 100, 1000, 800)

        # Initialize widgets and layout
        self.image_label = ZoomableImageLabel(self)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.info_label = QLabel(self)
        self.info_label.setAlignment(QtCore.Qt.AlignCenter)
        self.info_label.setStyleSheet("font-weight: bold;")
        self.analyze_button = QPushButton('Analyze Wound', self)
        self.analyze_button.clicked.connect(self.analyze_wound)
        self.load_button = QPushButton('Load Image', self)
        self.load_button.clicked.connect(self.load_image)
        self.reset_button = QPushButton('Reset', self)
        self.reset_button.clicked.connect(self.reset_app)
        self.reset_button.setEnabled(False)

        self.grid_layout = QGridLayout()
        self.grid_layout.addWidget(self.load_button, 0, 0)
        self.grid_layout.addWidget(self.analyze_button, 0, 1)
        self.grid_layout.addWidget(self.image_label, 1, 0, 1, 2)
        self.grid_layout.addWidget(self.info_label, 2, 0, 1, 2)
        self.grid_layout.addWidget(self.reset_button, 3, 0, 1, 2)

        self.segmented_image_labels = []
        self.segmented_image_label_texts = ["Color Analysis Result", "Texture Analysis Result", "Edge Detection Result"]
        for i, text in enumerate(self.segmented_image_label_texts):
            label = QLabel(text, self)
            label.setAlignment(QtCore.Qt.AlignCenter)
            self.segmented_image_labels.append(label)
            self.grid_layout.addWidget(label, 4 + i, 0)

        self.segmented_images = []
        for _ in range(len(self.segmented_image_label_texts)):
            segmented_image = ZoomableImageLabel(self)
            segmented_image.setAlignment(QtCore.Qt.AlignCenter)
            self.segmented_images.append(segmented_image)

        central_widget = QWidget()
        central_widget.setLayout(self.grid_layout)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(central_widget)
        self.setCentralWidget(scroll_area)

        self.image = None
        self.zoom_buttons = []

    def load_image(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, 'Open Wound Image', '', 'Image files (*.png *.jpg *.jpeg *.bmp)')
        if file_path:
            self.image = cv2.imread(file_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            max_width = self.image_label.width()
            max_height = self.image_label.height()
            self.image = self.resize_image(self.image, max_width, max_height)
            height, width, channel = self.image.shape
            bytes_per_line = channel * width
            q_image = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap)
            self.info_label.clear()
            self.enable_reset_button()

    def resize_image(self, image, max_width, max_height):
        height, width, _ = image.shape
        if width > max_width or height > max_height:
            if width / max_width > height / max_height:
                scale_factor = max_width / width
            else:
                scale_factor = max_height / height
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            resized_image = cv2.resize(image, (new_width, new_height))
            return resized_image
        else:
            return image

    def analyze_wound(self):
        if self.image is None:
            QMessageBox.warning(self, "Warning", "Please load an image first.")
            return

        lower_bound = np.array([0, 50, 50], dtype=np.uint8)
        upper_bound = np.array([10, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV), lower_bound, upper_bound)

        # Display the mask image
        mask_pixmap = self.create_pixmap(mask)
        self.add_mask_image(mask_pixmap)

        # Perform color analysis, texture analysis, edge detection, and generate a report.
        color_analysis_image = cv2.bitwise_and(self.image, self.image, mask=mask)
        texture_analysis_image = cv2.bitwise_and(self.image, self.image, mask=mask)
        edge_detection_image = cv2.bitwise_and(self.image, self.image, mask=mask)

        color_analysis_result = self.analyze_color(color_analysis_image, mask)
        texture_analysis_result = self.analyze_texture(texture_analysis_image, mask)
        edge_detection_result = self.detect_edges(edge_detection_image, mask)

        # Display segmented images
        color_analysis_pixmap = self.create_pixmap(color_analysis_image)
        texture_analysis_pixmap = self.create_pixmap(texture_analysis_image)
        edge_detection_pixmap = self.create_pixmap(edge_detection_image)

        self.segmented_images[0].setPixmap(color_analysis_pixmap)
        self.segmented_images[1].setPixmap(texture_analysis_pixmap)
        self.segmented_images[2].setPixmap(edge_detection_pixmap)

        report = self.generate_report(mask, color_analysis_result, texture_analysis_result, edge_detection_result)
        self.info_label.setText(report)
        self.add_zoom_buttons()

    def create_pixmap(self, image):
        if len(image.shape) == 3:  # RGB image
            height, width, channel = image.shape
            bytes_per_line = channel * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        else:  # Grayscale image
            height, width = image.shape
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        return pixmap

    def add_mask_image(self, mask_pixmap):
        mask_label = QLabel(self)
        mask_label.setPixmap(mask_pixmap)
        mask_label.setAlignment(QtCore.Qt.AlignCenter)
        self.grid_layout.addWidget(mask_label, 6, 0, 1, 2)

    def add_zoom_buttons(self):
        # Clear existing zoom buttons
        self.clear_zoom_buttons()
        for segmented_image, label in zip(self.segmented_images, self.segmented_image_labels):
            zoom_in_button = QPushButton('+', self)
            zoom_out_button = QPushButton('-', self)
            zoom_in_button.clicked.connect(segmented_image.zoomIn)
            zoom_out_button.clicked.connect(segmented_image.zoomOut)
            layout = QHBoxLayout()
            layout.addWidget(zoom_in_button)
            layout.addWidget(zoom_out_button)
            layout.setAlignment(QtCore.Qt.AlignCenter)
            widget = QWidget()
            widget.setLayout(layout)
            self.grid_layout.addWidget(widget, self.grid_layout.rowCount(), 1)
            self.grid_layout.addWidget(label, self.grid_layout.rowCount(), 0)
            self.grid_layout.addWidget(segmented_image, self.grid_layout.rowCount() - 1, 1)
            self.zoom_buttons.extend([zoom_in_button, zoom_out_button])

    def clear_zoom_buttons(self):
        for button in self.zoom_buttons:
            button.deleteLater()
        self.zoom_buttons.clear()

    def enable_reset_button(self):
        self.reset_button.setEnabled(True)

    def reset_app(self):
        self.image_label.clear()
        self.info_label.clear()
        for segmented_image in self.segmented_images:
            segmented_image.clear()
        self.disable_reset_button()
        self.clear_zoom_buttons()

    def disable_reset_button(self):
        self.reset_button.setEnabled(False)

    def analyze_color(self, image, mask):
        mean_color = cv2.mean(image, mask=mask)[:3]
        if mean_color[0] > 150:
            return "Normal"
        else:
            return "Infection Detected"
    def analyze_texture(self, image, mask):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude_squared = sobel_x ** 2 + sobel_y ** 2
        magnitude_squared[magnitude_squared < 0] = 0
        valid_magnitude = magnitude_squared[mask == 255]
        texture_score = np.sqrt(np.mean(valid_magnitude)) if valid_magnitude.size > 0 else 0
        if texture_score < 50:
            return "Smooth Texture"
        else:
            return "Irregular Texture (Possible Necrosis)"

    def detect_edges(self, image, mask):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray_image, 100, 200)
        edge_detected = cv2.bitwise_and(edges, edges, mask=mask)
        if np.sum(edge_detected) > 1000:
            return "Edges Detected"
        else:
            return "No Clear Edges"

    def generate_report(self, mask, color_analysis_result, texture_analysis_result, edge_detection_result):
        report_text = f"Wound Analysis Report:\n"
        report_text += f"Color Analysis Result: {color_analysis_result}\n"
        report_text += f"Texture Analysis Result: {texture_analysis_result}\n"
        report_text += f"Edge Detection Result: {edge_detection_result}\n"
        return report_text

if name == 'main':
    app = QApplication(sys.argv)
    window = WoundImageApp()
    window.show()
    sys.exit(app.exec_())