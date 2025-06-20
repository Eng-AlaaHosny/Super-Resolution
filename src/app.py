#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import time
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QWidget, QFileDialog, QMessageBox,
    QHBoxLayout, QProgressBar, QComboBox, QCheckBox, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import torch
from model import load_sr_model, predict_sr

class EnhancementThread(QThread):
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, model, image):
        super().__init__()
        self.model = model
        self.image = image

    def run(self):
        try:
            enhanced = predict_sr(self.model, self.image)
            self.finished.emit(enhanced)
        except Exception as e:
            self.error.emit(str(e))

class SuperResolutionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Super Resolution")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(800, 600)
        
        # Configuration
        # Fixed model directory to be relative to this file
        self.model_dir = Path(__file__).parent / "weights" / "pretrained"
        self.supported_scales = [2, 3, 4, 8]
        self.max_image_size = 2048  # Prevent OOM errors
        self.current_scale = 4  # Default scale
        
        # State
        self.model = None
        self.original_image = None
        self.enhanced_image = None
        self.enhancement_thread = None
        self._compare_mode = False
        
        self._init_ui()
        self._setup_connections()
        self._load_available_models()

    def _init_ui(self):
        # Main widget and layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout()
        central.setLayout(main_layout)
        
        # Model selection panel
        model_panel = QWidget()
        model_layout = QHBoxLayout(model_panel)
        
        # Model selection components
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        
        self.scale_combo = QComboBox()
        for scale in self.supported_scales:
            self.scale_combo.addItem(f"x{scale}", scale)
        self.scale_combo.setCurrentIndex(2)  # Default to x4
        
        self.light_blocks_checkbox = QCheckBox("Use Light Blocks")
        self.light_blocks_checkbox.setChecked(True)  # Default to checked
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.setStyleSheet("padding: 5px 15px;")
        
        model_layout.addWidget(QLabel("Model:"))
        model_layout.addWidget(self.model_combo, 1)
        model_layout.addWidget(QLabel("Scale:"))
        model_layout.addWidget(self.scale_combo)
        model_layout.addWidget(self.light_blocks_checkbox)
        model_layout.addWidget(self.load_model_btn)
        model_layout.addStretch()
        
        main_layout.addWidget(model_panel)

        # Image display area
        image_panel = QWidget()
        image_layout = QHBoxLayout(image_panel)
        image_layout.setContentsMargins(0, 10, 0, 10)

        # Original image panel
        original_panel = QVBoxLayout()
        original_panel.setSpacing(5)
        self.original_label = QLabel("Original Image")
        self.original_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_label.setMinimumSize(400, 400)
        self.original_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.original_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 2px solid #ccc;
                border-radius: 5px;
            }
        """)
        
        self.original_info = QLabel("No image loaded")
        self.original_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_info.setStyleSheet("font-weight: bold;")
        
        original_panel.addWidget(self.original_label)
        original_panel.addWidget(self.original_info)
        image_layout.addLayout(original_panel, 1)

        # Enhanced image panel
        enhanced_panel = QVBoxLayout()
        enhanced_panel.setSpacing(5)
        self.enhanced_label = QLabel("Enhanced Image")
        self.enhanced_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.enhanced_label.setMinimumSize(400, 400)
        self.enhanced_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.enhanced_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 2px solid #ccc;
                border-radius: 5px;
            }
        """)
        
        self.enhanced_info = QLabel("No enhancement performed")
        self.enhanced_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.enhanced_info.setStyleSheet("font-weight: bold;")
        
        enhanced_panel.addWidget(self.enhanced_label)
        enhanced_panel.addWidget(self.enhanced_info)
        image_layout.addLayout(enhanced_panel, 1)
        
        main_layout.addWidget(image_panel, 1)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 3px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
            }
        """)
        main_layout.addWidget(self.progress_bar)

        # Button panel
        button_panel = QWidget()
        button_layout = QHBoxLayout(button_panel)
        button_layout.setContentsMargins(0, 10, 0, 0)
        
        self.load_btn = QPushButton("Load Image")
        self.enhance_btn = QPushButton("Enhance Image")
        self.save_btn = QPushButton("Save Enhanced")
        self.compare_btn = QPushButton("Compare")
        
        for btn in [self.load_btn, self.enhance_btn, self.save_btn, self.compare_btn]:
            btn.setStyleSheet("""
                QPushButton {
                    padding: 8px 15px;
                    min-width: 100px;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                }
                QPushButton:disabled {
                    color: #888;
                }
            """)
            button_layout.addWidget(btn)
        
        button_layout.addStretch()
        main_layout.addWidget(button_panel)

        # Disable buttons initially
        self._update_button_states()

    def _setup_connections(self):
        self.load_btn.clicked.connect(self.load_image)
        self.enhance_btn.clicked.connect(self.start_enhancement)
        self.save_btn.clicked.connect(self.save_image)
        self.compare_btn.clicked.connect(self.toggle_compare)
        self.load_model_btn.clicked.connect(self.load_selected_model)
        self.scale_combo.currentIndexChanged.connect(self._update_scale)

    def _update_button_states(self):
        """Enable or disable buttons based on current state."""
        self.enhance_btn.setEnabled(self.model is not None and self.original_image is not None)
        self.save_btn.setEnabled(self.enhanced_image is not None)
        self.compare_btn.setEnabled(self.enhanced_image is not None)

    def _update_scale(self, index):
        """Handle scale selection change."""
        self.current_scale = self.scale_combo.currentData()

    def _load_available_models(self):
        """Scan weights directory for available models."""
        self.model_combo.clear()
        
        if not self.model_dir.exists():
            QMessageBox.warning(
                self, 
                "Warning", 
                f"Weights directory not found at: {self.model_dir}\n"
                f"Please ensure your weights are in the correct location."
            )
            return

        model_files = list(self.model_dir.glob("*.pth"))
        if not model_files:
            QMessageBox.warning(self, "Warning", f"No model files found in {self.model_dir}")
            return

        # Add any preferred names first
        preferred_order = ["RealESRGAN_x4plus.pth", "best.pth"]
        for preferred in preferred_order:
            for model in model_files:
                if model.name == preferred:
                    self.model_combo.addItem(model.name, model)
                    model_files.remove(model)
                    break

        # Add remaining models
        for model in model_files:
            self.model_combo.addItem(model.name, model)

    def load_selected_model(self):
        model_path = self.model_combo.currentData()
        use_light = self.light_blocks_checkbox.isChecked()

        if not model_path:
            QMessageBox.warning(self, "Warning", "Please select a model.")
            return

        # Show a loading state
        self.load_model_btn.setEnabled(False)
        self.load_model_btn.setText("Loading...")
        QApplication.processEvents()

        # Common model configurations to try
        configs_to_try = [
            # Config 1: Default with user's light_blocks choice
            {'use_light_blocks': use_light, 'num_feat': 64, 'num_block': 23, 'num_grow_ch': 32},
            # Config 2: Force light blocks
            {'use_light_blocks': True, 'num_feat': 64, 'num_block': 23, 'num_grow_ch': 32},
            # Config 3: Different architecture (Real-ESRGAN)
            {'use_light_blocks': use_light, 'num_feat': 64, 'num_block': 6, 'num_grow_ch': 32},
            # Config 4: Smaller model
            {'use_light_blocks': True, 'num_feat': 64, 'num_block': 6, 'num_grow_ch': 32},
            # Config 5: Let load_sr_model figure it out (with new implementation)
            {'use_light_blocks': use_light},
        ]

        last_error = None
        successful_config = None

        for i, config in enumerate(configs_to_try):
            try:
                self.model = load_sr_model(
                    scale=self.current_scale,
                    model_path=str(model_path),
                    progressive_scale=False,
                    **config
                )
                successful_config = config
                break
            except Exception as e:
                last_error = e
                continue

        self.load_model_btn.setEnabled(True)
        self.load_model_btn.setText("Load Model")

        if self.model is None:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load model with any configuration.\n"
                f"Last error: {last_error}\n\n"
                f"The model file might be incompatible or corrupted."
            )
            return

        # Success message
        config_str = ", ".join([f"{k}={v}" for k, v in successful_config.items()])
        QMessageBox.information(
            self,
            "Success",
            f"Successfully loaded model: {model_path.name}\n"
            f"Scale: x{self.current_scale}\n"
            f"Configuration: {config_str}"
        )
        self._update_button_states()

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)"
        )
        if not path:
            return

        try:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to read image. The file may be corrupted or unsupported.")

            h, w = img.shape[:2]
            if max(h, w) > self.max_image_size:
                raise ValueError(
                    f"Image dimensions ({w}×{h}) exceed maximum supported size "
                    f"of {self.max_image_size}×{self.max_image_size} pixels."
                )

            self.original_image = img
            self.enhanced_image = None
            self._compare_mode = False
            
            self._show_on_label(self.original_label, img)
            self._update_image_info(self.original_info, img, "Original")
            self.enhanced_label.clear()
            self.enhanced_label.setText("Enhanced Image")
            self.enhanced_info.setText("No enhancement performed")
            
            self._update_button_states()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Image",
                f"Failed to load image:\n{str(e)}"
            )

    def start_enhancement(self):
        if self.original_image is None or self.model is None:
            return

        # Disable interaction and show progress
        self.enhance_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.compare_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate

        self.enhanced_label.setText("Processing... Please wait")
        self.enhanced_info.setText("Enhancing image...")

        self.enhancement_thread = EnhancementThread(self.model, self.original_image)
        self.enhancement_thread.finished.connect(self.on_enhancement_complete)
        self.enhancement_thread.error.connect(self.on_enhancement_error)
        self.enhancement_thread.start()

    def on_enhancement_complete(self, enhanced_img):
        self.enhanced_image = enhanced_img
        
        self._show_on_label(self.enhanced_label, enhanced_img)
        self._update_image_info(self.enhanced_info, enhanced_img, "Enhanced")
        
        self.progress_bar.setVisible(False)
        self.load_btn.setEnabled(True)
        self._update_button_states()

    def on_enhancement_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.load_btn.setEnabled(True)
        self._update_button_states()
        
        QMessageBox.critical(
            self,
            "Enhancement Error",
            f"Failed to enhance image:\n{error_msg}"
        )

    def save_image(self):
        if self.enhanced_image is None:
            return

        default_name = f"enhanced_x{self.current_scale}_{int(time.time())}.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Enhanced Image", default_name,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tiff)"
        )
        if not path:
            return

        try:
            if not cv2.imwrite(path, self.enhanced_image):
                raise IOError("Failed to save image. Check write permissions.")
            
            QMessageBox.information(
                self,
                "Success",
                f"Enhanced image successfully saved to:\n{path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save image:\n{str(e)}"
            )

    def toggle_compare(self):
        if self.enhanced_image is None:
            return

        self._compare_mode = not self._compare_mode

        if self._compare_mode:
            if self.original_image is not None and self.enhanced_image is not None:
                combined = np.hstack((self.original_image, self.enhanced_image))
                self._show_on_label(self.original_label, combined)
                self.original_info.setText(
                    f"Left: Original | Right: Enhanced (x{self.current_scale})"
                )
                self.enhanced_label.clear()
                self.enhanced_label.setText("Comparison Mode")
                self.compare_btn.setText("Back to Normal")
            else:
                QMessageBox.warning(self, "Warning", "Both images must be available for comparison.")
                return
        else:
            if self.original_image is not None:
                self._show_on_label(self.original_label, self.original_image)
                self._update_image_info(self.original_info, self.original_image, "Original")
            if self.enhanced_image is not None:
                self._show_on_label(self.enhanced_label, self.enhanced_image)
                self._update_image_info(self.enhanced_info, self.enhanced_image, "Enhanced")
            self.compare_btn.setText("Compare")

    def _show_on_label(self, label: QLabel, img_bgr: np.ndarray):
        """Display a BGR NumPy image on the QLabel with proper scaling."""
        if img_bgr is None:
            label.clear()
            return
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        qimg = QImage(rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        scaled_pixmap = pixmap.scaled(
            label.width(), label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)

    def _update_image_info(self, label: QLabel, img: np.ndarray, img_type: str):
        """Update image information label with size and type."""
        if img is None:
            label.setText(f"{img_type}: No image")
            return
        h, w = img.shape[:2]
        channels = img.shape[2] if len(img.shape) > 2 else 1
        label.setText(f"{img_type}: {w}×{h} | Channels: {channels} | Scale: x{self.current_scale}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-display images to adapt to new label sizes
        if self.original_image is not None:
            if self._compare_mode and self.enhanced_image is not None:
                combined = np.hstack((self.original_image, self.enhanced_image))
                self._show_on_label(self.original_label, combined)
            else:
                self._show_on_label(self.original_label, self.original_image)
        if self.enhanced_image is not None and not self._compare_mode:
            self._show_on_label(self.enhanced_label, self.enhanced_image)

    def closeEvent(self, event):
        """Handle window close event by safely terminating any running thread."""
        if self.enhancement_thread and self.enhancement_thread.isRunning():
            self.enhancement_thread.terminate()
            self.enhancement_thread.wait(1000)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application font
    font = QFont()
    font.setFamily("Arial")
    font.setPointSize(10)
    app.setFont(font)
    
    win = SuperResolutionApp()
    win.show()
    sys.exit(app.exec_())