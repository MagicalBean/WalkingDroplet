import sys
import os
from pathlib import Path
import json

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import matplotlib
matplotlib.use("Qt5Agg")    

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from fcd import run_fcd
from select_roi import roi_filename_for, get_first_image
from renderer import render

class FCDWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    done = pyqtSignal(object) # emit animation object on finish

    def __init__(self, ref_img, folder_path, crop, render_mode):
        super().__init__()
        self.ref_img = ref_img
        self.folder_path = folder_path
        self.crop = crop
        self.render_mode = render_mode

    def run(self):
        obj = run_fcd(self.ref_img, self.folder_path, self.crop, self.render_mode, progress_cb=self.progress.emit, status_cb=self.status.emit)
        self.done.emit(obj)

class FCDWindow(QMainWindow):    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fast Checkerboard Demodulation")
        self.setGeometry(200, 200, 600, 300)
        
        self.ref_image_path = ""
        self.def_folder_path = ""
        self.render_mode = 2 # default
        self.crop_region = None

        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI';
                font-size: 11pt;
                background-color: #f8f9fa;
                color: #222
            }
            QPushButton {
                padding: 6px 12px;
                border: none;
                background-color: #e0e0e0;
                color: #333;
                border-radius: 4px;    
            }
            QPushButton:hover {
                background-color: #d6d6d6;
            }
            QLabel {
                padding: 2px;            
            }
            QProgressBar {
                height: 8px
                border-radius: 4px;
                text-align: center;
                font-size: 9px;    
            }
            QProgressBar::chunk {
                background-color: #28a745;
                border-radius: 4px;
            }
        """)
        
        self.init_ui()
            
    def init_ui(self):
        main_widget = QWidget()
        layout = QVBoxLayout()
        
        form_layout = QGridLayout()
        form_layout.setSpacing(10)
        
        # reference image selection
        self.ref_label = QLabel("Reference Image: None")
        ref_button = QPushButton("Browse")
        ref_button.clicked.connect(self.select_reference_image)
        
        # definition folder selection
        self.def_label = QLabel("Definition Folder: None")
        def_button = QPushButton("Browse")
        def_button.clicked.connect(self.select_definition_folder)
        
        # crop buttons
        self.crop_label = QLabel("Crop Region:")
        crop_button = QPushButton("Select New ROI")
        crop_button.clicked.connect(self.select_crop_region)
        
        load_crop_button = QPushButton("Load Previous ROI")
        load_crop_button.clicked.connect(self.load_previous_crop)
        
        # render mode
        render_label = QLabel("Render Mode:")
        self.radio_group = QButtonGroup()
        render_buttons_layout = QHBoxLayout()
        for mode in ["1d", "2d", "3d"]:
            btn = QRadioButton(mode.upper())
            if mode == "2d":
                btn.setChecked(True)
            btn.toggled.connect(lambda checked, m=mode: self.set_render_mode(int(m[0])))
            self.radio_group.addButton(btn)
            render_buttons_layout.addWidget(btn)

        # run button
        run_button = QPushButton("Run")
        run_button.clicked.connect(self.run_process)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_output)

        # progress bar
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(False)

        # status label
        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("font-size: 9px; color: #555;")

        # layout setup
        form_layout.addWidget(self.ref_label, 0, 0)
        form_layout.addWidget(ref_button, 0, 2)
        
        form_layout.addWidget(self.def_label, 1, 0)
        form_layout.addWidget(def_button, 1, 2)
        
        form_layout.addWidget(self.crop_label, 2, 0)
        form_layout.addWidget(crop_button, 2, 1)
        form_layout.addWidget(load_crop_button, 2, 2)

        form_layout.addWidget(render_label, 3, 0)
        form_layout.addLayout(render_buttons_layout, 3, 1, 1, 2)
        
        layout.addLayout(form_layout)
        layout.addWidget(run_button)
        layout.addWidget(save_button)
        layout.addWidget(self.progress)
        layout.addWidget(self.status_label)

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
        
    def select_reference_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Reference Image", "", "Images (*.png *.jpg *.tif)")
        if file_path:
            self.ref_image_path = file_path
            self.ref_label.setText(f"Reference Image: {os.path.basename(file_path)}")

    def select_definition_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder with Definition Images")
        if folder_path:
            self.def_folder_path = folder_path
            self.def_label.setText(f"Definition Folder: {os.path.basename(folder_path)}")

    def set_render_mode(self, mode):
        self.render_mode = mode
        
    def select_crop_region(self):
        if not self.def_folder_path:
            QMessageBox.warning(self, "Missing Image", "Please select a definition image folder first.")
            return
        
        roi_path = roi_filename_for(Path(self.def_folder_path))
        img = get_first_image(self.def_folder_path)
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')

        rect_selector = RectangleSelector(ax, useblit=True,
                                        button=[1], # left mouse button,
                                        minspanx=5, minspany=5, spancoords='pixels', interactive=True, state_modifier_keys=dict(square=''))
        
        # close window and submit if enter key is pressed
        def on_press(event):
            if event.key == 'enter':
                roi = tuple(map(round, rect_selector.extents))
                with open(roi_path, "w") as f:
                    json.dump(roi, f)

                self.crop_region = roi
                self.crop_label.setText(f"Crop Region: {roi}")

                plt.close()

        # require region to be square
        rect_selector.add_state('square')
        fig.canvas.mpl_connect('key_press_event', on_press)
        ax.set_title("Drag to select a region for analysis (Press 'Enter' to submit)")
        plt.show(block=False)
        
    def load_previous_crop(self):
        roi_path = roi_filename_for(Path(self.def_folder_path))
        if roi_path and os.path.exists(roi_path):
            with open(roi_path) as f:
                roi = json.load(f)
                self.crop_region = roi
                self.crop_label.setText(f"Crop Region: {roi}")
        else: 
            QMessageBox.warning(self, "No ROI Found", "No previous crop region found for this folder.")

    def save_output(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Rendered Output", "output.mp4", "Videos (*.mp4)")
        if save_path:
            self.anim.save(save_path, writer='ffmpeg', fps=10)
            # QMessageBox.information(self, "Saved", "Saved")
            # save

    def run_process(self):
        if not self.ref_image_path or not self.def_folder_path:
            QMessageBox.warning(self, "Missing Info", "Please select a reference image and a definition folder.")
            return
        
        self.worker = FCDWorker(
            self.ref_image_path,
            self.def_folder_path,
            self.crop_region,
            self.render_mode
        )

        self.worker.progress.connect(self.progress.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.done.connect(self.on_processing_done)
        self.worker.start()

    def on_processing_done(self, obj):
        self.status_label.setText("Idle")
        self.progress.setValue(0)

        if obj:
            height_maps, drop_diameter, scale = obj
            self.anim = render(height_maps, drop_diameter, scale, self.render_mode) # keep internal reference
            plt.show(block=False)
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = FCDWindow()
    gui.show()
    sys.exit(app.exec_())
