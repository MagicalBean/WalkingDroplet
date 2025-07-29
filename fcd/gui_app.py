import sys
import os
from traceback import format_list

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from fcd import run_fcd
from select_roi import select_region

class FCDWindow(QMainWindow):    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fast Checkerboard Demodulation")
        self.setGeometry(200, 200, 600, 300)
        
        self.ref_image_path = ""
        self.def_folder_path = ""
        self.render_mode = 2 # default
        self.crop_region = None

        # self.setStyleSheet("""
        #     QWidget {
        #         font-size: 11pt;
        #     }
        #     QPushButton {
        #         padding: 6px;    
        #     }
        #     QLabel {
        #         padding: 2px;            
        #     }
        # """)
        
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
        crop_button = QPushButton("Select New ROI")
        crop_button.clicked.connect(self.select_crop_region)
        
        load_crop_button = QPushButton("Load Preview ROI")
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

        # run nutton
        run_button = QPushButton("Run")
        run_button.clicked.connect(self.run_process)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_output)

        # layout setup
        form_layout.addWidget(self.ref_label, 0, 0)
        form_layout.addWidget(ref_button, 0, 2)
        
        form_layout.addWidget(self.def_label, 1, 0)
        form_layout.addWidget(def_button, 1, 2)
        
        form_layout.addWidget(QLabel("Crop Region:"), 2, 0)
        form_layout.addWidget(crop_button, 2, 1)
        form_layout.addWidget(load_crop_button, 2, 2)

        form_layout.addWidget(render_label, 3, 0)
        form_layout.addLayout(render_buttons_layout, 3, 1, 1, 2)
        
        layout.addLayout(form_layout)
        layout.addWidget(run_button)
        layout.addWidget(save_button)

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
        self.crop_region = select_region(self.def_folder_path, user_override=True)
        
    def load_previous_crop(self):
        self.crop_region = select_region(self.def_folder_path, preview=False)

    def save_output(self):
        save_path, _ = QFileDialog(self, "Save Rendered Output", "output.png", "Videos (*.mp4)")
        if save_path:
            QMessageBox.information(self, "Saved", "Saved")
            # save

    def run_process(self):
        if not self.ref_image_path or not self.def_folder_path:
            QMessageBox.warning(self, "Missing Info", "Please select a reference image and a definition folder.")
            return
        
        # Place processing call here
        ani = run_fcd(self.ref_image_path, self.def_folder_path, self.crop_region, self.render_mode)
        
        QMessageBox.information(self, "Run Complete", f"FCD run complete with render mode: {self.render_mode.upper()}")
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = FCDWindow()
    gui.show()
    sys.exit(app.exec_())
