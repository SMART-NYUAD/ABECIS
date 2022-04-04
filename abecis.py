# importing libraries
from operator import truediv
import warnings
from PyQt6 import QtCore, QtWidgets
from tkinter import scrolledtext
from PyQt6.QtWidgets import *
from PyQt6 import QtCore, QtGui
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import gdown
import sys
import os
import re
import time
import sys
import threading
from datetime import datetime
# For Extracting Metadata
from PIL import Image
from PIL.ExifTags import TAGS
# For quantification
import numpy as np
# Generating Report
from showinfm import show_in_file_manager

# Detectron 2 libraries
import cv2
import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import numpy as np
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# Ignore depreciated warnings
warnings.filterwarnings('ignore')

# For Progressbar


class PercentageWorker(QtCore.QObject):
    started = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()
    percentageChanged = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._percentage = 0

    @property
    def percentage(self):
        return self._percentage

    @percentage.setter
    def percentage(self, value):
        if self._percentage == value:
            return
        self._percentage = value
        self.percentageChanged.emit(self.percentage)

    def start(self):
        self.started.emit()

    def finish(self):
        self.finished.emit()


class FakeWorker:
    def start(self):
        pass

    def finish(self):
        pass

    @property
    def percentage(self):
        return 0

    @percentage.setter
    def percentage(self, value):
        pass


def analyseImage(foo, dir_path, thresholdLower, thresholdUpper, ResultPossibleFolder, ResultConfidentFolder, baz="1", worker=None):
    imageFileList = []
    # get the path/directory
    folder_dir = dir_path
    for images in os.listdir(folder_dir):
        # check image types and process
        if (images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg")):
            imageFileList.append(images)
    # Report Image Number
    total_images = len(imageFileList)
    if(total_images > 0):
        print("Analyzing "+str(total_images)+" Images...")
        if worker is None:
            worker = FakeWorker()
        worker.start()
        current_id = 0
        while worker.percentage < 100:
            if(current_id < (total_images)):
                # Detect by multithreading
                image_name = imageFileList[current_id]
                worker.percentage += ((1/total_images)*100)
                percent = (int(worker.percentage)+1)
                if(percent > 100):
                    percent = 100
                print("["+str(percent)+"%] Analyzing " +
                      str(image_name))  # mask
                if(percent == 100):
                    print("\nCrack Analysis Completed. You may generate a report.")
                im = cv2.imread(os.path.join(
                    folder_dir, image_name))
                # Detection Classes
                classes = ['diagonal_crack', 'hairline_crack', 'horizontal_crack',
                           'vertical_crack']
                cfg = get_cfg()
                cfg.merge_from_file(model_zoo.get_config_file(
                    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
                cfg.DATASETS.TRAIN = ("category_train",)
                cfg.DATASETS.TEST = ()
                cfg.DATALOADER.NUM_WORKERS = 2
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
                cfg.SOLVER.IMS_PER_BATCH = 2
                cfg.SOLVER.BASE_LR = 0.00025
                cfg.SOLVER.MAX_ITER = 3000
                cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # Change this according to your classes
                os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
                cfg.MODEL.WEIGHTS = os.path.join(
                    cfg.OUTPUT_DIR, "model_final.pth")
                cfg.MODEL.DEVICE = 'cpu'
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
                    thresholdLower/100)
                cfg.DATASETS.TEST = ("crack_test", )
                predictor = DefaultPredictor(cfg)
                # Detect
                outputs = predictor(im)
                # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
                # print(outputs["instances"].pred_classes)
                # print(outputs["instances"].pred_boxes)
                # Labelling config
                MetadataCatalog.get("category_train").set(
                    thing_classes=classes)
                microcontroller_metadata = MetadataCatalog.get(
                    "category_train")
                # We can use `Visualizer` to draw the predictions on the image.
                v = Visualizer(
                    im[:, :, ::-1], metadata=microcontroller_metadata, scale=1.2)
                out = v.draw_instance_predictions(
                    outputs["instances"].to("cpu"))
                # Classify the images according to their scores
                if(len(outputs["instances"].scores.tolist()) > 0):
                    max_score = max(outputs["instances"].scores.tolist())*100
                else:
                    max_score = 0
                # Confidence type
                confidence_type = ""
                # Put in Confident Folder if greater than threshold upper
                if((max_score > thresholdUpper)):
                    image_output_path = os.path.join(
                        ResultConfidentFolder, image_name)
                    cv2.imwrite(image_output_path, out.get_image()
                                [:, :, ::-1])  # mask
                    confidence_type = "Confident Crack"
                elif((max_score >= thresholdLower) and (max_score <= thresholdUpper)):
                    image_output_path = os.path.join(
                        ResultPossibleFolder, image_name)
                    cv2.imwrite(image_output_path, out.get_image()
                                [:, :, ::-1])  # mask
                    confidence_type = "Possible Crack"
                else:
                    image_output_path = ""
                if(image_output_path != ""):
                    # Write the mask of the images to files
                    mask_array = outputs["instances"].pred_masks.to(
                        "cpu").numpy()
                    num_instances = mask_array.shape[0]
                    mask_array = np.moveaxis(mask_array, 0, -1)
                    mask_array_instance = []
                    output = np.zeros_like(im)  # black
                    for i in range(num_instances):
                        mask_array_instance.append(mask_array[:, :, i:(i+1)])
                        output = np.where(
                            mask_array_instance[i] == True, 255, output)
                    image_output_path_modified = image_output_path
                    image_output_path_modified = image_output_path_modified.replace(
                        '.png', '_mask.png')
                    image_output_path_modified = image_output_path_modified.replace(
                        '.jpg', '_mask.jpg')
                    image_output_path_modified = image_output_path_modified.replace(
                        '.jpeg', '_mask.jpeg')
                    cv2.imwrite(image_output_path_modified, output)
                    # Apply morphological thinning

                    # Read the image as a grayscale image
                    img = cv2.imread(image_output_path_modified, 0)

                    # Threshold the image
                    ret, img = cv2.threshold(img, 127, 255, 0)

                    # Step 1: Create an empty skeleton
                    size = np.size(img)
                    skel = np.zeros(img.shape, np.uint8)

                    # Get a Cross Shaped Kernel
                    element = cv2.getStructuringElement(
                        cv2.MORPH_CROSS, (3, 3))

                    # Repeat steps 2-4
                    while True:
                        # Step 2: Open the image
                        open_img = cv2.morphologyEx(
                            img, cv2.MORPH_OPEN, element)
                        # Step 3: Substract open from the original image
                        temp = cv2.subtract(img, open_img)
                        # Step 4: Erode the original image and refine the skeleton
                        eroded = cv2.erode(img, element)
                        skel = cv2.bitwise_or(skel, temp)
                        img = eroded.copy()
                        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
                        if cv2.countNonZero(img) == 0:
                            break

                    image_output_path_skel = image_output_path
                    image_output_path_skel = image_output_path_skel.replace(
                        '.png', '_length_estimation.png')
                    image_output_path_skel = image_output_path_skel.replace(
                        '.jpg', '_length_estimation.jpg')
                    image_output_path_skel = image_output_path_skel.replace(
                        '.jpeg', '_length_estimation.jpeg')
                    cv2.imwrite(image_output_path_skel, skel)

                    # Get Metadata, and analyze the % Coverage
                    # open the image
                    date_taken = ""
                    image = Image.open(os.path.join(
                        folder_dir, image_name))
                    # extracting the exif metadata
                    exifdata = image.getexif()
                    # looping through all the tags present in exifdata
                    for tagid in exifdata:
                        # getting the tag name instead of tag id
                        tagname = TAGS.get(tagid, tagid)
                        # passing the tagid to get its respective value
                        value = exifdata.get(tagid)
                        if(tagname == "DateTime"):
                            date_taken = str(value)
                    # Get the class names
                    predicted_classes = outputs["instances"].pred_classes.tolist(
                    )
                    predicted_classes_names = []
                    for x in predicted_classes:
                        predicted_classes_names.append(classes[x])
                    predicted_classes_names = list(
                        dict.fromkeys(predicted_classes_names))
                    predicted_classes_names.sort()
                    # Quantification of crack as %
                    number_of_white_pix = np.sum(output == 255)
                    number_of_black_pix = np.sum(output == 0)
                    crack_coverage = round(
                        ((number_of_white_pix/(number_of_white_pix+number_of_black_pix))*100), 2)

                    # Get Crack Length
                    crack_length = np.sum(skel == 255)
                    # Write results as text file
                    image_output_path_modified = image_output_path_modified.replace(
                        '.png', '_data.txt')
                    image_output_path_modified = image_output_path_modified.replace(
                        '.jpg', '_data.txt')
                    image_output_path_modified = image_output_path_modified.replace(
                        '.jpeg', '_data.txt')
                    f = open(image_output_path_modified, "w")
                    f.write(image_name+","+str(confidence_type)+","+str(date_taken)+",\"" +
                            str(predicted_classes_names).strip('[]').replace('_', ' ').title()+"\","+str(int(max_score))+","+str(crack_coverage)+","+str(crack_length))
                    f.close()

                current_id += 1
                worker.finish()
    else:
        print("\nNo image to analyze in "+dir_path)


class MainWindow(QWidget):
    def __init__(self):
        self.dir_path = ""
        self.console_out = ""
        self.thresholdUpper = 85
        self.thresholdLower = 60
        super().__init__()
        # Variables for image processing
        self.imageFileList = []
        self.total_images = 0
        # Verification Process
        self.verificationImageFileList = []
        self.total_verification_images = 0
        self.current_image_id = 0
        self.verify_mode = ""
        # Methods of Main Class
        self.initUI()

    def consoleUpdate(self, message):
        self.console_out = message+"\n"
        print(self.console_out)
        # Add new lines for GUI console output
        self.console_out = re.sub(
            "(.{50})", "\\1\n", self.console_out, 0, re.DOTALL)
        self.Console.setText("Application Status : "+self.console_out)

    def changeThresholdUpper(self):
        if(self.thresholdUpper < self.thresholdLower):
            QMessageBox.critical(self, "Threshold Value Error.",
                                 "The Upper Threshold must be greater than Lower threshold and vice versa.")
            self.thresholdUpper = self.thresholdLower + 10
            self.thresholdSliderUpper.setValue(self.thresholdUpper)
            self.labelThresholdUpperValue.setText(
                "("+str(self.thresholdUpper)+" %)")
            self.consoleUpdate("Confidence Score Upper Threshold set at " +
                               str(self.thresholdUpper)+" %")
        else:
            self.thresholdUpper = self.sender().value()
            self.labelThresholdUpperValue.setText(
                "("+str(self.thresholdUpper)+" %)")
            self.consoleUpdate("Confidence Score Upper Threshold set at " +
                               str(self.thresholdUpper)+" %")

    def changeThresholdLower(self):
        if(self.thresholdUpper < self.thresholdLower):
            QMessageBox.critical(self, "Threshold Value Error.",
                                 "The Upper Threshold must be greater than Lower threshold and vice versa.")
            self.thresholdLower = self.thresholdUpper - 10
            self.thresholdSliderLower.setValue(self.thresholdLower)
            self.labelThresholdLowerValue.setText(
                "("+str(self.thresholdLower)+" %)")
            self.consoleUpdate("Confidence Score Lower Threshold set at " +
                               str(self.thresholdLower)+" %")
        else:
            self.thresholdLower = self.sender().value()
            self.labelThresholdLowerValue.setText(
                "("+str(self.thresholdLower)+" %)")
            self.consoleUpdate("Confidence Score Lower Threshold set at " +
                               str(self.thresholdLower)+" %")

    def CheckVerifyFolder(self):
        # This function checks whether there exists the analyzed images folder for verification
        analyzedFolderPath = os.path.join(self.dir_path, "Crack_Analysis")
        ResultPossibleFolder = os.path.join(
            analyzedFolderPath, "Possible")
        ResultConfidentFolder = os.path.join(
            analyzedFolderPath, "Confident")
        # Check whether the specified path exists or not
        ResultFolderExist = os.path.exists(analyzedFolderPath)
        ResultPossibleFolderExists = os.path.exists(ResultPossibleFolder)
        ResultConfidentFolderExists = os.path.exists(ResultConfidentFolder)
        if(ResultFolderExist and ResultPossibleFolderExists and ResultConfidentFolderExists):
            return True
        else:
            return False

    # Image Analysis Functions
    def PrevImage(self):
        if(self.current_image_id > 0):
            self.current_image_id -= 1
        if(self.current_image_id > 0):
            self.btnPrevImage.show()
        else:
            self.btnPrevImage.hide()
        if(self.current_image_id < self.total_verification_images-1):
            self.btnNextImage.show()
        else:
            self.btnNextImage.hide()
        if(self.total_verification_images > 0):
            self.showImage(os.path.join(
                self.dir_path, "Crack_Analysis", self.verify_mode, self.verificationImageFileList[self.current_image_id]))
            self.consoleUpdate("Image "+str(self.current_image_id+1) +
                               " of "+str(self.total_verification_images))

    def NextImage(self):
        if(self.current_image_id < self.total_verification_images-1):
            self.current_image_id += 1
        if(self.current_image_id < self.total_verification_images-1):
            self.btnNextImage.show()
        else:
            self.btnNextImage.hide()
        if(self.current_image_id > 0):
            self.btnPrevImage.show()
        else:
            self.btnPrevImage.hide()
        if(self.total_verification_images > 0):
            self.showImage(os.path.join(
                self.dir_path, "Crack_Analysis", self.verify_mode, self.verificationImageFileList[self.current_image_id]))
            self.consoleUpdate("Image "+str(self.current_image_id+1) +
                               " of "+str(self.total_verification_images))

    def RemoveImage(self):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Confirm removal")
        dlg.setText("Are you sure you want to remove this result?")
        dlg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        dlg.setIcon(QMessageBox.Icon.Question)
        button = dlg.exec()
        if button == QMessageBox.StandardButton.Yes:
            if(len(self.verificationImageFileList) > 0):
                image_path = os.path.join(
                    self.dir_path, "Crack_Analysis", self.verify_mode, self.verificationImageFileList[self.current_image_id])
                try:
                    os.remove(image_path)
                except:
                    self.consoleUpdate("Error removing file "+image_path)
                image_output_path_modified = image_path
                image_output_path_modified = image_output_path_modified.replace(
                    '.png', '_mask.png')
                image_output_path_modified = image_output_path_modified.replace(
                    '.jpg', '_mask.jpg')
                image_output_path_modified = image_output_path_modified.replace(
                    '.jpeg', '_mask.jpeg')
                try:
                    os.remove(image_output_path_modified)
                except:
                    self.consoleUpdate("Error removing file" +
                                       image_output_path_modified)
                # Remove Text Data
                image_output_path_modified = image_path
                image_output_path_modified = image_output_path_modified.replace(
                    '.png', '_data.txt')
                image_output_path_modified = image_output_path_modified.replace(
                    '.jpg', '_data.txt')
                image_output_path_modified = image_output_path_modified.replace(
                    '.jpeg', '_data.txt')
                try:
                    os.remove(image_output_path_modified)
                except:
                    self.consoleUpdate("Error removing file" +
                                       image_output_path_modified)
                try:
                    self.verificationImageFileList.remove(
                        self.verificationImageFileList[self.current_image_id])
                    # print(self.verificationImageFileList)
                except:
                    self.consoleUpdate("Error processing removal.")
                self.total_verification_images = len(
                    self.verificationImageFileList)
                # Stop display if none left
                if(self.total_verification_images == 0):
                    self.btnPrevImage.hide()
                    self.btnRemoveImage.hide()
                    self.btnZoomImage.hide()
                    self.btnNextImage.hide()
                    self.imageHolder.hide()
                self.consoleUpdate("Image Removed from Results")
                if(self.current_image_id > self.total_verification_images-1):
                    self.current_image_id = 0
                else:
                    self.NextImage()
        else:
            return

    def ZoomImage(self):
        try:
            image = cv2.imread(os.path.join(
                self.dir_path, "Crack_Analysis", self.verify_mode, self.verificationImageFileList[self.current_image_id]))
            cv2.imshow(self.verify_mode +
                       ' Image. Press any key to close image.', image)
            # add wait key. window waits until user presses a key
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            QMessageBox.critical(self, "Error Opening Image.",
                                 "There was an error opening the image.")

    def showImage(self, image_path):
        try:
            # Modify Memory Allocation
            os.environ['QT_IMAGEIO_MAXALLOC'] = str(
                os.path.getsize(image_path))
            # Show Image
            self.pixmap = QPixmap(image_path)
            self.pixmap = self.pixmap.scaledToWidth(480)
            self.imageHolder.setPixmap(self.pixmap)
            self.imageHolder.show()
        except:
            QMessageBox.critical(self, "Error Opening Image.",
                                 "There was an error opening the image.")

    def VerifyPossible(self):
        self.current_image_id = 0
        self.consoleUpdate("Verifying Possible Results")
        self.verify_mode = "Possible"
        if(self.dir_path == ""):
            # Throw warning when no folder is selected
            QMessageBox.critical(self, "No folder selected.",
                                 "Please select a folder containing the images of analysis.")
        else:
            if(self.CheckVerifyFolder()):
                self.verificationImageFileList = []
                folder_dir = os.path.join(
                    self.dir_path, "Crack_Analysis", "Possible")
                for images in os.listdir(folder_dir):
                    # check image types and process
                    if (images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg")):
                        if("mask" not in images and "length_estimation" not in images):
                            self.verificationImageFileList.append(images)
                self.total_verification_images = len(
                    self.verificationImageFileList)
                self.consoleUpdate(
                    str(self.total_verification_images)+" images found for verification.")
                # Show Images
                if(self.total_verification_images > 0):
                    self.showImage(os.path.join(
                        self.dir_path, "Crack_Analysis", "Possible", self.verificationImageFileList[0]))
                    self.btnRemoveImage.show()
                    self.btnZoomImage.show()
                    self.btnNextImage.show()
                else:
                    QMessageBox.critical(self, "No Analyzed Results.",
                                         "The analyzed folders contain no Possible images.")
            else:
                QMessageBox.critical(self, "No Analyzed Results.",
                                     "There are no analyzed results in this folder. Try running Crack Analysis.")

    def VerifyConfident(self):
        self.current_image_id = 0
        self.consoleUpdate("Verifying Confident Results")
        self.verify_mode = "Confident"
        if(self.dir_path == ""):
            # Throw warning when no folder is selected
            QMessageBox.critical(self, "No folder selected.",
                                 "Please select a folder containing the images of analysis.")
        else:
            if(self.CheckVerifyFolder()):
                self.verificationImageFileList = []
                folder_dir = os.path.join(
                    self.dir_path, "Crack_Analysis", "Confident")
                for images in os.listdir(folder_dir):
                    # check image types and process
                    if (images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg")):
                        if("mask" not in images and "length_estimation" not in images):
                            self.verificationImageFileList.append(images)
                self.total_verification_images = len(
                    self.verificationImageFileList)
                self.consoleUpdate(
                    str(self.total_verification_images)+" images found for verification.")
                # Show Images
                if(self.total_verification_images > 0):
                    self.showImage(os.path.join(
                        self.dir_path, "Crack_Analysis", "Confident", self.verificationImageFileList[0]))
                    self.btnRemoveImage.show()
                    self.btnZoomImage.show()
                    self.btnNextImage.show()
                else:
                    QMessageBox.critical(self, "No Analyzed Results.",
                                         "The analyzed folders contain no Confident images.")
            else:
                QMessageBox.critical(self, "No Analyzed Results.",
                                     "There are no analyzed results in this folder. Try running Crack Analysis.")

    def generateReport(self):
        try:
            text_file_paths = []
            # Create Result Directory
            analyzedFolderPath = os.path.join(self.dir_path, "Crack_Analysis")
            ResultPossibleFolder = os.path.join(
                analyzedFolderPath, "Possible")
            ResultConfidentFolder = os.path.join(
                analyzedFolderPath, "Confident")
            # Check whether the specified path exists or not
            ResultPossibleFolderExists = os.path.exists(ResultPossibleFolder)
            ResultConfidentFolderExists = os.path.exists(ResultConfidentFolder)
            # Get all the text files analyzed
            if ResultPossibleFolderExists:
                for result_text in os.listdir(ResultPossibleFolder):
                    # check file types and process
                    if (result_text.endswith(".txt")):
                        text_file_paths.append(os.path.join(
                            ResultPossibleFolder, result_text))
            if ResultConfidentFolderExists:
                for result_text in os.listdir(ResultConfidentFolder):
                    # check file types and process
                    if (result_text.endswith(".txt")):
                        text_file_paths.append(os.path.join(
                            ResultConfidentFolder, result_text))
            self.consoleUpdate("Generating Report")
            # Auto Filename
            now = datetime.now()
            date_time = now.strftime("%b_%d_%Y-%H_%M_%S")
            Report_Filename = "Crack_Analysis_Report_"+date_time+".csv"
            f = open(os.path.join(self.dir_path, Report_Filename), "w")
            output_data = ""
            for data in text_file_paths:
                data_file = open(data, "r")
                output_data += str(data_file.read()) + "\n"
                data_file.close()
            f.write("Filename"+","+"Confidence Type"+","+"Date/Time Taken"+"," +
                    "Crack Types"+","+"Maximum Confidence Score"+","+"Crack Coverage %"+","+"Total Crack Length (pixels)"+"\n"+output_data+"\nTotal Files:"+str(len(text_file_paths)))
            f.close()
            # Open Report Containing Folder
            show_in_file_manager(os.path.join(self.dir_path, Report_Filename))
            self.consoleUpdate("Report Successfully Generated")
        except Exception as e:
            QMessageBox.critical(self, "Error Generating Report.",
                                 "Something went wrong." + str(e))

    def openFolder(self):
        self.dir_path = QFileDialog.getExistingDirectory(
            self, "Choose Directory", "")
        if(self.dir_path == ""):
            # Throw warning when no folder is selected
            QMessageBox.critical(self, "No folder selected.",
                                 "Please select a folder containing the images for crack analysis.")
        else:
            self.labelHelp.setText("Folder : "+str(self.dir_path))
            self.consoleUpdate("Folder Selected at "+str(self.dir_path))
            # Hide previous analysis images
            self.btnPrevImage.hide()
            self.btnRemoveImage.hide()
            self.btnZoomImage.hide()
            self.btnNextImage.hide()
            self.imageHolder.hide()

    def downloadPretrainedModel(self):
        # If the pretrained model does not exist, download and set up automatically
        modelPath = os.path.join(
            os.getcwd(), "output", "model_final.pth")
        modelExist = os.path.exists(modelPath)
        self.consoleUpdate("Checking if pretrained model exists.")
        if not modelExist:
            os.makedirs(os.path.join(
                os.getcwd(), "output"))
            self.consoleUpdate(
                "Pretrained Model does not exist. Downloading model.")
            url = 'https://drive.google.com/uc?id=1V5biplCaJYHTxIp8achw0KKpm52qPLIa'
            gdown.download(url, modelPath, quiet=False)
            self.consoleUpdate(
                "Model successfully downloaded.")
        else:
            self.consoleUpdate("Pretrained Model exists.")

    def runAnalysis(self):
        # Main Crack Detection Algorithm
        if(self.dir_path == ""):
            # Throw warning when no folder is selected
            QMessageBox.critical(self, "No folder selected.",
                                 "Please select a folder containing the images for crack analysis.")
        else:
            QMessageBox.information(self, "Running Analysis.",
                                    "Crack Analysis will begin and will take some time. Plese don't close any windows. Look at the terminal for progress.")
            # get the path/directory
            self.imageFileList = []
            folder_dir = self.dir_path
            for images in os.listdir(folder_dir):
                # check image types and process
                if (images.endswith(".png") or images.endswith(".jpg") or images.endswith(".jpeg")):
                    self.imageFileList.append(images)
            # Report Image Number
            self.total_images = len(self.imageFileList)
            self.consoleUpdate(str(self.total_images)+" images found in the folder " +
                               str(folder_dir))
            # Create Result Directory
            analyzedFolderPath = os.path.join(folder_dir, "Crack_Analysis")
            ResultPossibleFolder = os.path.join(
                analyzedFolderPath, "Possible")
            ResultConfidentFolder = os.path.join(
                analyzedFolderPath, "Confident")
            # Check whether the specified path exists or not
            ResultFolderExist = os.path.exists(analyzedFolderPath)
            ResultPossibleFolderExists = os.path.exists(ResultPossibleFolder)
            ResultConfidentFolderExists = os.path.exists(ResultConfidentFolder)
            # If not exists, create result directories
            if not ResultFolderExist:
                os.makedirs(analyzedFolderPath)
                self.consoleUpdate("Created folder "+str(analyzedFolderPath))
            if not ResultPossibleFolderExists:
                os.makedirs(ResultPossibleFolder)
                self.consoleUpdate("Created folder "+str(ResultPossibleFolder))
            if not ResultConfidentFolderExists:
                os.makedirs(ResultConfidentFolder)
                self.consoleUpdate("Created folder " +
                                   str(ResultConfidentFolder))
            worker = PercentageWorker()
            worker.percentageChanged.connect(self.progress.setValue)
            threading.Thread(
                target=analyseImage,
                args=("foo", self.dir_path, self.thresholdLower,
                      self.thresholdUpper, ResultPossibleFolder, ResultConfidentFolder),
                kwargs=dict(baz="baz", worker=worker),
                daemon=True,
            ).start()

    def initUI(self):
        # Set up UI and Method Bindings here
        self.resize(500, 800)
        self.setWindowTitle('ABECIS v.1.0 - S.M.A.R.T. Construction Research Group')
        # GUI Setup
        # Labels
        self.labelHelp = QLabel(
            "To begin, select a folder containing the images of building.", self)
        self.labelThresholdUpper = QLabel(
            "3. Set Upper Confidence Score Threshold : ", self)
        self.labelThresholdLower = QLabel(
            "2. Set Lower Confidence Score Threshold : ", self)

        self.labelThresholdUpperValue = QLabel(
            "("+str(self.thresholdUpper)+" %)", self)
        self.labelThresholdLowerValue = QLabel(
            "("+str(self.thresholdLower)+" %)", self)
        # Buttons
        self.btnSelectFolder = QPushButton("1. Select Image Folder", self)
        self.btnRunAnalysis = QPushButton("3. Run Crack Analysis", self)
        self.btnVerifyConfidentResults = QPushButton(
            "4. Verify Confident Results (Optional)", self)
        self.btnVerifyPossibleResults = QPushButton(
            "5. Verify Possible Results (Optional)", self)
        self.btnOpenResults = QPushButton(
            "6. Generate Report and Open Results Folder", self)
        # Threshold Slider Upper
        self.thresholdSliderUpper = QSlider(Qt.Orientation.Horizontal, self)
        self.thresholdSliderUpper.setMinimum(0)
        self.thresholdSliderUpper.setValue(self.thresholdUpper)
        self.thresholdSliderUpper.setMaximum(100)
        self.thresholdSliderUpper.setTickPosition(
            QSlider.TickPosition.TicksBelow)
        self.thresholdSliderUpper.setTickInterval(10)
        # Threshold Slider Lower
        self.thresholdSliderLower = QSlider(Qt.Orientation.Horizontal, self)
        self.thresholdSliderLower.setMinimum(0)
        self.thresholdSliderLower.setValue(self.thresholdLower)
        self.thresholdSliderLower.setMaximum(100)
        self.thresholdSliderLower.setTickPosition(
            QSlider.TickPosition.TicksBelow)
        self.thresholdSliderLower.setTickInterval(10)
        # Console Output
        self.Console = QLabel("", self)
        self.Console.setText(self.console_out)

        # Add widgets to layout
        vbox = QVBoxLayout()
        vbox.addWidget(self.labelHelp)
        vbox.addWidget(self.btnSelectFolder)

        # Threshold Slider Lower Widget add to layout
        thresholdHboxLower = QHBoxLayout()
        thresholdHboxLower.addWidget(self.labelThresholdLowerValue)
        thresholdHboxLower.addWidget(self.thresholdSliderLower)
        thresholdHboxLower.addWidget(self.labelThresholdLowerValue)

        # Threshold Slider Upper Widget add to layout
        thresholdHboxUpper = QHBoxLayout()
        thresholdHboxUpper.addWidget(self.labelThresholdUpperValue)
        thresholdHboxUpper.addWidget(self.thresholdSliderUpper)
        thresholdHboxUpper.addWidget(self.labelThresholdUpperValue)

        # Image Viewer/Holder Widget
        self.imageHolder = QLabel(self)

        # Progressbar
        self.progress = QtWidgets.QProgressBar()

        vbox.addWidget(self.labelThresholdLower)
        vbox.addLayout(thresholdHboxLower)
        vbox.addWidget(self.labelThresholdUpper)
        vbox.addLayout(thresholdHboxUpper)

        vbox.addWidget(self.btnRunAnalysis)
        vbox.addWidget(self.progress)

        vbox.addWidget(self.btnVerifyConfidentResults)
        vbox.addWidget(self.btnVerifyPossibleResults)
        vbox.addWidget(self.btnOpenResults)

        # Image Analysis Tools
        self.btnPrevImage = QPushButton("Previous", self)
        self.btnRemoveImage = QPushButton("Remove", self)
        self.btnZoomImage = QPushButton("Zoom", self)
        self.btnNextImage = QPushButton("Next", self)
        analysisHboxUpper = QHBoxLayout()
        analysisHboxUpper.addWidget(self.btnPrevImage)
        analysisHboxUpper.addWidget(self.btnRemoveImage)
        analysisHboxUpper.addWidget(self.btnZoomImage)
        analysisHboxUpper.addWidget(self.btnNextImage)

        vbox.addLayout(analysisHboxUpper)
        # Image Output
        vbox.addWidget(self.imageHolder)

        vbox.addStretch()
        vbox.addWidget(self.Console)

        # Bind GUI to Methods
        self.thresholdSliderUpper.valueChanged.connect(
            self.changeThresholdUpper)
        self.thresholdSliderLower.valueChanged.connect(
            self.changeThresholdLower)
        self.btnSelectFolder.clicked.connect(self.openFolder)
        self.btnRunAnalysis.clicked.connect(self.runAnalysis)
        self.btnVerifyPossibleResults.clicked.connect(self.VerifyPossible)
        self.btnVerifyConfidentResults.clicked.connect(self.VerifyConfident)
        self.btnOpenResults.clicked.connect(self.generateReport)
        # Image Analysis Functions
        self.btnPrevImage.clicked.connect(self.PrevImage)
        self.btnRemoveImage.clicked.connect(self.RemoveImage)
        self.btnZoomImage.clicked.connect(self.ZoomImage)
        self.btnNextImage.clicked.connect(self.NextImage)
        # Hide unnecessary buttons, e.g. show results, before analysis
        self.btnPrevImage.hide()
        self.btnRemoveImage.hide()
        self.btnZoomImage.hide()
        self.btnNextImage.hide()
        self.imageHolder.hide()
        self.setLayout(vbox)
        self.show()
        self.downloadPretrainedModel()


if __name__ == '__main__':
    qApp = QApplication(sys.argv)
    w = MainWindow()
    sys.exit(qApp.exec())
