from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import albumentations as alb
import matplotlib.pyplot as plt
import lxml.etree as ET
import shutil

annotation_folder = "annotations"
output_folder = "output"
image_folder = "images"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, image_folder), exist_ok=True)
os.makedirs(os.path.join(output_folder, annotation_folder), exist_ok=True)

def preprocessing_image(I_N,H_F,R_B_C,R_G,RGB_S,V_F,C_J,G_N,S_S_R,image_folder_loc):
    augmentor = alb.Compose([
        alb.HorizontalFlip(p=H_F),
        alb.RandomBrightnessContrast(p=R_B_C),
        alb.RandomGamma(p=R_G),
        alb.ShiftScaleRotate(scale_limit=S_S_R, rotate_limit=0,p=0.2),
        alb.RGBShift(p=RGB_S),
        alb.GaussNoise(p=G_N),
        alb.ColorJitter(p=C_J),
        alb.VerticalFlip(p=V_F)],
        alb.BboxParams(format="pascal_voc", label_fields=['class_label']))
    # Process each image
    image_files = os.listdir(image_folder_loc)
    for image_file in image_files:
        # Read XML data
        xml_file = os.path.splitext(image_file)[0] + ".xml"
        annotation_folder_loc =  os.path.join("annot", xml_file)
        xml_path = annotation_folder_loc
        if not os.path.exists(xml_path):
            # Load the image
            image_path = os.path.join(image_folder, image_file)
            img = cv2.imread(image_path)

            # Get image width and height
            image_height, image_width, _ = img.shape

            # Create a dummy annotation
            root = ET.Element("annotation")
            size_element = ET.SubElement(root, "size")
            width_element = ET.SubElement(size_element, "width")
            width_element.text = str(image_width)
            height_element = ET.SubElement(size_element, "height")
            height_element.text = str(image_height)
            depth_element = ET.SubElement(size_element, "depth")
            depth_element.text = "3"
            
            object_element = ET.SubElement(root, "object")
            name_element = ET.SubElement(object_element, "name")
            name_element.text = "0"
            pose_element = ET.SubElement(object_element, "pose")
            pose_element.text = "Unspecified"
            truncated_element = ET.SubElement(object_element, "truncated")
            truncated_element.text = "0"
            difficult_element = ET.SubElement(object_element, "difficult")
            difficult_element.text = "0"
            bndbox_element = ET.SubElement(object_element, "bndbox")
            xmin_element = ET.SubElement(bndbox_element, "xmin")
            xmin_element.text = "0"
            ymin_element = ET.SubElement(bndbox_element, "ymin")
            ymin_element.text = "0"
            xmax_element = ET.SubElement(bndbox_element, "xmax")
            xmax_element.text = "0"
            ymax_element = ET.SubElement(bndbox_element, "ymax")
            ymax_element.text = "0"
            Created_Custom = True
            # Save the dummy annotation XML
            output_xml_path = annotation_folder_loc
            with open(output_xml_path, 'wb') as f:
                f.write(ET.tostring(root))    

        with open(xml_path, 'r') as f:
            xml_data = f.read()
        # Parse XML
        label = ET.fromstring(xml_data)

        # Get image size
        image_size = label.find("size")
        image_width = int(image_size.find("width").text)
        image_height = int(image_size.find("height").text)

        # Access all the bndbox values
        bboxes = []
        for object_element in label.findall("object"):
            bndbox_element = object_element.find("bndbox")
            xmin = int(bndbox_element.find("xmin").text)
            ymin = int(bndbox_element.find("ymin").text)
            xmax = int(bndbox_element.find("xmax").text)
            ymax = int(bndbox_element.find("ymax").text)
            coords = [xmin, ymin, xmax, ymax]
            bboxes.append(coords)

        # Load the image
        image_path = os.path.join(image_folder, image_file)
        img = cv2.imread(image_path)

        class_labels = ['person'] * len(bboxes)
        
        # Apply augmentation and save augmented images with annotations
        for i in range(I_N):
            if xmin == 0:
                augmentor = alb.Compose([
                    alb.HorizontalFlip(p=H_F),
                    alb.RandomBrightnessContrast(p=R_B_C),
                    alb.RandomGamma(p=R_G),
                    alb.ShiftScaleRotate(scale_limit=S_S_R,rotate_limit=0, p=0.2),
                    alb.RGBShift(p=RGB_S),
                    alb.GaussNoise(p=G_N),
                    alb.ColorJitter(p=C_J),
                    alb.VerticalFlip(p=V_F)])
                augmented = augmentor(image=img)
                augmented_image = augmented['image']
            else:
                augmented = augmentor(image=img, bboxes=bboxes, class_label=class_labels)
                augmented_image = augmented['image']
                augmented_bboxes = augmented['bboxes']


            # Save augmented image
            output_image_file = os.path.splitext(image_file)[0] + f"_augmented_{i+1}.jpg"
            output_image_path = os.path.join(output_folder, image_folder, output_image_file)
            cv2.imwrite(output_image_path, augmented_image)

            if xmin != 0:
                # Update annotation XML with augmented bounding boxes
                for idx, bbox in enumerate(augmented_bboxes):
                    xmin, ymin, xmax, ymax = bbox
                    xmin /= augmented_image.shape[1]
                    ymin /= augmented_image.shape[0]
                    xmax /= augmented_image.shape[1]
                    ymax /= augmented_image.shape[0]

                    # Update original XML with augmented bounding boxes
                    bndbox_element = label.findall("object")[idx].find("bndbox")
                    bndbox_element.find("xmin").text = str(int(xmin * image_width))
                    bndbox_element.find("ymin").text = str(int(ymin * image_height))
                    bndbox_element.find("xmax").text = str(int(xmax * image_width))
                    bndbox_element.find("ymax").text = str(int(ymax * image_height))

                # Save augmented annotation XML
                output_xml_file = os.path.splitext(image_file)[0] +f"_augmented_{i+1}"+ ".xml"
                output_xml_path = os.path.join(output_folder, annotation_folder, output_xml_file)
                with open(output_xml_path, 'wb') as f:
                    f.write(ET.tostring(label))
            else:
                # write the answer
                name, format = xml_file.split(".")
                source_file = "annot/"+xml_file
                destination_folder = "output/annotations/"+name+f"_augmented_{i+1}"+ ".xml"

                # Copy the file to the destination folder
                shutil.copy(source_file, destination_folder)

app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def index():
    return render_template("index.html")

@app.route('/preprocess', methods=['POST'])
def preprocess():

    I_N = int(request.form['I_N'])
    H_F = float(request.form['H_F'])
    R_B_C = float(request.form['R_B_C'])
    R_G = float(request.form['R_G'])
    RGB_S = float(request.form['RGB_S'])
    V_F = float(request.form['V_F'])
    G_N = float(request.form['G_N'])
    C_J = float(request.form['C_J'])
    S_S_R = float(request.form['S_S_R'])
    I_D = str(request.form["D"])
    images_we_have = len(os.listdir(I_D))
    
    preprocessing_image(I_N, H_F, R_B_C, R_G, RGB_S, V_F, C_J, G_N, S_S_R,I_D)
    
    img_count = len(os.listdir("output/images/"))
    file_dir = os.path.join(os.getcwd(), "output")
    print(images_we_have,"hello")

    return render_template("result.html",
                            total_image=img_count,
                            Directory=file_dir,
                            non_aug_images = images_we_have)

if __name__ == '__main__':
    app.run(debug=True)