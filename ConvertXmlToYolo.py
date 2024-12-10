import os
import xml.etree.ElementTree as ET

# Path to the directory containing XML files and the directory to save YOLO labels
xml_dir = r"C:\Users\Admin\Desktop\yolov5\Data_Training\images_ch00\XML"  # Update path as needed
yolo_dir = "C:/Users/Admin/Desktop/yolov5/dataset/output"  # Directory to save YOLO labels
classes = ["car"]  # List of your classes
subclass =['sedan', 'van', '3wheels_car', 'car_others', 'pickup_truck']

# Create the YOLO output directory if it doesn't exist
os.makedirs(yolo_dir, exist_ok=True)

# Function to convert XML annotation to YOLO format
def convert_xml_to_yolo(xml_file, output_dir):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Image size
        size = root.find("size")
        if size is None:
            raise ValueError("Missing 'size' element in the XML file.")
        img_width = int(size.find("width").text)
        img_height = int(size.find("height").text)

        yolo_lines = []

        # Iterate over each object in the XML
        for obj in root.findall("object"):
            cls_obj = obj.find("name")
            if cls_obj.text not in classes:
                continue  # Skip classes not in the list
            cls_id = subclass.index(obj.find("subname").text)

            # Get the bounding box coordinates
            bndbox = obj.find("bndbox")
            if bndbox is None:
                print(f"Warning: Missing 'bndbox' element in {xml_file} for object '{cls_obj}'.")
                continue

            try:
                xmin = float(bndbox.get('left'))
                ymin = float(bndbox.get("top"))
                xmax = float(bndbox.get("right"))
                ymax = float(bndbox.get("bottom"))
                #subclass = bndbox.find('subname')
            except (AttributeError, ValueError) as e:
                print(f"Warning: Invalid bounding box data in {xml_file} for object '{cls_obj}'.")
                continue

            # Calculate YOLO format coordinates
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            # Append the YOLO line
            yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Save the YOLO label file if there are any valid data
        if yolo_lines:
            output_file = os.path.join(output_dir, os.path.basename(xml_file).replace(".xml", ".txt"))
            with open(output_file, "w") as f:
                f.write("\n".join(yolo_lines))
        else:
            print(f"Warning: No valid objects found in {xml_file}.")

    except ET.ParseError as e:
        print(f"Error parsing {xml_file}: {e}")
    except Exception as e:
        print(f"Error processing {xml_file}: {e}")

# Convert all XML files in the directory
for xml_file in os.listdir(xml_dir):
    if xml_file.endswith(".xml"):
        convert_xml_to_yolo(os.path.join(xml_dir, xml_file).replace('\\', '/'), yolo_dir)

print("Conversion completed!")
print(subclass)
