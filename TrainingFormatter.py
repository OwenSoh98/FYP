import os
import shutil
from lxml import etree
import yaml

class TrainingFormatter():
    def __init__(self, train, val, test, name):
        # Note: YOLOv4 requires absolute path to the images/annotations, use gdrive_path for it
        self.gdrive_path = '/content/drive/MyDrive/ColabNotebooks/normal/'

        self.train_path = train
        self.val_path = val
        self.test_path = test

        self.training_format = './data/' + name + '/'

        self.default_class = 0
        self.classes = [x.strip() for x in open('labels.txt').readlines()]
        self.name = name

        #self.YOLOv4_Training()
        self.YOLOv5_Training()
        #self.SSD_Training()

    def create_directory(self, path):
        """ Make directory """
        if not os.path.exists(path):
            os.makedirs(path)

    def create_YOLO_txt(self, data_path, target_path, txt_file):
        """ Create train, text and val files"""
        with open(os.path.join(target_path, txt_file), mode='w') as f:
            for folder_name in os.listdir(data_path):
                folder_path = os.path.join(data_path, folder_name)

                for filename in os.listdir(folder_path):
                    if filename.endswith('.jpg'):
                        f.write(self.gdrive_path + 'obj/' + filename +'\n')

    def get_file_ids(self, dataset_path):
        """ Get all image file ids """
        file_ids = []
        for foldername in os.listdir(dataset_path):
            folderpath = os.path.join(dataset_path, foldername)
            for filename in os.listdir(folderpath):
                if filename.endswith('.jpg'):
                    id = os.path.splitext(filename)[0]
                    file_ids.append(id)
        return file_ids

    def create_SSD_txt(self, target_path):
        """ Create train, text and val files"""
        train_ids = self.get_file_ids(self.train_path)
        val_ids = self.get_file_ids(self.val_path)
        test_ids = self.get_file_ids(self.test_path)

        with open(os.path.join(target_path, 'train.txt'), mode='w') as f:
            for item in train_ids:
                f.write(item+"\n")

        with open(os.path.join(target_path, 'val.txt'), mode='w') as f:
            for item in val_ids:
                f.write(item+"\n")

        with open(os.path.join(target_path, 'test.txt'), mode='w') as f:
            for item in test_ids:
                f.write(item+"\n")
        
        with open(os.path.join(target_path, 'trainval.txt'), mode='w') as f:
            for item in train_ids:
                f.write(item+"\n")
            
            for item in val_ids:
                f.write(item+"\n")

    def create_YOLO_cfg(self, target_path):
        """ Create obj.names and obj.data """
        shutil.copy('labels.txt', os.path.join(target_path, 'obj.names'))

        classes = len([x.strip() for x in open('labels.txt').readlines()])
        with open(os.path.join(target_path, 'obj.data'), mode='w') as f:
            f.write('classes = {}\n'.format(classes))
            f.write('train = {}train.txt\n'.format(self.gdrive_path))
            f.write('valid = {}valid.txt\n'.format(self.gdrive_path))
            f.write('test = {}test.txt\n'.format(self.gdrive_path))
            f.write('names = {}obj.names\n'.format(self.gdrive_path))
            f.write('backup = backup/\n')

    def voc2yolo(self, xml_path):
        """ Convert from VOC to YOLO format"""
        xml_file = open(xml_path)
        my_tree = etree.parse(xml_file)
        my_root = my_tree.getroot()

        w = int(my_root.find('size').find('width').text)
        h = int(my_root.find('size').find('height').text)

        objects = []
        for x in my_root.findall('object'):
            name = x.find('name').text
            bbox = x.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)

            x_center = round((x1 + ((x2 - x1) / 2)) / w, 6)
            y_center = round((y1 + ((y2 - y1) / 2)) / h, 6)
            width = round((x2 - x1) / w, 6)
            height = round((y2 - y1) / h, 6)
            
            try:
                class_number = self.classes.index(name)
            except ValueError:
                class_number = self.default_class
            # print(xml_path)
            # breakpoint()
            obj =  {'x': x_center, 'y': y_center, 'w': width, 'h': height, 'class': class_number}
            objects.append(obj)

        return objects

    def save_yolo_labels(self, labels, filename, target_path):
        """ Create and save YOLO labels"""
        label_file_path = os.path.join(target_path, filename + '.txt')
        label_file = open(label_file_path, "w")

        for i in range(len(labels)):
            content = '{} {} {} {} {}\n'.format(labels[i]['class'], labels[i]['x'], labels[i]['y'], 
                        labels[i]['w'], labels[i]['h'])
            label_file.write(content)

        label_file.close()

    def loop_images(self, data_folder, cmd, extra_cmd=None):
        """ Loop through images in a folder and perform action depending on cmd"""
        for filename in os.listdir(data_folder):
            if filename.endswith('.xml'):
                xml_path = os.path.join(data_folder, filename)
                obj = self.voc2yolo(xml_path)

                name = os.path.basename(xml_path)
                name_wo_ext = os.path.splitext(name)[0]

                if cmd == 'YOLOv4-Training':
                    path = self.training_format + 'YOLOv4-Training/obj/'
                    self.save_yolo_labels(obj, name_wo_ext, path)

                    source = os.path.join(os.path.dirname(xml_path), name_wo_ext + '.jpg') 
                    destination = os.path.join(path, name_wo_ext + '.jpg')
                    shutil.copy(source, destination)
                elif cmd == 'YOLOv5-Training':
                    path = self.training_format + 'YOLOv5-Training/' + self.name + '/labels/' + extra_cmd +'/'
                    img_path = self.training_format + 'YOLOv5-Training/' + self.name + '/images/'  + extra_cmd +'/'
                    self.save_yolo_labels(obj, name_wo_ext, path)

                    source = os.path.join(os.path.dirname(xml_path), name_wo_ext + '.jpg') 
                    destination = os.path.join(img_path, name_wo_ext + '.jpg')
                    shutil.copy(source, destination)
                elif cmd == 'SSD-Training':
                    path = self.training_format + 'SSD-Training/'
                    path_img = path + 'JPEGImages/'
                    path_xml = path + 'Annotations/'

                    # Copy Images
                    source = os.path.join(os.path.dirname(xml_path), name_wo_ext + '.jpg') 
                    destination = os.path.join(path_img, name_wo_ext + '.jpg')
                    shutil.copy(source, destination)

                    # Copy XML
                    source = os.path.join(xml_path) 
                    destination = os.path.join(path_xml, filename)
                    shutil.copy(source, destination)

    def YOLOv4_Training(self):
        """ Create YOLOv4-Training File Formats"""
        cmd = 'YOLOv4-Training'              
        path1 = self.training_format + cmd + '/'
        path2 = self.training_format + cmd + '/obj/'

        self.create_directory(self.training_format)
        self.create_directory(path1)
        self.create_directory(path2)

        self.loop_images(self.train_path, cmd)
        self.loop_images(self.val_path, cmd)
        self.loop_images(self.test_path, cmd)

        self.create_YOLO_txt(self.train_path, path1, 'train.txt')
        self.create_YOLO_txt(self.val_path, path1, 'valid.txt')
        self.create_YOLO_txt(self.test_path, path1, 'test.txt')
        self.create_YOLO_cfg(path1)

    def SSD_Training(self):
        """ Create SSD-Training File Formats"""
        cmd = 'SSD-Training'              
        path1 = self.training_format + cmd + '/'
        path2 = path1 + 'Annotations/'
        path3 = path1 + 'ImageSets/Main/'
        path4 = path1 + 'JPEGImages/'

        self.create_directory(self.training_format)
        self.create_directory(path1)
        self.create_directory(path2)
        self.create_directory(path3)
        self.create_directory(path4)
        
        self.loop_images(self.train_path, cmd)
        self.loop_images(self.val_path, cmd)
        self.loop_images(self.test_path, cmd)
        self.create_SSD_txt(path3)
        shutil.copy('./labels.txt', path1)

    def YOLOv5_Training(self):
        """ Create YOLOv5-Training File Formats"""
        cmd = 'YOLOv5-Training'      

        path1 = self.training_format + cmd + '/' + self.name + '/images/train/'
        path2 = self.training_format + cmd + '/' + self.name + '/images/valid/'
        path3 = self.training_format + cmd + '/' + self.name + '/images/test/'
        path4 = self.training_format + cmd + '/' + self.name + '/labels/train/'
        path5 = self.training_format + cmd + '/' + self.name + '/labels/valid/'
        path6 = self.training_format + cmd + '/' + self.name + '/labels/test/'

        self.create_directory(path1)
        self.create_directory(path2)
        self.create_directory(path3)
        self.create_directory(path4)
        self.create_directory(path5)
        self.create_directory(path6)

        d = {'train': './data/data/' + self.name + '/' + cmd + '/' + self.name + '/images/train/',
        'val': './data/data/' + self.name + '/' + cmd + '/' + self.name + '/images/valid/',
        'test': './data/data/' + self.name + '/' + cmd + '/' + self.name + '/images/test/',
        'nc': 2,
        'names': ['vehicle', 'bike']}
        with open(self.training_format + cmd + '/' + 'config.yaml', 'w') as yaml_file:
            yaml.dump(d, yaml_file, default_flow_style=False)

        self.loop_images(self.train_path, cmd, 'train')
        self.loop_images(self.val_path, cmd, 'valid')
        self.loop_images(self.test_path, cmd, 'test')

