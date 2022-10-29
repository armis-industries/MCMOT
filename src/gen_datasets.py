from glob import glob
import os.path as osp
import os
from re import L
import shutil
from sre_constants import JUMP

import cv2
import numpy as np
from tqdm import tqdm
import typer

MCMOT_ROOT = "/home/ec2-user/MCMOT"
FAIRMOT_MANIFEST = os.path.join(MCMOT_ROOT, "dataset/drone-data/detrac.train")
COMBINED_MANIFEST = os.path.join(MCMOT_ROOT, "dataset/combined_detrac.train")
VISDRONE_MANIFEST = os.path.join(MCMOT_ROOT, "dataset/VisDrone2019/detrac.train")
FAIRMOT_DRONE_DATA = os.path.join(MCMOT_ROOT, "dataset/drone-data/")
IMAGE_DEST_PATH = os.path.join(FAIRMOT_DRONE_DATA, "images")
LABEL_DEST_PATH = os.path.join(FAIRMOT_DRONE_DATA, "labels_with_ids")
YOLO_INFERENCE_PATH = "/home/ec2-user/yolov7/runs/detect"

CLOSE_OBJECTS = {
                    "IR_AIRPLANE_001", "IR_AIRPLANE_015", "IR_AIRPLANE_018", "IR_AIRPLANE_062", "IR_AIRPLANE_063", "IR_AIRPLANE_064",
                    "IR_AIRPLANE_065", "IR_AIRPLANE_066", "IR_AIRPLANE_067", "IR_BIRD_008", "IR_BIRD_009", "IR_BIRD_027", "IR_BIRD_030", "IR_BIRD_031",
                    "IR_BIRD_032", "IR_BIRD_034", "IR_BIRD_035", "IR_BIRD_036", "IR_BIRD_037", "IR_DRONE_001", "IR_DRONE_002", "IR_DRONE_007", "IR_DRONE_008"
                    "IR_DRONE_009", "IR_DRONE_018", "IR_DRONE_024", "IR_DRONE_025", "IR_DRONE_026", "IR_DRONE_034", "IR_DRONE_035", "IR_DRONE_043",
                    "IR_DRONE_056", "IR_DRONE_074", "IR_DRONE_095", "IR_DRONE_100", "IR_DRONE_101", "IR_DRONE_109", "IR_DRONE_119", "IR_DRONE_120",
                    "IR_DRONE_128", "IR_DRONE_129", "IR_DRONE_137", "IR_DRONE_157", "IR_HELICOPTER_001", "IR_HELICOPTER_002", "IR_HELICOPTER_003",
                    "IR_HELICOPTER_004", "IR_HELICOPTER_005", "IR_HELICOPTER_023", "IR_HELICOPTER_024", "IR_HELICOPTER_025", "IR_HELICOPTER_026",
                    "IR_HELICOPTER_027", "IR_HELICOPTER_040", "IR_HELICOPTER_041", "IR_HELICOPTER_042", "IR_HELICOPTER_054", "IR_HELICOPTER_055",
                    "V_AIRPLANE_002", "V_AIRPLANE_003", "V_AIRPLANE_008", "V_AIRPLANE_012", "V_AIRPLANE_018", "V_AIRPLANE_035", "V_AIRPLANE_038",
                    "V_AIRPLANE_039", "V_AIRPLANE_040", "V_AIRPLANE_041", "V_AIRPLANE_042", "V_AIRPLANE_044", "V_AIRPLANE_046", "V_AIRPLANE_047",
                    "V_AIRPLANE_048", "V_AIRPLANE_049", "V_AIRPLANE_050", "V_BIRD_005", "V_BIRD_011", "V_BIRD_016", "V_BIRD_017", "V_BIRD_018",
                    "V_BIRD_019", "V_BIRD_020", "V_BIRD_021", "V_BIRD_022", "V_BIRD_048", "V_DRONE_001", "V_DRONE_005", "V_DRONE_013", "V_DRONE_023", 
                    "V_DRONE_025", "V_DRONE_027", "V_DRONE_036", "V_DRONE_040", "V_DRONE_054", "V_DRONE_072", "V_DRONE_074", "V_DRONE_075", "V_DRONE_078",
                    "V_DRONE_080", "V_DRONE_081", "V_DRONE_086", "V_DRONE_087", "V_DRONE_091", "V_DRONE_092", "V_DRONE_096", "V_DRONE_101", "V_HELICOPTER_001",
                    "V_HELICOPTER_002", "V_HELICOPTER_007", "V_HELICOPTER_008", "V_HELICOPTER_011", "V_HELICOPTER_012", "V_HELICOPTER_019", "V_HELICOPTER_020",
                    "V_HELICOPTER_021", "V_HELICOPTER_022", "V_HELICOPTER_023", "V_HELICOPTER_026", "V_HELICOPTER_027", "V_HELICOPTER_028", "V_HELICOPTER_032",
                    "V_HELICOPTER_033", "V_HELICOPTER_035", "V_HELICOPTER_036", "V_HELICOPTER_037", "V_HELICOPTER_038", "V_HELICOPTER_042", "V_HELICOPTER_044",
                    "V_HELICOPTER_049", "V_HELICOPTER_050", "V_HELICOPTER_051", "V_HELICOPTER_053", "V_HELICOPTER_056"
}

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def get_video_meta(video):
    vid = cv2.VideoCapture(video)
    fps = vid.get(cv2.CAP_PROP_FPS)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    return {'frameRate': int(fps), 'imWidth': int(width), 'imHeight': int(height), 'seqLength': int(frame_count)}


def get_sequence_info(path):
    """Write seqinfo.ini file from video metadata"""
    video_path = glob(f"{path}/*.mp4")[0]
    video_name = video_path.split('/')[-1].split('.mp4')[0]
    if video_name not in CLOSE_OBJECTS:
        return False
    #if video_name[:2] == 'IR':
    #    return False
    # Create /home/ec2-user/FairMOT/drone-data/video_name
    fairmot_write_dir = os.path.join(FAIRMOT_DRONE_DATA, video_name)
    # mkdirs(fairmot_write_dir)
    
    
    #if 'IR' in video_name or 'V_DRONE_017' in video_name:
    #   return False
    
    # video_meta = get_video_meta(video_path)
    
    #with open(os.path.join(fairmot_write_dir, 'seqinfo.ini'), 'w') as f:
    #    f.write('[Sequence]\n')
    #    f.write(f'name={video_name}\n')
    #    f.write(f'imDir={video_name}\n')
    #    f.write(f'frameRate={video_meta["frameRate"]}\n')
    #    f.write(f'seqLength={video_meta["seqLength"]}\n')
    #    f.write(f'imWidth={video_meta["imWidth"]}\n')
    #    f.write(f'imHeight={video_meta["imHeight"]}\n')
    #    f.write("imExt='.jpg'\n")
    return video_name


def rename_frames(path):
    files = list(set([int(file.split('/')[-1].split('.')[0]) for file in glob(f"{path}/**")]))
    sorted_files = sorted(files)
    for idx, file in enumerate(sorted_files):
        if file == idx:
            continue
        old_jpg = os.path.join(path, str(file) + '.jpg')
        new_jpg = os.path.join(path, str(idx) + '.jpg')
        os.rename(old_jpg, new_jpg)
        
        old_txt = os.path.join(path, str(file) + '.txt')
        new_txt = os.path.join(path, str(idx) + '.txt')
        os.rename(old_txt, new_txt)


def gen_dot_train_file(data_root, rel_path, out_root, f_name='detrac.train'):
    """
    To generate the dot train file
    :param data_root:
    :param rel_path:
    :param out_root:
    :param f_name:
    :return:
    """
    if not (os.path.isdir(data_root) and os.path.isdir(out_root)):
        print('[Err]: invalid root')
        return

    out_f_path = out_root + '/' + f_name
    cnt = 0
    with open(out_f_path, 'w') as f:
        root = data_root + rel_path
        # print(f"Root: {root}")
        seqs = [x for x in os.listdir(root)]
        if "drone.train" in seqs:
            seqs.remove("detrac.train")
        # print(f"Seqs: {seqs}")
        seqs.sort()
        # seqs = sorted(seqs, key=lambda x: int(x.split('_')[-1]))
        for seq in tqdm(seqs):
            img_dir = root + '/' + seq  # + '/img1'
            # print(f"Image Directory: {img_dir}")
            img_list = [x for x in os.listdir(img_dir)]
            img_list.sort()
            # print(f"Img_list: {img_list}")
            for img in img_list:
                if img.endswith('.jpg'):
                    img_path = img_dir + '/' + img
                    if os.path.isfile(img_path):
                        item = img_path.replace(data_root + '/', '')
                        # print(item)
                        f.write(item + '\n')
                        cnt += 1

    print('Total {:d} images for training'.format(cnt))
    

def write_label_files(path, video_name):
    """Creates combined and individual inference txt files"""
    fairmot_write_dir = os.path.join(FAIRMOT_DRONE_DATA, video_name)
    # detection_dir = os.path.join(fairmot_write_dir, 'det')
    # Create /home/ec2-user/FairMOT/drone-data/train/video_name/det
    # mkdirs(detection_dir)
    fairmot_label_dir = os.path.join(FAIRMOT_DRONE_DATA, 'labels_with_ids', video_name)
    # Create /home/ec2-user/FairMOT/drone-data/train/video_name/video_name
    #mkdirs(fairmot_label_dir)
    detection_file = os.path.join(fairmot_write_dir, 'det/det.txt')
    labels_path = os.path.join(path, 'labels')
    # print(f"Writing file: {detection_file}")    10: 'drone',
    if 'DRONE' in video_name:
        class_id = "10"
        # class_id = "0"
    elif 'HELICOPTER' in video_name:
        class_id = "11"
        # class_id = "1"
    elif 'BIRD' in video_name:
        class_id = "12"
        # class_id = "2"
    elif 'AIRPLANE' in video_name:
        class_id = "13"
        # class_id = "3"
    data = []
    track_id = 2
    for file in glob(f"{labels_path}/*.txt"):
        with open(file, 'r') as f:
            line = f.readlines()[0]
        label = file.split('/')[-1].split(".txt")[0].split('_')[-1]
        items = line.split(' ')
        items[0] = class_id
        items.insert(1, str(track_id))
        data.append(' '.join(items))
        label_write_path = os.path.join(fairmot_label_dir, label+'.txt')
        # print(f"Writing file: {label_write_path}")
        # Writes individual text files
        with open(label_write_path, 'w') as f:
            f.write(' '.join(items))
    #create_det_file(detection_file, data)


def create_det_file(write_path, content):
    with open(write_path, 'w') as out:
        for line in content:
            out.write(line+'\n')
            
def move_data(dir_path, dir_name):
    img_in = os.path.join(dir_path, dir_name)
    for i in glob(f"{img_in}/**"):
        filename = i.split('/')[-1].split("_")[-1]
        frame_number = filename.split('.jpg')[0]
        label_file = os.path.join(LABEL_DEST_PATH, dir_name, str(frame_number)+'.txt')
        #print(label_file)
        if os.path.exists(label_file):
            img_out = os.path.join(IMAGE_DEST_PATH, dir_name, filename)
            shutil.copy(i, img_out)
    return


def remove_unmatched_files(path):
    """Removes frames with no data tracked"""
    jpg_lst = set([i.split('/')[-1].split('.jpg')[0] for i in glob(f"{path}/*.jpg")])
    txt_lst = set([i.split('/')[-1].split('.txt')[0] for i in glob(f"{path}/*.txt")])
    union = jpg_lst.intersection(txt_lst)
    for file in glob(f"{path}/**"):
        name = file.split('/')[-1].split('.')[0]
        if name not in union:
            os.remove(file)


def make_txt_files(path):
    """Creates Empty Text Files"""
    jpg_lst = [i.split('/')[-1].split('.jpg')[0] for i in glob(f"{path}/*.jpg")]
    txt_lst = [i.split('/')[-1].split('.txt')[0] for i in glob(f"{path}/*.txt")]
    for jpg in jpg_lst:
        if jpg not in txt_lst:
            with open(f"{path}/{jpg}.txt", "w") as f:
                f.write('')


def write_manifest():
    """Writes manifest file for all training data"""
    files = []
    for dir in glob("/home/ec2-user/FairMOT/drone-data/train/**"):
        name = dir.split("/")[-1]
        if "IR_" in name or "BIRD" in name:
            continue
        new_path = os.path.join(dir, name)
        for img in glob(f"{new_path}/*.jpg"):
            img = img.split("FairMOT/")[-1]
            files.append(img)

    with open(FAIRMOT_MANIFEST, 'w') as f:
        for line in files:
            f.write(line+'\n')
            
            
def write_combined_manifest():
    """Writes manifest file for all training data"""
    with open(FAIRMOT_MANIFEST, 'r') as f:
        drone_data = f.readlines()
    with open(VISDRONE_MANIFEST, 'r') as f:
        visdrone_data = f.readlines()
    with open(COMBINED_MANIFEST, 'w') as f:
        for line in drone_data:
            f.write(line)
        for line in visdrone_data:
            f.write(line)
    
    
    
def main(path: str):
    mkdirs(FAIRMOT_DRONE_DATA)
    for dir in tqdm(glob(f"{path}/**"), total=len(CLOSE_OBJECTS)):
        video = get_sequence_info(dir)
        if not video:
            continue
        # image_path = os.path.join(FAIRMOT_DRONE_DATA, video, video)
        image_path = os.path.join(FAIRMOT_DRONE_DATA, "images")
        mkdirs(image_path)
        mkdirs(os.path.join(image_path, video))
        label_path = os.path.join(FAIRMOT_DRONE_DATA, "labels_with_ids")
        mkdirs(label_path)
        mkdirs(os.path.join(label_path, video))
        # print(f'Processing: {video}')
        write_label_files(dir, video)
        move_data(dir, video)
    gen_dot_train_file(data_root="/home/ec2-user/MCMOT/",
                    rel_path="/dataset/drone-data/images",
                    out_root="/home/ec2-user/MCMOT/dataset/drone-data", f_name="detrac.train")
    #write_manifest()
    write_combined_manifest()
        #remove_unmatched_files(image_path)
        #rename_frames(image_path)
        

                
                
if '__name__' == '__main__':
    typer.run(main)
main("/home/ec2-user/yolov7/runs/detect")