# pre-process ScanNet 2D data
# note: depends on the sens file reader from ScanNet:
#       https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py
# if export_label_images flag is on:
#   - depends on https://github.com/ScanNet/ScanNet/tree/master/BenchmarkScripts/util.py
#   - also assumes that label images are unzipped as scene*/label*/*.png
# expected file structure:
#  - prepare_2d_data.py
#  - https://github.com/ScanNet/ScanNet/tree/master/BenchmarkScripts/util.py
#  - https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py
#
# example usage:
#    python prepare_2d_data.py --scannet_path data/scannetv2 --output_path data/scannetv2_images --export_label_images

import argparse
import os, sys
import numpy as np
import skimage.transform as sktf
import imageio
from multiprocessing import Pool
from tqdm import tqdm
import subprocess
import zipfile


try:
    from SensorData import SensorData
except:
    print('Failed to import SensorData (from ScanNet code toolbox)')
    sys.exit(-1)
try:
    import util
except:
    print('Failed to import ScanNet code toolbox util')
    sys.exit(-1)


# params
parser = argparse.ArgumentParser()
parser.add_argument('--scannet_path', required=True, help='path to scannet data')
parser.add_argument('--output_path', required=True, help='where to output 2d data')
parser.add_argument('--export_label_images', dest='export_label_images', action='store_true')
parser.add_argument('--label_type', default='label-filt', help='which labels (label or label-filt)')
parser.add_argument('--frame_skip', type=int, default=20, help='export every nth frame')
parser.add_argument('--label_map_file', default='', help='path to scannetv2-labels.combined.tsv (required for label export only)')
parser.add_argument('--output_image_width', type=int, default=320, help='export image width')
parser.add_argument('--output_image_height', type=int, default=240, help='export image height')

parser.set_defaults(export_label_images=False)
opt = parser.parse_args()
# if opt.export_label_images:
#     print(opt.export_label_images)
#     assert opt.label_map_file != ''
print(opt)


def print_error(message):
    sys.stderr.write('ERROR: ' + str(message) + '\n')
    sys.exit(-1)

# from https://github.com/ScanNet/ScanNet/tree/master/BenchmarkScripts/2d_helpers/convert_scannet_label_image.py
def map_label_image(image, label_mapping):
    mapped = np.copy(image)
    for k,v in label_mapping.items():
        mapped[image==k] = v
    return mapped.astype(np.uint8)

def randomize():
    np.random.seed()

def run_command_generic(command):
    #This command could have multiple commands separated by a new line \n
    # some_command = "export PATH=$PATH://server.sample.mo/app/bin \n customupload abc.txt"

    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)

    (output, err) = p.communicate()  

    #This makes the wait possible
    p_status = p.wait()

    #This will give you the output of the command being executed
    print("Command output: " + output.decode('utf-8'))

def process(x):
    scene = x[0]
    i = x[1]
    i_total = x[2]
    label_map = x[3]

    sens_file = os.path.join(opt.scannet_path, scene, scene + '.sens')
    label_path = os.path.join(opt.scannet_path, scene, opt.label_type)

    zip_path = os.path.join(opt.scannet_path, scene, '%s_2d-%s.zip'%(scene, opt.label_type))
    # run_command_generic('unzip -f %s -d %s'%(zip_path, os.path.join(opt.scannet_path, scene)))
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(os.path.join(opt.scannet_path, scene))
    print('Unzipping %s'%zip_path)

    # if opt.export_label_images and not os.path.isdir(label_path):
    #     zip_path = os.path.join(opt.scannet_path, scene, '%s_2d-%s.zip'%(scene, opt.label_type))
    #     # if os.path.exists(zip_path):
    #         os.system('unzip %s -d %s'%(zip_path, os.path.join(opt.scannet_path, scene)))
    #         print('Unzipping %s'%zip_path)
    #     else:
    #         print_error('Error: using export_label_images option but label path %s does not exist' % label_path)
    # Variable to indicate what kind of sensor data is to be extracted
    reextract = []
    output_color_path = os.path.join(opt.output_path, scene, 'color')
    # if not os.path.isdir(output_color_path):
    os.makedirs(output_color_path, exist_ok=True)
    reextract.append('color')
    output_depth_path = os.path.join(opt.output_path, scene, 'depth')
    # if not os.path.isdir(output_depth_path):
    os.makedirs(output_depth_path, exist_ok=True)
    reextract.append('depth')
    output_pose_path = os.path.join(opt.output_path, scene, 'pose')
    # if not os.path.isdir(output_pose_path):
    os.makedirs(output_pose_path, exist_ok=True)
    reextract.append('pose')
    output_label_path = os.path.join(opt.output_path, scene, 'label')
    # if opt.export_label_images and not os.path.isdir(output_label_path):
    if opt.export_label_images:
        os.makedirs(output_label_path, exist_ok=True)
        reextract.append('label')

    # If we only have to extract 'label', we can skip exporting sensor data
    # We can also continue to the next scene if we do not have to reextract anything
    print('reextract: ', reextract)
    if (len(reextract) == 1 and reextract[0] == 'label') or reextract == []:
        sys.stdout.write('\r[ %d | %d ] %s\tskipping sensordata...' % ((i+1), i_total, scene))
        sys.stdout.flush()
        if reextract == []:
            return

        # read and export
        sys.stdout.write('\r[ %d | %d ] %s\tloading...' % ((i + 1), i_total, scene))
        sys.stdout.flush()
        print(sens_file)
        sd = SensorData(sens_file)
    else:
        sys.stdout.write('\r[ %d | %d ] %s\texporting...' % ((i + 1), i_total, scene))
        sys.stdout.flush()
        sd = SensorData(sens_file)
        sd.export_color_images(output_color_path, image_size=[opt.output_image_height, opt.output_image_width], frame_skip=opt.frame_skip)
        sd.export_depth_images(output_depth_path, image_size=[opt.output_image_height, opt.output_image_width], frame_skip=opt.frame_skip)
        sd.export_poses(output_pose_path, frame_skip=opt.frame_skip)

    if opt.export_label_images:
        for f in range(0, len(sd.frames), opt.frame_skip):
            label_file = os.path.join(label_path, str(f) + '.png')
            image = np.array(imageio.imread(label_file))
            image = sktf.resize(image, [opt.output_image_height, opt.output_image_width], order=0, preserve_range=True)
            mapped_image = map_label_image(image, label_map)
            imageio.imwrite(os.path.join(output_label_path, str(f) + '.png'), mapped_image)
    
def main():
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    label_mapping = None
    if opt.export_label_images:
        label_map = util.read_label_mapping(opt.label_map_file, label_from='id', label_to='nyu40id')

    scenes = [d for d in os.listdir(opt.scannet_path) if os.path.isdir(os.path.join(opt.scannet_path, d))]
    print('Found %d scenes' % len(scenes))
    print(scenes[:5])
    # for i in range(len(scenes)):
    combined_x = [(scenes[i], i, len(scenes), label_map) for i in range(len(scenes))]
    with Pool(processes=32, initializer=randomize) as pool:
        # _ = list(tqdm(pool.imap_unordered(process, zip(scenes, range(len(scenes)), [len(scenes)]*len(scenes))), total=len(scenes)))
        _ = list(tqdm(pool.imap_unordered(process, combined_x), total=len(scenes)))
        
    print('')


if __name__ == '__main__':
    main()

