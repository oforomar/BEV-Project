import configargparse
import os
import numpy as np

classes = ['Car', 'Pedestrian', 'Cyclist']

def line_edit(line):
    global threshold
    line_edit = ''
    dont_care = "-1 -1 -1 -1 -1 -1 -1"
    if float(line[-2]) >= threshold:
       class_ = classes[int(float(line[-1]))]
       h, w, l, x, y, z, ry, score = line[5], line[4], line[3], line[0], line[1], line[2], line[6], line[7]
       line_edit = " ".join((class_,dont_care,h,w,l,x,y,z,ry,score))
    return line_edit

def read_label_file(folderpath):
    data = {}
    fileLocation = os.listdir(folderpath)
    for currentFile in fileLocation:
        filepath = os.path.join(folderpath, currentFile)
        print("Opening FIle: ",currentFile)  
        with open(filepath, 'r') as f:
            values_list = []
            for line in f.readlines():
                line = line.split()
                line = line_edit(line)
                if len(line) == 0:
                    continue
                #print(line)    
                values_list.append(line) #list of current line.
            try:
                data[currentFile] = values_list
            except ValueError:
                pass

    return data

def save_label_file(data, save_path):
    if not os.path.exists(save_path):
    	os.makedirs(save_path)
    for key in data.keys():
    	with open(os.path.join(save_path,key), 'w') as f:
    	     f.writelines(line+"\n" for line in data[key])
    print("COMPLETED")	     	

parser = configargparse.ArgParser(description='KITTI LABEL')

parser.add_argument('--avod_label_path', type=str, default='',
                    help='path to avod output label')

parser.add_argument('--save_path', type=str, default='',
                    help='path to save the converted labels output')
parser.add_argument('--threshold', type=float, default=0.02,
                    help='BEV detection threshold')
                    
args = parser.parse_args()
threshold = args.threshold

print(args)

dataread = read_label_file(args.avod_label_path)
save_label_file(dataread, args.save_path)

