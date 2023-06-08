## Instru tracker and analysis pipeline for fluorecent image
#required input parameter is path (ome.tif location)

import argparse
import os
from glob import glob 
from processFinal import Process
from tools.help import Helper


parser = argparse.ArgumentParser(
    description="""Download results in the folder and ouputs results
                """)

parser.add_argument('--path','-p',required=True, help='Path to folder. eg. C:/data/imgs')
parser.add_argument('--save','-s',default=False, help='Do you want to save', required= False)
parser.add_argument('--savePath','-d',default="default", help='Path where results are saved', required = False)
parser.add_argument('--threshold','-t',default=0.1, help='trackesr threshold')
parser.add_argument('--vis','-v',default=False,type = bool, help='visualization')

# Save arguments and set other parameters
args = parser.parse_args()

if args.savePath == "default":
    args.savePath = args.path
    args.savePath = os.path.join(args.path, "results")

if not os.path.exists(args.savePath):
    os.makedirs(args.savePath)
    print("Created a new path", args.savePath)

#Print visualization option
if args.vis == True:
    print("Visualizing")
else:
    print("print wont visualize")

#Init classes
process = Process(args)

#Process images to videos, track and save metaparameters
metadata, tracks = process.readVideo()

#Analyze 

#helper = Helper(args)
#Analyse
#helper.ProcessPickle()

print("Anlysis Done! Remember to look at the data, please!")
