## Instru tracker and analysis pipeline for fluorecent image
#required input parameter is path (ome.tif location)

import argparse
import os
from tools.source import opticalFlow


parser = argparse.ArgumentParser(
    description="""Download results in the folder and ouputs results
                """)

parser.add_argument('--path','-p',required=True, help='Path to folder. eg. C:/data/imgs')
parser.add_argument('--save','-s',default=False, help='Do you want to save', required= False)
parser.add_argument('--vis','-v',default=False,type = bool, help='visualization')
parser.add_argument('--mode','-m',default="vol", type = str, help='Analysis Type')
parser.add_argument('--channel','-c',default="TexasRed", type = str, help='Analysis Type')
parser.add_argument('--level','-l',default=8, type = int, help='Analysis Type')


# Save arguments and set other parameters
args = parser.parse_args()

args.savePath = os.path.join(args.path, "results")

if not os.path.exists(args.savePath):
    os.makedirs(args.savePath)
    print("Created a new path", args.savePath)

#Print visualization option
if args.vis == True:
    print("Visualizing")
else:
    print("wont visualize")

#Print visualization option
if args.save == True:
    print("Saving")
else:
    print("wont save")

#Init classes
_ = opticalFlow(args)

print("Anlysis Done! Remember to look at the data, please!")
