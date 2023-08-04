import os
import json

DATAPATH = './data/'
files = os.listdir(path=DATAPATH)#[1:]
BASEPAY = 2.5

with open("bonus.txt","w+") as txt:
    for idx, file in enumerate(files):
        with open(DATAPATH + file) as json_file:
            subjdata = json.load(json_file)       
            bonus = float(subjdata['money']) - BASEPAY
            subjID = subjdata['subjectID']
            txt.write("%s,%1.2f \n" %(subjID,bonus))