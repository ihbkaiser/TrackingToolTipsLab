import supervision as sv
from cotracker_utils import *
import networkx as nx
from ROI import *
from Models import *
from Image import *
from LoadEndovis15 import *
from Video import *
from Mask import *

load = LoadEndovis15(1)
frames = load.process(multiprocessing=False)

def check(model_tip, abstract_tip, d = 100):
    fit = False
    for tip in abstract_tip:
        # resize tip to fit with current ROI
        tip = (tip[0] - ROI[0][0], tip[1] - ROI[1][0])
        # check if model_tip fit with tip
        fit = (tip[0] - d/2 <= model_tip[0] <= tip[0] + d/2) and (tip[1] - d/2 <= model_tip[1] <= tip[1] + d/2)
    return fit
        
def evaluate(load, frames, index, d = 100):
    frame = frames[index]
    model_tip, handle, endpoints = frame.getIns()
    tool_tips = [load.tool_tip[i][index][2:] for i in range(len(load.tool_tip))]
    true_pos = [False]*len(model_tip)
    satisfy = [False]*len(tool_tips)
    true_positive = 0
    for idx, tip in enumerate(model_tip):
        true_pos[idx] = False
        for jdx, tool_tip in enumerate(tool_tips):
            # print("tool_tip", tool_tip)
            if(check(tip, tool_tip, d) and not satisfy[jdx]):
                true_pos[idx] = True
                true_positive += 1
                satisfy[jdx] = True
                break
    false_positive = len(model_tip) - true_positive 
    false_negative = 0
    for tool_satisfy in satisfy:
        if not tool_satisfy:
            false_negative += 1
    accuracy = (true_positive) / (true_positive + false_positive + false_negative)
    print(f'Frame index: {index}, accuracy : {accuracy*100}%, TP: {true_positive}, FP: {false_positive}, FN: {false_negative}')
    return accuracy

jaccard_metric = 0
for idx in range(len(frames)):
    jaccard_metric += evaluate(load, frames, idx)

print(f'Test on {len(frames)} frames, avg jaccard metric: {jaccard_metric/len(frames)*100}%')