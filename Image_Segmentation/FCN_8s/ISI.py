import torch
import numpy as np

def DataInput(Pre_Label,Tre_Label):
    num0, dim0, pixel00, pixel01 = Pre_Label.shape
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    if(Pre_Label.shape == Tre_Label.shape):
        for i in range(num0):
            for j in range(dim0):
                for k in range(pixel00):
                    for l in  range(pixel01):
                        pre_value = Pre_Label[i][j][k][l]
                        tre_value = Tre_Label[i][j][k][l]
                        if(pre_value == tre_value == 1):
                            TP += 1
                            
                        if(pre_value == 1 and tre_value == 0):
                            FP += 1
                            
                        if(pre_value == 0 and tre_value == 1):
                            FN += 1
                            
                        if(pre_value == tre_value == 0):
                            TN += 1
                            
    else:
        print("Size error:")
        print("Pre_Label size(",Pre_Label.shape,")!=Tre_Label size(",Tre_Label.shape,")")
    if TP == 0:
        print("TP=0")
    if FP == 0:
        print("FP=0")
    if FN == 0:
        print("FN=0")
    if TN == 0:
        print("TN=0")
    return TP,FP,FN,TN

def mIOU(Pre_Label,Tre_Label):
    TP,FP,FN,TN = DataInput(Pre_Label,Tre_Label)
    IOU = TP/(TP+FP+FN)
    return IOU

def mACC(Pre_Label,Tre_Label):
    TP,FP,FN,TN = DataInput(Pre_Label,Tre_Label)
    AP = (TP + TN)/(TP+TN+FP+FN)
    return AP
        
def mPrecision(Pre_Label,Tre_Label):
    TP,FP,FN,TN, = DataInput(Pre_Label,Tre_Label)
    Precision = TP/(TP + FP)
    return Precision

def mRecall(Pre_Label,Tre_Label):
    TP,FP,FN,TN, = DataInput(Pre_Label,Tre_Label)
    Recall = TP/(TP + FN)
    return Recall

def ISI(epochs,Pre_Label,Tre_Label):
    """
    Image segmentation index:图像分割指标
    IOU:交并比
    ACC:平均准确度
    Precision:查准率
    Recall:召回率
    """
    TP,FP,FN,TN = DataInput(Pre_Label,Tre_Label)
    out = {"epoch":str(epochs)}
    return[str(epochs),str(round(TP/(TP+FP+FN),4)),str(round((TP + TN)/(TP+TN+FP+FN),4)),str(round(TP/(TP+FP),4)),str(round(TP/(TP+FN),4))]