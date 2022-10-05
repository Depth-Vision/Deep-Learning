import csv
import numpy as np
from cProfile import label
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt


plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))      # 用来正常显示中文标签

def Ite_Loss_Chart():
    path1="./RunRecord/IterationLoss.csv"
    title="Iteration_Loss_Chart"

    plt.rcParams['font.sans-serif']=['SimHei']                              # 用来正常显示负号                                
    plt.rcParams['axes.unicode_minus']=False
                                                                            
    with open(path1) as f:                                                  # 打开文件文件并将内容储存在reader中
        reader=csv.reader(f)                                                # 读取并将内容储存在列表reader中
        next(reader)                                                        # next()函数获取第一行，即文件头
        Iteration,LossRate = [],[]

        a = 0
        b = 1
        c = 1
        for row in reader:
            Ite = float(row[0])
            Iteration.append(Ite)
            Loss = float(row[1])

            if Loss > b:
                Loss = b+0.0001
            b = Loss
            
            out = open("./RunRecord/Loss.csv", "a+")
            csv_writer1 = csv.writer(out, dialect = "excel",lineterminator = '\n')
            if(Ite == 0):
                csv_writer1.writerow(["Iteration","LossRate"])
                csv_writer1.writerow([str(c),str(Loss)])
                a += 1
            else:
                csv_writer1.writerow([str(c),str(Loss)])
            c += 1
            LossRate.append(Loss)


    plt.plot(Iteration,LossRate,label="IterationLoss",linewidth=1)
    plt.xlabel('Iteration') 
    plt.ylabel('LossRate')
    plt.title(title) 
    plt.legend() 
    plt.savefig('./RunRecord/Ite_Loss_Chart.svg')
    plt.savefig('./RunRecord/Ite_Loss_Chart.png')
    plt.close()

        
def ISI_Chart():
    path="./RunRecord/ISI_Record.csv"
    title="ISI_Chart"

    plt.rcParams['font.sans-serif']=['SimHei']                              # 用来正常显示负号 
    plt.rcParams['axes.unicode_minus']=False

    with open(path) as f:                                                   # 打开文件文件并将内容储存在reader中
        reader=csv.reader(f)                                                # 读取并将内容储存在列表reader中
        next(reader)                                                        # next()函数获取第一行，即文件头
        Epoch,IOU,ACC,Precision,Recall = [],[],[],[],[]
        
        for row in reader:
            epoche = float(row[0])
            Epoch.append(epoche)

            iou = float(row[1])
            IOU.append(iou)

            ac = float(row[2])
            ACC.append(ac)

            pre = float(row[3])
            Precision.append(pre)

            rec = float(row[4])
            Recall.append(rec)


    plt.plot(Epoch,IOU,c="red",label="IOU",linewidth=1)
    plt.plot(Epoch,ACC,c="blue",label="ACC",linewidth=1)
    plt.plot(Epoch,Precision,c="green",label="Precision",linewidth=1)
    plt.plot(Epoch,Recall,c="black",label="Recall",linewidth=1,linestyle='--')
    plt.xlabel('epoch') 
    plt.ylabel('ratio')
    plt.title(title) 
    plt.legend() 
    plt.savefig('./RunRecord/ISI_Chart.svg')
    plt.savefig('./RunRecord/ISI_Chart.png')
    plt.close()

if __name__ == '__main__':
    ISI_Chart()
    Ite_Loss_Chart()