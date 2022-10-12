import csv
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt

plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))      # 用来正常显示中文标签
        
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