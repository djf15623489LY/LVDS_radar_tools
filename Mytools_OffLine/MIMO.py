import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"] #用来正常显示中文字符
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
import math

class Config:
    def __init__(self):
        self.numADCSamples = None
        self.numChirps = None
        self.numRX = None

        self.fs = None
        self.K = None

#仿Matlab写的readDCA1000函数
def readDCA1000(filename , numADCSamples , numChirps):
    # 相当于Matlab里面的fread
    adcData = np.fromfile(filename , dtype='int16')
    #对应4个通道
    retVal_list = [[],[],[],[]]
    #实部数据
    Im_Data_All = []
    Im_Data_All_1 = []
    Im_Data_All_2 = []
    #虚部数据
    Re_Data_All = []
    Re_Data_All_1 = []
    Re_Data_All_2 = []
    RXptr = 0
    Sampleptr = 0
    # 对于4 收 2 发
    for i in range(0, len(adcData), 4):
        Im_Data_All_1.append(adcData[i])
        Im_Data_All_2.append(adcData[i+1])
        Re_Data_All_1.append(adcData[i+2])
        Re_Data_All_2.append(adcData[i+3])

    Im_Data_All = Im_Data_All_1 + Im_Data_All_2
    Re_Data_All = Re_Data_All_1 + Re_Data_All_2
    return Im_Data_All,Re_Data_All

def readCofig(fliename):
    cfg_file = open(fliename, 'r')
    cfg = cfg_file.readlines()
    my_config = Config()
    for line in cfg:
        args = line.split()
        if (len(args) > 0):
            if (args[0] == 'profileCfg'):
                my_config.numADCSamples = int(args[10])
                my_config.fs = int(args[11]) * (10 ** 3)
                my_config.K = int(args[8]) * (10 ** 12)

            elif (args[0] == 'frameCfg'):
                my_config.numChirps = int(args[3])

            elif (args[0] == 'channelCfg'):
                my_config.numRX = 4

    return my_config


if __name__ == '__main__':

    bin_file = 'python_adc_data_2'
    #读取使用的配置文件
    config = readCofig('../config/iwr6843_fps20.cfg')
    numADCSamples = config.numADCSamples
    numChirps = config.numChirps #每一帧有多少个chirp
    numRX = config.numRX

    #读取文件
    Im_Data_All, Re_Data_All = readDCA1000( bin_file + '.bin', config.numADCSamples , config.numChirps )

    plt.subplot(211)
    plt.plot(Im_Data_All)
    plt.subplot(212)
    plt.plot(Re_Data_All)
    plt.show()