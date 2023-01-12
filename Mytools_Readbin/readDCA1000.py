import numpy as np
import matplotlib.pyplot as plt

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
    RXptr = 0
    Sampleptr = 0
    # 分别为4个通道建立数组，虚实柔和在一起
    for i in range(0, len(adcData), 4):
        temp = complex(adcData[i], adcData[i + 2])
        retVal_list[RXptr].append(temp)
        Sampleptr += 1

        temp = complex(adcData[i + 1], adcData[i + 3])
        retVal_list[RXptr].append(temp)
        Sampleptr += 1

        if Sampleptr == numADCSamples:
            Sampleptr = 0
            RXptr += 1
            if RXptr == 4:
                RXptr = 0

    retVal = np.array(retVal_list)
    # resize整体数据的大小
    # print(retVal.shape)
    # print(retVal)
    return retVal

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
    numChirps = config.numChirps
    numRX = config.numRX

    #读取文件
    retVal = readDCA1000( bin_file + '.bin', config.numADCSamples , config.numChirps )

    #计算numFrames * numChiprs 的大小
    numFrames = int( len(retVal[0]) / (numADCSamples * numChirps) )

#不想保持一致的话，就把下面的代码注释掉就行了
###################################################################################################
    #修改数据，进行fix(契合成 mmwavestudio 一致的数据)
    for j in range(0,numRX):
        dataRX = retVal[j]
        dataRX.resize((numFrames * numChirps, numADCSamples))
        # 对数据进行修正
        for i in range(numFrames * numChirps):
            dataRX[i] = dataRX[i][::-1]
###################################################################################################


    '''毫米波雷达参数'''
    # 距离采样点数
    Nr = numChirps
    # 每个chirp采样率(Sample Rate)
    fs = config.fs
    # Frequency Slope = B/T,调频斜率
    K = config.K

    '''基本参数'''
    # 电磁波传播速度
    c = 3 * (10 ** 8)

    # 距离FFT，得到距离-时间图像
    f = fs / numADCSamples * np.array(range(1, numADCSamples + 1))
    t0 = f / K
    r = c * t0 / 2

    # 绘图
    # 取RX0 的数据
    dataRX = retVal[1]
    dataRX.resize((numFrames * numChirps, numADCSamples))

    fft1D = np.zeros((numFrames * numChirps, numADCSamples), dtype='complex128')
    for i in range(numFrames * numChirps):
        # 对每一行做fft
        fft1D[i] = np.fft.fft(dataRX[i])
    # 对应上Matlab的fft1D
    fft1D = fft1D.T
    # 绘图
    # plt.xticks( np.arange(0, numFrames * numChirps,numFrames * numChirps/10)  )
    # plt.yticks( np.arange(0,r[-1],r[-1]/8))
    plt.imshow(np.abs(fft1D), aspect='auto', extent=(1, numFrames * numChirps, r[-1], 0))
    # plt.imshow(np.abs(fft1D), aspect='auto')
    plt.ylabel('Range (m)')
    plt.xlabel('Pulses number')
    plt.title('Hrrp')
    plt.show()
