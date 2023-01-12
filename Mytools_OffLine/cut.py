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
        self.fps = None

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
                my_config.fps = 1000 / int(args[5])

            elif (args[0] == 'channelCfg'):
                my_config.numRX = 4

    return my_config

#仿Matlab写的filter函数
def filter_matlab(b,a,x):
    y = []
    y.append(b[0] * x[0])
    for i in range(1,len(x)):
        y.append(0)
        for j in range(len(b)):
            if i >= j :
                y[i] = y[i] + b[j] * x[i - j ]
                j += 1
        for l in range(len(b)-1 ):
            if i >l:
                y[i] = (y[i] - a[l+1] * y[i -l-1])
                l += 1
        i += 1
    return np.array(y)


if __name__ == '__main__':

    bin_file = 'python_adc_data_2'
    #读取使用的配置文件
    config = readCofig('../config/iwr6843_fps20.cfg')
    numADCSamples = config.numADCSamples
    numChirps = config.numChirps #每一帧有多少个chirp
    numRX = config.numRX

    #读取文件
    retVal = readDCA1000( bin_file + '.bin', config.numADCSamples , config.numChirps )
    print(retVal.shape)

    #计算numFrames * numChiprs 的大小
    numFrames = int( len(retVal[0]) / (numADCSamples * numChirps) )

#不想保持一致的话，就把下面的代码注释掉就行了
# ###################################################################################################
#     #修改数据，进行fix(契合成 mmwavestudio 一致的数据)
    for j in range(0,numRX):
        dataRX = retVal[j]
        dataRX.resize((numFrames * numChirps, numADCSamples))
        # 对数据进行修正
        for i in range(numFrames * numChirps):
            dataRX[i] = dataRX[i][::-1]
# ###################################################################################################


    '''毫米波雷达参数'''
    # 每个chirp采样率(Sample Rate)
    fs = config.fs
    # Frequency Slope = B/T,调频斜率
    K = config.K

    '''基本参数'''
    # 电磁波传播速度
    c = 3 * (10 ** 8)
    # 绘图序号
    figure_num = 1

    # 距离FFT，得到距离-时间图像
    f = fs / numADCSamples * np.array(range(1, numADCSamples + 1))
    t0 = f / K
    r = c * t0 / 2

    # 时间点数比例尺
    t_ruler = (numFrames / config.fps) / (numFrames * numChirps)


    ## 绘图，距离FFT，得到距离-时间图像
    # 取RX0 的数据
    dataRX = retVal[0]
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
    plt.figure(figure_num)
    figure_num += 1
    plt.subplot(211)
    plt.imshow(np.abs(fft1D), aspect='auto', extent=(1, numFrames * numChirps, r[-1], 0))
    # plt.imshow(np.abs(fft1D), aspect='auto')
    plt.ylabel('Range (m)')
    plt.xlabel('Pulses number')
    plt.title('Hrrp')
    # plt.show()

    for i in range(0,10):
        fft1D[i][:] = 0

    ## 杂波去除
    dataFFT1D_complex  = []  #用于将每一帧去除背景杂波的数据存储起来
    # 每次处理一帧，一帧256chirps
    for idxFrame in range(0 , fft1D.shape[1] , numChirps):

        dataFrameFFT1D = fft1D[:,idxFrame:idxFrame+numChirps]

        background = np.tile (np.mean(fft1D , axis=1),(numChirps,1))
        background = background.T

        # 去噪声
        dataFrameFFT1D = dataFrameFFT1D - background

        # 堆叠
        if idxFrame == 0:
            dataFFT1D_complex = dataFrameFFT1D
        else:
            dataFFT1D_complex = np.hstack((dataFFT1D_complex , dataFrameFFT1D))

    plt.subplot(212)
    plt.imshow(np.abs(dataFFT1D_complex), aspect='auto', extent=(1, numFrames * numChirps, r[-1], 0))
    # plt.imshow(np.abs(fft1D), aspect='auto')
    plt.ylabel('Range (m)')
    plt.xlabel('Pulses number')
    plt.title('Hrrp after mean')
    # plt.show()

    ## 进行信号积累
    dataFFT1D = abs(dataFFT1D_complex)
    fft_jilei = np.zeros((numADCSamples,1))
    for j in range(0, numADCSamples):
        for i in range(0, numFrames * numChirps):
            #对去除杂波后的回波进行积累，否则杂波会影响积累的效果
            fft_jilei[j] = fft_jilei[j] + dataFFT1D[j][i]
    plt.figure(figure_num)
    figure_num += 1
    plt.subplot(211)
    plt.plot(r,dataFFT1D[:,1])
    plt.ylabel('幅度')
    plt.xlabel('距离')
    plt.title('距离FFT ')
    plt.subplot(212)
    plt.plot(r, fft_jilei)
    plt.ylabel('幅度')
    plt.xlabel('距离')
    plt.title('非相干积累后的距离FFT ')
    # plt.show()


    ## 提取相位信息
    real_data = np.real(fft1D)
    imag_data = np.imag(fft1D)
    # 相位抽取,对每一个点进行相位抽取
    angle_fft = np.zeros( (numADCSamples, numFrames * numChirps) ) # numFrames * numChirps个脉冲，numADCSamples为距离
    for i in range(0, numFrames * numChirps):
        for j in range(0,numADCSamples):
            angle_fft[j][i] = math.atan2(imag_data[j][i],real_data[j][i])

    #找出人体的位置，即能量最大处（extract phase from selected range bin）
    range_max = 0
    max_num = 0
    for j in range(0, numADCSamples):
        if fft_jilei[j] > range_max:
            range_max = fft_jilei[j]
            max_num = j
    angle_fft_last = angle_fft[max_num,:] #取max距离处的相位
    #进行相位解缠  phase unwrapping，这个时候才是正确的相位信息
    n = 0
    for i in range(1, numFrames * numChirps):
        diff = angle_fft_last[i] - angle_fft_last[i-1]
        if diff > np.pi:
            angle_fft_last[i:] = angle_fft_last[i:] - 2 * np.pi
            n  = n + 1
        elif diff < - np.pi:
            angle_fft_last[i:] = angle_fft_last[i:] + 2 * np.pi
    #但是实际上要取的是相位差 phase difference，所以再进行相位差
    angle_fft_last1 = np.zeros((numFrames * numChirps - 1))
    for i in range(0,numFrames * numChirps - 1):
        angle_fft_last1[i] = angle_fft_last[i+1] - angle_fft_last[i]
    #绘图
    plt.figure(figure_num)
    figure_num += 1
    t =  1/20 * np.array(range(0,  numFrames * numChirps))
    plt.plot(t[0 : numFrames * numChirps-1 ], angle_fft_last1)
    plt.title('相位差波形')
    # plt.show()


    ## 分离呼吸心跳信号
    # 滤波器参数4阶
    fs_b = 20 # 呼吸滤波器的采样率
    fs_h = 20 # 心跳滤波器的采样率
    from scipy import signal
    from scipy.io import loadmat
    #对于滤波器而言应该最好的是自己设计，但没有办法还原其滤波器设计的具体参数,如果需要设计需要使用下面的函数设计
    #b, a = signal.iirfilter(17, [2 * np.pi * 50, 2 * np.pi * 200], rs=60,btype='band', analog=True, ftype='cheby2')
    #读取保存下来的E_B 和 E_H
    E_B = loadmat('E_B.mat') #E_B里面存储的是呼吸滤波器的参数SOS_e_b和G_e_b
    temp_G = 1
    for i in E_B['G_e_b']:
        temp_G = temp_G * i
    b, a = signal.sos2tf(E_B['SOS_e_b'])
    b = b * temp_G
    #滤波
    breath_data = filter_matlab(b, a, angle_fft_last1)
    #绘图
    plt.figure(figure_num)
    figure_num += 1
    plt.plot(range(0,numChirps * numFrames - 1), breath_data) #呼吸波形
    plt.ylabel('幅度')
    plt.xlabel('时间(t)')
    plt.title('滤波器滤波后呼吸波形')
    # plt.show()

    breath_data_0 = breath_data
    mid = np.mean(abs(breath_data))
    #呼吸波形去掉干扰的过程就是将小于中值的部分当作干扰，并将之置为零
    for i in range(0,len(breath_data)):
        temp = abs(breath_data[i])
        if (temp < mid ):
            breath_data_0[i] = 0

    # 绘图
    plt.figure(figure_num)
    figure_num += 1
    plt.plot(range(0,numChirps * numFrames - 1), breath_data_0) #呼吸波形
    plt.ylabel('幅度')
    plt.xlabel('时间(t)')
    plt.title('去掉心跳干扰后的呼吸波形')
    # plt.show()

    #Spectral Estimation -FFT -Peak interval
    breath_fre = np.fft.fftshift(abs(np.fft.fft(breath_data_0)))
    # 绘图
    plt.figure(figure_num)
    figure_num += 1
    freq_b = range( (-(numFrames * numChirps)//2 + 1) , ((numFrames * numChirps)//2) )
    freq_b = np.array(freq_b) * (fs_b/(numFrames * numChirps))
    plt.plot(freq_b, breath_fre)
    plt.title('滤波器滤波后呼吸波形的频率')
    #plt.show()

    # 对于滤波器而言应该最好的是自己设计，但没有办法还原其滤波器设计的具体参数,如果需要设计需要使用下面的函数设计
    # b, a = signal.iirfilter(17, [2 * np.pi * 50, 2 * np.pi * 200], rs=60,btype='band', analog=True, ftype='cheby2')
    # 读取保存下来的E_B 和 E_H
    E_H = loadmat('E_H.mat') #同理E_H里面存储的是心跳滤波器的参数SOS_e_H和G_e_h
    temp_G = 1
    for i in E_H['G_e_h']:
        temp_G = temp_G * i
    b, a = signal.sos2tf(E_H['SOS_e_h'])
    b = b * temp_G
    # 滤波
    heart_data = filter_matlab(b, a, angle_fft_last1)
    #绘图
    plt.figure(figure_num)
    figure_num += 1
    plt.plot( [ i * t_ruler for i in list(range(0,numChirps * numFrames - 1))], heart_data) #呼吸波形
    plt.ylabel('幅度')
    plt.xlabel('时间(t)')
    plt.title('滤波器滤波后心跳波形')


    ## 运动片段去除（窗口大小的影响）（用在心跳信号部分）
    # 暂不计算

    ## 展示绘图
    plt.show()