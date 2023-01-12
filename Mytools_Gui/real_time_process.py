import threading as th
import numpy as np
import socket
import math
from scipy import signal
from scipy.io import loadmat

##仿Matlab写的filter函数
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

def RefactorADC(adcData,adc_sample):
    retVal_list = [[], [], [], []]
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

        if Sampleptr == adc_sample:
            Sampleptr = 0
            RXptr += 1
            if RXptr == 4:
                RXptr = 0

    retVal = np.array(retVal_list)
    return retVal

## 距离FFT，得到距离-时间图像
def Hrrp(dataRX,numADCSamples,numChirps,numFrames):
    dataRX.resize((numFrames * numChirps, numADCSamples))

    fft1D = np.zeros((numFrames * numChirps, numADCSamples), dtype='complex128')

    for i in range(numFrames * numChirps):
        # 对每一行做fft
        fft1D[i] = np.fft.fft(dataRX[i])
    # 对应上Matlab的fft1D
    fft1D = fft1D.T
    return fft1D

## 杂波去除
def NoiseRemove(fft1D,numADCSamples,numChirps,numFrames):
    dataFFT1D_complex = []  # 用于将每一帧去除背景杂波的数据存储起来
    # 每次处理一帧，一帧256chirps
    for idxFrame in range(0, fft1D.shape[1], numChirps):

        dataFrameFFT1D = fft1D[:, idxFrame:idxFrame + numChirps]
        background = np.tile(np.mean(fft1D, axis=1), (numChirps, 1))
        background = background.T
        # 去噪声
        dataFrameFFT1D = dataFrameFFT1D - background

        # 堆叠
        if idxFrame == 0:
            dataFFT1D_complex = dataFrameFFT1D
        else:
            dataFFT1D_complex = np.hstack((dataFFT1D_complex, dataFrameFFT1D))

    return dataFFT1D_complex

## 进行信号积累
def SignalAdd(dataFFT1D_complex,numADCSamples,numChirps,numFrames):
    dataFFT1D = abs(dataFFT1D_complex)
    fft_jilei = np.zeros((numADCSamples,1))
    for j in range(0, numADCSamples):
        for i in range(0, numFrames * numChirps):
            #对去除杂波后的回波进行积累，否则杂波会影响积累的效果
            fft_jilei[j] = fft_jilei[j] + dataFFT1D[j][i]
    return fft_jilei

## 提取相位信息
def PhaseUnwrapping(fft_jilei,fft1D,numADCSamples,numChirps,numFrames):
    real_data = np.real(fft1D)
    imag_data = np.imag(fft1D)
    # 相位抽取,对每一个点进行相位抽取
    angle_fft = np.zeros((numADCSamples, numFrames * numChirps))  # numFrames * numChirps个脉冲，numADCSamples为距离
    for i in range(0, numFrames * numChirps):
        for j in range(0, numADCSamples):
            angle_fft[j][i] = math.atan2(imag_data[j][i], real_data[j][i])

    # 找出人体的位置，即能量最大处（extract phase from selected range bin）
    range_max = 0
    max_num = 0
    for j in range(0, numADCSamples):
        if fft_jilei[j] > range_max:
            range_max = fft_jilei[j]
            max_num = j
    angle_fft_last = angle_fft[max_num, :]  # 取max距离处的相位
    # 进行相位解缠  phase unwrapping，这个时候才是正确的相位信息
    n = 0
    for i in range(1, numFrames * numChirps):
        diff = angle_fft_last[i] - angle_fft_last[i - 1]
        if diff > np.pi:
            angle_fft_last[i:] = angle_fft_last[i:] - 2 * np.pi
            n = n + 1
        elif diff < - np.pi:
            angle_fft_last[i:] = angle_fft_last[i:] + 2 * np.pi
    # 但是实际上要取的是相位差 phase difference，所以再进行相位差
    angle_fft_last1 = np.zeros((numFrames * numChirps - 1))
    for i in range(0, numFrames * numChirps - 1):
        angle_fft_last1[i] = angle_fft_last[i + 1] - angle_fft_last[i]

    return angle_fft_last1

## 分离呼吸心跳信号
def Separate_HB(heart_a,heart_b, breath_a , breath_b,angle_fft_last1,numADCSamples,numChirps,numFrames):

    # 呼吸滤波
    breath_data = filter_matlab(breath_b,breath_a,angle_fft_last1)

    breath_data_0 = breath_data
    mid = np.mean(abs(breath_data))
    #呼吸波形去掉干扰的过程就是将小于中值的部分当作干扰，并将之置为零
    for i in range(0,len(breath_data)):
        temp = abs(breath_data[i])
        if (temp < mid ):
            breath_data_0[i] = 0

    #Spectral Estimation -FFT -Peak interval
    breath_fre = np.fft.fftshift(abs(np.fft.fft(breath_data_0)))



    #心跳滤波
    heart_data = filter_matlab(heart_b, heart_a, angle_fft_last1)
    # Spectral Estimation -FFT -Peak interval
    heart_fre = np.fft.fftshift(abs(np.fft.fft(heart_data)))

    return breath_fre,heart_fre



class UdpListener(th.Thread):
    def __init__(self, name, bin_data, data_frame_length, data_address, buff_size ,save_flag):
        """
        :param name: str
                        Object name

        :param bin_data: queue object
                        A queue used to store adc data from udp stream

        :param data_frame_length: int
                        Length of a single frame

        :param data_address: (str, int)
                        Address for binding udp stream, str for host IP address, int for host data port

        :param buff_size: int
                        Socket buffer size
        """
        th.Thread.__init__(self, name=name)
        self.bin_data = bin_data
        self.frame_length = data_frame_length
        self.data_address = data_address
        self.buff_size = buff_size
        self.save_flag = save_flag

    def run(self):
        # convert bytes to data type int16
        dt = np.dtype(np.int16)
        dt = dt.newbyteorder('<')
        # array for putting raw data
        np_data = []
        # count frame
        count_frame = 0
        data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data_socket.bind(self.data_address)
        print("Create socket successfully")
        print("Now start data streaming")
        # main loop
        while True:
            data, addr = data_socket.recvfrom(self.buff_size)
            data = data[10:]
            np_data.extend(np.frombuffer(data, dtype=dt))
            # while np_data length exceeds frame length, do following
            if len(np_data) >= self.frame_length:
                count_frame += 1
                print("count_frame.", count_frame)
                # put one frame data into bin data array
                self.bin_data.put(np_data[0:self.frame_length])
                # remove one frame length data from array
                np_data = np_data[self.frame_length:]

class DataProcessor(th.Thread):
    def __init__(self, name, config, bin_queue, pic_queue):
        """
        :param name: str
                        Object name

        :param config: sequence of ints
                        Radar config in the order
                        [0]: samples number
                        [1]: chirps number
                        [3]: transmit antenna number
                        [4]: receive antenna number

        :param bin_queue: queue object
                        A queue for access data received by UdpListener

        :param rdi_queue: queue object
                        A queue for store RDI

        :param rai_queue: queue object
                        A queue for store RDI

        """
        th.Thread.__init__(self, name=name)
        self.numADCSamples = config.numADCSamples
        self.numChirps = config.numChirps
        self.numTX = config.numTX
        self.numRX = config.numRX
        self.numFrames = 1
        self.bin_queue = bin_queue
        self.pic_queue = pic_queue

        '''毫米波雷达参数'''
        # 每个chirp采样率(Sample Rate)
        self.fs = config.fs
        # Frequency Slope = B/T,调频斜率
        self.K = config.K

        '''基本参数'''
        # 电磁波传播速度
        self.c = 3 * (10 ** 8)
        # 绘图序号
        self.figure_num = 1

        # 距离FFT，得到距离-时间图像
        self.f = self.fs / self.numADCSamples * np.array(range(1, self.numADCSamples + 1))
        self.t0 = self.f / self.K
        self.r = self.c * self.t0 / 2

    def run(self):
        frame_count = 0
        while True:
            adcData = self.bin_queue.get() #长度等于frame_length
            # 处理一帧 不能使用这个代码，因为是针对14XX写的
            # print('frame_count.',frame_count)
            # data = np.reshape(data, [-1, 4])
            # data = data[:, 0:2:] + 1j * data[:, 2::]
            # data = np.reshape(data, [self.chirp_num, -1, self.adc_sample])
            # data = data.transpose([0, 2, 1])

            # 重构成Matlab格式的数据
            retVal = RefactorADC(adcData=adcData,adc_sample=self.numADCSamples)
            frame_count += 1
            # print(frame_count)
            if frame_count % 3 == 1:
                self.pic_queue.put('101')
            else:
                self.pic_queue.put('010')

            # print(retVal[0])
            # # 对重构之后的数据进行处理
            # dataRX = retVal[0] #取通道0的数据
            # ## 距离FFT，得到距离-时间图像
            # fft1D = Hrrp(dataRX,self.numADCSamples,self.numChirps,self.numFrames)
            # ## 杂波去除
            # dataFFT1D_complex = NoiseRemove(fft1D,self.numADCSamples,self.numChirps,self.numFrames)
            # ## 进行信号积累
            # fft_jilei = SignalAdd(dataFFT1D_complex,self.numADCSamples,self.numChirps,self.numFrames)
            # ## 提取相位信息
            # angle_fft_last1 = PhaseUnwrapping(fft_jilei,fft1D,self.numADCSamples,self.numChirps,self.numFrames)
            #
            # ## 分离呼吸心跳信号
            # breath_fre, heart_fre = Separate_HB(self.heart_a,self.heart_b,self.breath_a,self.breath_b,angle_fft_last1,self.numADCSamples,self.numChirps,self.numFrames)
            # print('breath_fre:',breath_fre.shape ,breath_fre)
            # print('heart_fre:',heart_fre.shape ,heart_fre)
            #
            # ##计算最后的呼吸心跳
            # pos_brea = np.argmax(breath_fre)
            # pos_hear = np.argmax(heart_fre)
            # b_final = -(pos_brea / len(breath_fre) * self.fs_b) +(self.fs_b/2)
            # h_final = -(pos_hear / len(heart_fre) * self.fs_h) +(self.fs_h/2)
            # print('b_final',b_final )
            # print('h_final', h_final)


            # rdi = DSP.Range_Doppler(data, mode=1, padding_size=[128, 64])
            # rai = DSP.Range_Angle(data, mode=1, padding_size=[128, 64, 32])
            # self.rdi_queue.put(rdi)
            # self.rai_queue.put(rai)

