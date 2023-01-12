from real_time_process import UdpListener
from radar_config import SerialConfig
from queue import Queue
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import numpy as np
import threading
import time
import sys
import socket


#打开雷达，传输配置文件
def openradar(uart):
    global radar_ctrl
    #波特率为115200
    radar_ctrl = SerialConfig(name='ConnectRadar', CLIPort=uart, BaudRate=115200)
    radar_ctrl.StopRadar()
    radar_ctrl.SendConfig(config)

#传送FPGA命令
def send_cmd(code):
    # command code list
    CODE_1 = (0x01).to_bytes(2, byteorder='little', signed=False)
    CODE_2 = (0x02).to_bytes(2, byteorder='little', signed=False)
    CODE_3 = (0x03).to_bytes(2, byteorder='little', signed=False)
    CODE_4 = (0x04).to_bytes(2, byteorder='little', signed=False)
    CODE_5 = (0x05).to_bytes(2, byteorder='little', signed=False)
    CODE_6 = (0x06).to_bytes(2, byteorder='little', signed=False)
    CODE_7 = (0x07).to_bytes(2, byteorder='little', signed=False)
    CODE_8 = (0x08).to_bytes(2, byteorder='little', signed=False)
    CODE_9 = (0x09).to_bytes(2, byteorder='little', signed=False)
    CODE_A = (0x0A).to_bytes(2, byteorder='little', signed=False)
    CODE_B = (0x0B).to_bytes(2, byteorder='little', signed=False)
    CODE_C = (0x0C).to_bytes(2, byteorder='little', signed=False)
    CODE_D = (0x0D).to_bytes(2, byteorder='little', signed=False)
    CODE_E = (0x0E).to_bytes(2, byteorder='little', signed=False)

    # packet header & footer
    header = (0xA55A).to_bytes(2, byteorder='little', signed=False)
    footer = (0xEEAA).to_bytes(2, byteorder='little', signed=False)

    # data size
    dataSize_0 = (0x00).to_bytes(2, byteorder='little', signed=False)
    dataSize_6 = (0x06).to_bytes(2, byteorder='little', signed=False)

    # data
    data_FPGA_config = (0x01020102031e).to_bytes(6, byteorder='big', signed=False)
    data_packet_config = (0xc005350c0000).to_bytes(6, byteorder='big', signed=False)

    # connect to DCA1000
    connect_to_FPGA = header + CODE_9 + dataSize_0 + footer
    read_FPGA_version = header + CODE_E + dataSize_0 + footer
    config_FPGA = header + CODE_3 + dataSize_6 + data_FPGA_config + footer
    config_packet = header + CODE_B + dataSize_6 + data_packet_config + footer
    start_record = header + CODE_5 + dataSize_0 + footer
    stop_record = header + CODE_6 + dataSize_0 + footer

    if code == '9':
        re = connect_to_FPGA
    elif code == 'E':
        re = read_FPGA_version
    elif code == '3':
        re = config_FPGA
    elif code == 'B':
        re = config_packet
    elif code == '5':
        re = start_record
    elif code == '6':
        re = stop_record
    else:
        re = 'NULL'
    print('send command:', re.hex())
    return re

class Config:
    def __init__(self,numADCSamples,numChirps,numRX):

        self.numADCSamples = numADCSamples
        self.numChirps = numChirps
        self.numRX = numRX
def readCofig(fliename):
    cfg_file = open(fliename, 'r')
    cfg = cfg_file.readlines()
    for line in cfg:
        args = line.split()
        if (len(args) > 0):
            if (args[0] == 'profileCfg'):
                numADCSamples = int(args[10])

            elif (args[0] == 'frameCfg'):
                numChirps = int(args[3])

            elif (args[0] == 'channelCfg'):
                numRX = 4

    return Config(numADCSamples,numChirps,numRX)

# 定义一些队列用以保存结果
BinData = Queue()
#主要配置文件：
###################################################################################################
# 定义配置文件和用于通讯的串口号
mode = ['frame',100]# 'frame'帧停模式,第二个参数为多少帧 , 'time'时间停模式，第二个参数为多少秒
output_file = 'python_adc_data_2.bin'
config = '../config/iwr6843_fps20.cfg'
uart = 'COM8'
###################################################################################################

#毫米波雷达参数
cfg = readCofig(config)
# ADC Samples
adc_sample = cfg.numADCSamples
# No.of Chrip Loops
chirp = cfg.numChirps
# 发送天线
tx_num = 1
# 接受天线
rx_num = cfg.numRX
#每帧长公式(按点算)
frame_length = adc_sample * chirp * rx_num * 2
print('frame_length:',frame_length)

# 网口配置
address = ('192.168.33.30', 4098)
buff_size = 524288 #缓冲区
# config DCA1000 to receive bin data
config_address = ('192.168.33.30', 4096)
FPGA_address_cfg = ('192.168.33.180', 4096)
cmd_order = ['9', 'E', '3', 'B', '5', '6']
sockConfig = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sockConfig.bind(config_address)


# 通过网口对FPGA进行配置
for k in range(5):
    # Send the command
    sockConfig.sendto(send_cmd(cmd_order[k]), FPGA_address_cfg)
    time.sleep(0.1)
    # Request data back on the config port
    msg, server = sockConfig.recvfrom(2048)
    print('receive command:', msg.hex())



#建立线程监听网口
collector = UdpListener('Listener', BinData, frame_length, address, buff_size, mode , output_file)

#通过串口发送配置文件到雷达
openradar(uart)

#开始启动
collector.start()

if mode[0] == 'frame':
    # 等待采集线程结束
    collector.join()
elif mode[0] == 'time':
    time.sleep(mode[1])

#通过网口关闭雷达
sockConfig.sendto(send_cmd('6'), FPGA_address_cfg)
sockConfig.close()
print("Main func close")