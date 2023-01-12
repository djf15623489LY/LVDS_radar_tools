import threading as th
import numpy as np
import socket

import math

class UdpListener(th.Thread):
    def __init__(self, name, bin_data, data_frame_length, data_address, buff_size, mode, output_file):
        """
        :param name: str
                        Object name

        :param bin_data: queue object
                        A queue used to store adc data from udp stream

        :param data_frame_length: int
                        Length of a single frame

        :param data_address: (str, int) 接受数据端口
                        Address for binding udp stream, str for host IP address, int for host data port

        :param buff_size: int
                        Socket buffer size
        """
        th.Thread.__init__(self, name=name)
        self.bin_data = bin_data
        self.frame_length = data_frame_length
        self.data_address = data_address
        self.buff_size = buff_size

        self.mode = mode
        self.oldData = bytes()
        self.output_file = output_file
        self.output_file_num = 0

    def run(self):
        # convert bytes to data type int16
        dt = np.dtype(np.int16)
        dt = dt.newbyteorder('<')
        # array for putting raw data
        np_data = []
        temp = bytes()
        temp = b''
        self.oldData = b''
        # count frame
        count_frame = 0
        data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data_socket.bind(self.data_address)
        print("Create socket successfully")
        print("Now start data streaming")
        # main loop
        while True:
            data, addr = data_socket.recvfrom(self.buff_size)
            # 每个data 的长度 1466 ，buffsize是最大的长度，
            #去掉 Sequence number 和 Byte count
            # sNum = int.from_bytes(data[0:4], byteorder="little")
            # rn = int.from_bytes(data[4:10], byteorder="little")
            # print('rn',rn)
            # print('dk',(-rn) % (4 * 256))
            # NC = math.ceil(2 * 4 * 256 * 4 / (1456))
            # print('NC',NC)
            #
            # sLen = len(data)
            # sData = np.zeros((sLen - 10) // 2)
            # for m,k in zip(range(10, sLen, 2), range(0,sLen-10)):
            #     binByte = int.from_bytes(data[m:m+2], byteorder="little", signed=True)
            #     sData[k] = binByte
            # print(sData[0:5])

            data = data[10:]
            temp += data
            #因为基于int16读法会使其长度减小一半，字节流变成点
            # np_data_2 = []
            # np_data_2.extend(np.frombuffer(data, dtype=dt))
            # print(np_data_2[0:5])

            np_data.extend(np.frombuffer(data, dtype=dt))
            # while np_data length exceeds frame length, do following
            if len(np_data) >= self.frame_length:
                count_frame += 1
                # print("Frame No.", count_frame)
                # put one frame data into bin data array
                self.bin_data.put(np_data[0:self.frame_length])
                self.oldData += temp[0:self.frame_length*2]
                # remove one frame length data from array
                np_data = np_data[self.frame_length:]
                temp = temp[self.frame_length*2:]
                print('count_frame:',count_frame)

                #每1000帧保存一下
                if self.mode[0] == 'frame' and count_frame % 1000 == 0:
                    fileName = self.output_file.split('.')
                    bfile = open(fileName[0]+ '_'+ str(self.output_file_num) + '.' + fileName[1], 'wb')
                    self.output_file_num = self.output_file_num + 1
                    bfile.write(self.oldData)
                    self.oldData = b''
                    bfile.close()

                #达到预定的帧数保存一下
                if self.mode[0] == 'frame' and count_frame == self.mode[1] :
                    fileName = self.output_file.split('.')
                    bfile = open(fileName[0]+ '_'+ str(self.output_file_num) + '.' + fileName[1], 'wb')
                    bfile.write(self.oldData)
                    bfile.close()
                    break





