from hand import Ui_MainWindow
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow


def setColor(color,ui_components):
    color_list = color
    for index,ele in enumerate(color_list):
        if ele == "0":
            ui_components.label_list[index].setPixmap(ui_components.spacePicture)

        if ele == "1":
            ui_components.label_list[index].setPixmap(ui_components.handPicture)

if __name__ == '__main__':

    color = "001"

    # application 对象
    app = QApplication(sys.argv)

    # QMainWindow对象
    mainwindow = QMainWindow()

    # 这是qt designer实现的Ui_MainWindow类
    ui_components = Ui_MainWindow()
    # 调用setupUi()方法，注册到QMainWindwo对象
    ui_components.setupUi(mainwindow)

    #print(ui_components.label_list)
    setColor(color,ui_components)


    # 显示
    mainwindow.show()

    sys.exit(app.exec_())
