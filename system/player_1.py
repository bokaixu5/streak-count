from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import QFileDialog, QApplication
from PyQt5 import uic

class videoPlayer:
    def __init__(self):
        self.ui = uic.loadUi('video_1.ui')  # 加载designer设计的ui程序
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.ui.wgt_player)
        self.ui.btn_select.clicked.connect(self.openVideoFile)
    # 打开视频文件并播放
    def openVideoFile(self):
        self.player.setMedia(QMediaContent(QFileDialog.getOpenFileUrl()[0]))
        self.player.play()


if __name__ == "__main__":
    app = QApplication([])
    myPlayer = videoPlayer()
    myPlayer.ui.show()
    app.exec()
