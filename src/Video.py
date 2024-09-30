import cv2
from Image import Image
class Video:
    def __init__(self, vid_path):
        self.vid_path = vid_path
        self.vid = cv2.VideoCapture(self.vid_path)
        self.frames = None

    def __del__(self):
        self.vid.release()

    def getFps(self):
        return self.vid.get(cv2.CAP_PROP_FPS)

    def getFrameCount(self):
        return int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))

    def getFrame(self, i, crop=True):
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = self.vid.read()
        assert ret == True, f'Read frame -- {i} -- failed, number of video frames: {self.getFrameCount()}'
        img = Image(frame)
        if crop:
            img.crop()
        return img

    def setFrames(self, crop=True):
        n = self.getFrameCount()
        self.frames = [self.getFrame(i) for i in range(n)]
        if crop:
            for frame in self.frames:
                frame.crop()