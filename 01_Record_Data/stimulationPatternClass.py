import numpy as np
import matplotlib.pyplot as plt

class stimulationPatternClass:
    def __init__(self, ID = '', amp = 0, numb = 0):
        self.chipWidth = 220
        self.chipHeight = 120
        self.chipID = ID

        self.xCoordinate = numb % self.chipWidth
        self.yCoordinate = int(numb / self.chipWidth)
        self.chessboard = 0

        self.electrodeNumbers = [self.getElectrodeNumber()]
        self.amplitude = amp

        #In Pulse durations
        self.interPeakInterval = 0
        self.offset = 0

    def setAmplitude(self,amplitude):
        self.amplitude = float(amplitude)

    def setOffset(self,offset):
        self.offset = int(offset+0.5)

    def setInterPulseInterval(self,interPeakInterval):
        self.interPeakInterval = int(interPeakInterval+0.5)

    def largestElectrodeNumber(self):
        return max(self.electrodeNumbers)

    def appendElectrodeNumber(self,electrodes):
        for newNumber in electrodes:
            if newNumber not in self.electrodeNumbers:
                self.electrodeNumbers.append(int(newNumber))

    def clearDuplicateElectrodes(self):
        electrodes = self.electrodeNumbers
        self.electrodeNumbers = []
        for newNumber in electrodes:
            if newNumber not in self.electrodeNumbers:
                self.electrodeNumbers.append(int(newNumber))

    def copyPattern(self, stimulationPattern):
        self.amplitude = stimulationPattern.amplitude
        self.interPeakInterval = stimulationPattern.interPeakInterval
        self.offset = stimulationPattern.offset

    def selectCoordinate(self,pastElectrode,voltageMap, centerElectrodeAxisX, centerElectrodeAxisY, patchWidth, patchHeight):
        if centerElectrodeAxisX < self.chipWidth and centerElectrodeAxisY < self.chipHeight:
            topLeftX = max(0,centerElectrodeAxisX - int(patchWidth / 2))
            topLeftY = max(0, centerElectrodeAxisY - int(patchHeight / 2))
        else:
            topLeftX = 0
            topLeftY = 0

        def selectCoordinates(event):
            ix, iy = event.xdata, event.ydata
            self.xCoordinate = int(ix) + topLeftX
            self.yCoordinate = int(iy) + topLeftY
            newNumber = int(self.getElectrodeNumber())
            if newNumber not in self.electrodeNumbers and pastElectrode in self.electrodeNumbers:
                self.electrodeNumbers[self.electrodeNumbers.index(pastElectrode)] = newNumber
            fig.canvas.mpl_disconnect(cid)
            plt.close()
            return
        self.calculateChessboard(voltageMap,topLeftX,topLeftY)
        fig, ax = plt.subplots()
        ax.matshow(2*voltageMap + self.chessboard, cmap='magma')
        cid = fig.canvas.mpl_connect('button_press_event', selectCoordinates)
        plt.show()

    def getElectrodeNumber(self):
        return self.xCoordinate + self.yCoordinate * self.chipWidth

    def getXYCoordinates(self, electrodeNumber):
        return electrodeNumber % self.chipWidth, int(electrodeNumber / self.chipWidth)



    def equal(self, stimulationPattern):
        if self.amplitude != stimulationPattern.amplitude:
            return False
        if self.interPeakInterval != stimulationPattern.interPeakInterval:
            return False
        if self.offset != stimulationPattern.offset:
            return False
        return True

    def calculateChessboard(self, voltageMap, topLeftX, topLeftY):
        self.chessboard = np.indices(voltageMap.shape).sum(axis=0) % 2

        for i in self.electrodeNumbers:
            xCoordinate, yCoordinate = self.getXYCoordinates(i)
            if (voltageMap.shape[1] > xCoordinate - topLeftX >= 0) and (voltageMap.shape[0] > yCoordinate - topLeftY >= 0):
                self.chessboard[yCoordinate - topLeftY, xCoordinate - topLeftX] = 0
                self.chessboard[yCoordinate - topLeftY, xCoordinate - topLeftX] = 10

    def isfloat(self, value):
        if value == '':
            return False
        try:
            float(value)
            return True
        except ValueError:
            return False
