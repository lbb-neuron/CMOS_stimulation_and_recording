from xml.etree.ElementInclude import include
import numpy as np
import h5py
import os
import sys
import time
import ipywidgets as wdg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import is_color_like
import copy
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from scipy.signal import butter, lfilter
from scipy.signal import find_peaks as fp
from scipy.signal import convolve
from scipy.ndimage import convolve1d
import scipy.signal as sig
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

class AnalyZor:
    def __init__(
        self,
        filename,
        auto_parse=False,
        inputPath=os.path.join(sys.path[0], "input"),
        outputPath=os.path.join(sys.path[0], "output"),
        chipHeight=120,
        chipWidth=220,
        sampleFrequency=20000,
        yearIndex=0,
        monthIndex=1,
        dayIndex=2,
        DIVIndex=3,
        customTextIndex=4,
        chipIDIndex=5,
        networkNrIndex=7,
        indexRange=8,
        microVoltPerBit=6.3,
        stimulation_electrodes=[
            0,
        ],
        stimulation_voltages=[
            0,
        ],
    ):
        """
        The AnalyZor Class loads h5 data produced by the Software for the high resolution MEAs of Maxwell Bio Systems.
        It is used to generate and save plots of different kinds.

        :param filename:
                The name of the h5 file.
                The filename is split by either '_' or '.'.
        :param auto_parse: Whether to get information (DIV, stimulation electrode,...) from the filename and txt-file (recorder.py compatible).
                For auto_parse, all files need to be placed in a folder called "input"
        :param inputPath: The directory where the file is stored.
        :param outputPath: The directory where the folder of the generated plots will be stored.
        :param chipHeight: The number of electrodes on the y-axis of the chip.
        :param chipWidth: The number of electrodes on the x-axis of the chip.
        :param indexRange: How many Indices there are in total.
        :param sampleFrequency: The sample frequency of the System in Hz.
        :param yearIndex: The index value which indicates where the year is found.
        :param monthIndex: The index value which indicates where the month is found.
        :param dayIndex: The index value which indicates where the day is found.
        :param DIVIndex: The index value which indicates where day in vitro is found.
        :param customTextIndex: The index value which indicates where a custom text is found.
        :param chipIDIndex: The index value which indicates where chip ID is found.
        :param networkNrIndex: The index value which indicates where network number is found.
        :param microVoltPerBit: How much volt one bit corresponds to if it is amplified by 512.
        :param stimulation_electrodes: In case auto_parse = False, stimulation electrodes are given here
        :param stimulation_voltages: The voltage(s) at which the stimulation occured [mV]
        """

        self.inputPath = inputPath
        self.outputPath = outputPath
        self.chipHeight = chipHeight
        self.chipWidth = chipWidth

        self.yearIndex = yearIndex
        self.monthIndex = monthIndex
        self.dayIndex = dayIndex
        self.DIVIndex = DIVIndex
        self.customTextIndex = customTextIndex  # No _ or . in this please
        self.chipIDIndex = chipIDIndex
        self.networkNrIndex = networkNrIndex
        self.indexRange = indexRange
        self.filename = filename
        self.sampleFrequency = sampleFrequency
        self.microVoltPerBit = microVoltPerBit

        self.gain = None
        self.electrodeChannelMapping = None

        self.filenameList = []
        self.spikes = []
        self.blankingEnd = np.empty(shape=(0))

        self.experimentDuration = None
        self.experimentDurationBlanked = None
        self.stimulatedElectrodesList = []
        self.stimulationVoltagesList = stimulation_voltages
        self.auto_parse = auto_parse

        if auto_parse:
            self.outputFolder = outputPath
            try:
                os.makedirs(self.outputFolder)
            except:
                pass
            stimulationTXTString = filename[: len(filename) - 7] + ".txt"
            try:
                f = open(os.path.join(inputPath, stimulationTXTString), "r")
                x = f.readline()
                for x in f:
                    number = ""
                    i = 0
                    while x[i] != ",":
                        number += str(x[i])
                        i += 1
                    self.stimulatedElectrodesList.append(int(number))
            except:
                pass
        else:
            self.outputFolder = os.path.join(outputPath, "Results")
            try:
                os.makedirs(self.outputFolder)
            except:
                pass
            self.stimulatedElectrodesList = stimulation_electrodes

    def loadData(
        self,
        blankingBool=False,
        blankBlockBool=False,
        noStimDuration=3000,
        blankingThresh=80,
        blankingWindow=[-10, 10],
        cutOffFrequency=200,
        filterOrder=2,
        spikeThreshold=5,
        spikeDistance=100,
        window=[-75, 75],
        spikeTraceBool=False,
        scaleTraceMap=True,
        traceMapcolorCoding="lin",
        traceMapFilename="trace.png",
        loadingSteps=1,
        use_sneo=False, 
        returnSpikeTrace=False, 
    ):
        """
        Loads all the traces of the h5 file and generates the spike list with custom filtering. Can also save the average spike map.

        :param blankingBool: If true, replaces all frames on which an artefact was detected with 0 on all traces.
        :param blankBlockBool: If true, replaces all frames on which an artefact was detected with 0 on all traces.
                                If the artefacts are close together (this is defined by noStimDuration), all frames between the
                                artefacts are also set to 0.
        :param noStimDuration: How many frames of no artefacts it takes in order to not blank in between.
        :param blankingThresh: How large the signal must be in order to be an artefact.
        :param blankingWindow: How many frames before and after the spike are blanked. [-m,n] with n,m non negative integers.
        :param cutOffFrequency: Low frequencies up to this are filtered out by a highpass filter.
        :param filterOrder: Order of the butterworth filter used.
        :param spikeThreshold: How many times the standard deviation of the signal is required in order for a peak to be recognized as a spike.
        :param spikeDistance: When a spike occurs on a channel, for how many frames other high signal values are not mistaken for a spike.
        :param window: How many frames before and after the peak still belong to the spike. [-m,n] with n,m non negative integers.
        :param spikeTraceBool: If true, saves the average spike map.
        :param scaleTraceMap: If true, averages spikes are scaled to fit the window.
        :param traceMapcolorCoding: 'log','lin' or 'off' for the color coding of the number of peaks of a channel.
        :param traceMapFilename: As what the average spike map should be saved.
        :param loadingSteps: The loading of the traces is split in n steps. For larger n less RAM is used, but calculations take longer.
        :return: Saves all spikes to self.spikes. Latter is a list containing the spike times and meta information about the peaks on the channel.
                If specified, also saves the average spike map.
        """

        # Check if auto_parse is on
        if self.auto_parse:
            rawData = h5py.File(os.path.join(self.inputPath, self.filename), "r")
        else:
            try:
                rawData = h5py.File(os.path.join(self.inputPath, self.filename), "r")            
            except:                
                rawData = h5py.File(self.filename, "r")    

        self.gain = np.asarray(rawData["settings"]["gain"])[0]
        self.microVoltPerBit = 512 / self.gain * self.microVoltPerBit
        electrodeInfo = np.asarray(rawData["mapping"]["channel", "electrode"])
        mask = [i["electrode"] != -1 for i in electrodeInfo]
        clean_abs_inds = np.asarray(
            [i[0]["electrode"][i[1]] for i in zip(electrodeInfo, mask)], dtype=np.int32
        )
        clean_rel_inds = np.asarray(
            [i[0]["channel"][i[1]] for i in zip(electrodeInfo, mask)], dtype=np.int32
        )
        self.electrodeChannelMapping = np.zeros(
            [2, clean_rel_inds.shape[0]], dtype=np.int32
        )
        self.electrodeChannelMapping[0, :] = np.squeeze(clean_abs_inds)
        self.electrodeChannelMapping[1, :] = np.squeeze(clean_rel_inds)
        del electrodeInfo
        del mask
        del clean_abs_inds
        del clean_rel_inds

        firstTime = True
        cutOffDiscrete = cutOffFrequency / self.sampleFrequency * 2
        coeffB, coeffA = butter(filterOrder, cutOffDiscrete, btype="highpass")
        blankingIndices = []
        beginStim = None
        endStim = None
        spikeTraceList = []

        if blankingBool:
            for electrode in self.stimulatedElectrodesList:
                channel = self.electrodeChannelMapping[
                    1, np.argwhere(self.electrodeChannelMapping[0, :] == electrode)
                ]
                traces = rawData.get("sig")[np.squeeze(channel), :]
                traces = lfilter(coeffB, coeffA, traces)

                if firstTime:
                    self.experimentDuration = len(traces)
                    self.experimentDurationBlanked = self.experimentDuration
                    firstTime = False

                artefacts = (
                    fp(np.abs(traces), height=blankingThresh, distance=spikeDistance)
                )[0]
                blankingIndices.extend(artefacts)
                del artefacts
                del traces

            blankingIndices.sort()
            blankingIndices = np.squeeze(np.asarray(blankingIndices))
            if blankBlockBool:
                differences = np.diff(blankingIndices)
                endTemp = (
                    blankingIndices[np.argwhere(differences > noStimDuration)]
                    + blankingWindow[1]
                )
                endTemp[np.argwhere(endTemp >= self.experimentDuration)] = (
                    self.experimentDuration - 1
                )
                beginTemp = (
                    blankingIndices[np.argwhere(differences > noStimDuration) + 1]
                    + blankingWindow[0]
                )
                beginTemp[np.argwhere(beginTemp < 0)] = 0
                endStim = np.zeros(endTemp.shape[0] + 1, dtype=np.int32)
                beginStim = np.zeros(beginTemp.shape[0] + 1, dtype=np.int32)
                beginStim[0] = max(blankingIndices[0] + blankingWindow[0], 0)
                endStim[-1] = min(
                    blankingIndices[-1] + blankingWindow[1], self.experimentDuration
                )
                if endStim.shape[0] > 1:
                    endStim[0 : endStim.shape[0] - 1] = np.squeeze(endTemp)
                if beginStim.shape[0] > 1:
                    beginStim[1:] = np.squeeze(beginTemp)
                del endTemp
                del beginTemp

        if endStim is not None:
            self.blankingEnd = endStim
        elif len(blankingIndices) != 0:
            self.blankingEnd = np.asarray(blankingIndices) + window[1]

        loadingIndicesStart = [0]
        loadingIndicesEnd = []

        for i in range(1, loadingSteps - 1):
            step = int(i / loadingSteps * self.electrodeChannelMapping.shape[1])
            loadingIndicesEnd.append(step)
            loadingIndicesStart.append(step)
        loadingIndicesEnd.append(self.electrodeChannelMapping.shape[1])

        for j in range(len(loadingIndicesStart)):
            channels = self.electrodeChannelMapping[
                1, np.arange(loadingIndicesStart[j], loadingIndicesEnd[j])
            ]
            traces = rawData.get("sig")[np.squeeze(channels), :]

            if traces.ndim == 1:
                traces = np.expand_dims(traces, axis=0)

            if firstTime:
                self.experimentDuration = traces.shape[1]
                self.experimentDurationBlanked = self.experimentDuration
                firstTime = False

            traces = lfilter(coeffB, coeffA, traces)

            if use_sneo:
                traces = convolve1d(traces, sig.savgol_coeffs(5, 2), axis=1)
                K_SNEO = 4
                BW_LEN = 4*K_SNEO+1
                BW = sig.triang(BW_LEN)

                kneo = np.square(traces[:, K_SNEO:-K_SNEO]) - traces[:, 0:-2*K_SNEO]*traces[:, 2*K_SNEO:]
                kneo = np.hstack((np.zeros((kneo.shape[0], K_SNEO)), kneo, np.zeros((kneo.shape[0], K_SNEO))))
                sneo = convolve1d(kneo, BW)
                del kneo

            if beginStim is not None:
                for i in range(beginStim.shape[0]):
                    traces[:, np.arange(beginStim[i], endStim[i])] = 0
                if self.experimentDurationBlanked == self.experimentDuration:
                    self.experimentDurationBlanked -= np.sum(
                        np.squeeze(endStim - beginStim)
                    )
            else:
                durationBlanked = 0
                for i in blankingIndices:
                    traces[
                        :,
                        max(0, i + blankingWindow[0]) : min(
                            traces.shape[1], i + blankingWindow[1]
                        ),
                    ] = 0
                    durationBlanked += min(
                        traces.shape[1], i + blankingWindow[1]
                    ) - max(0, i + blankingWindow[0])
                if self.experimentDurationBlanked == self.experimentDuration:
                    self.experimentDurationBlanked -= durationBlanked

            def findPeaks(i):
                peaks = fp(
                    np.abs((traces)[i, :]),
                    height=spikeThreshold * np.std((traces)[i, :]),
                    distance=spikeDistance,
                )
                peaks[1]["peak_heights"] = np.asarray((traces)[i, :][peaks[0]]) 
                peaks[1]["nr_of_peaks"] = (peaks[0]).shape[0]
                peaks[1]["electrode"] = self.electrodeChannelMapping[
                    0,
                    np.squeeze(
                        np.argwhere(channels[i] == self.electrodeChannelMapping[1, :])
                    ),
                ]
                return peaks

            def SNEOPeaks(i):
                peaks = fp(
                    sneo[i, :],
                    height=spikeThreshold * np.std((sneo)[i, :]),
                    distance=spikeDistance,
                )
                peaks[1]["peak_heights"] = np.asarray((traces)[i, :][peaks[0]])
                peaks[1]["nr_of_peaks"] = (peaks[0]).shape[0]
                peaks[1]["electrode"] = self.electrodeChannelMapping[
                    0,
                    np.squeeze(
                        np.argwhere(channels[i] == self.electrodeChannelMapping[1, :])
                    ),
                ]
                return peaks                

            if use_sneo:
                self.spikes.extend(list(map(SNEOPeaks, range(traces.shape[0]))))
                fig_sneo, ax_sneo = plt.subplots(10, figsize=(12,10))
                sp_select = 0
            else:
                self.spikes.extend(list(map(findPeaks, range(traces.shape[0]))))

            if spikeTraceBool:
                for i in range(traces.shape[0]):
                    batchIndices = (
                        np.asarray(self.spikes[i + loadingIndicesStart[j]][0])
                        + window[0]
                    )
                    batchIndices = batchIndices[batchIndices >= 0]
                    batchIndices = batchIndices[
                        batchIndices < self.experimentDuration - window[1] + window[0]
                    ]
                    if batchIndices.any():
                        actionPotentials = np.asarray(traces[i])[
                            batchIndices[:, None] + np.arange(window[1] - window[0])
                        ]
                    else:
                        actionPotentials = np.zeros(window[1] - window[0])

                    actionPotentials = actionPotentials.reshape(
                        [-1, window[1] - window[0]]
                    )
                    spikeTraceList.append(np.mean(actionPotentials, axis=0))

        if spikeTraceList:
            boundX = [
                min(self.electrodeChannelMapping[0, :] % self.chipWidth),
                max(self.electrodeChannelMapping[0, :] % self.chipWidth) + 1,
            ]
            boundY = [
                int(min(self.electrodeChannelMapping[0, :] / self.chipWidth)),
                int(max(self.electrodeChannelMapping[0, :] / self.chipWidth)) + 1,
            ]
            columns = np.arange(boundX[0], boundX[1])
            rows = np.arange(boundY[0], boundY[1])

            spikeTraceMap = np.zeros(
                [rows.shape[0], columns.shape[0], window[1] - window[0]]
            )
            spike_count = np.zeros([rows.shape[0], columns.shape[0]])
            max_sig = 0
            min_sig = 0
            for i in range(self.electrodeChannelMapping.shape[1]):
                indexX = self.electrodeChannelMapping[0, i] % self.chipWidth - boundX[0]
                indexY = int(
                    self.electrodeChannelMapping[0, i] / self.chipWidth - boundY[0]
                )
                spike_count[indexY, indexX] = self.spikes[i][1]["nr_of_peaks"]
                spikeTraceMap[indexY, indexX, :] = spikeTraceList[i]

                if use_sneo:    
                    colours = [k for k in mcolors.TABLEAU_COLORS.keys()]
                    if spike_count[indexY, indexX] > 200:
                        ax_sneo[sp_select].plot(sneo[i,:], c=colours[i%10])
                        ax_sneo[sp_select].plot(traces[i,:], linestyle='dashed', c=colours[i%10])
                        sp_select += 1
                        sp_select = sp_select%10

                if scaleTraceMap == False:
                    if np.max(spikeTraceList[i]) > max_sig:
                        max_sig = np.max(spikeTraceList[i])
                    if np.min(spikeTraceList[i]) < min_sig:
                        min_sig = np.min(spikeTraceList[i])

            max_sig = max_sig * 1.1
            min_sig = min_sig * 1.1

            fig, ax_array = plt.subplots(
                len(rows), len(columns), squeeze=False, figsize=(23, 23)
            )
            for i, ax_row in enumerate(ax_array):  # rows
                for j, axes in enumerate(ax_row):  # columns
                    # colorcoding background
                    if traceMapcolorCoding == "log":
                        gb_val = 1 - np.log10(1 + spike_count[i, j]) / (
                            np.log10(1 + np.max(spike_count))
                        )
                    elif traceMapcolorCoding == "lin":
                        gb_val = 1 - spike_count[i, j] / np.max(spike_count)
                    elif traceMapcolorCoding == "off":
                        gb_val = 1
                    else:
                        raise ValueError("Colorcoding not valid")

                    axes.set_yticklabels([])
                    axes.set_xticklabels([])
                    axes.plot(spikeTraceMap[i, j, :], "k-")
                    axes.set_facecolor((1, gb_val, gb_val))
                    axes.spines["bottom"].set_color("white")
                    axes.spines["top"].set_color("white")
                    axes.spines["right"].set_color("white")
                    axes.spines["left"].set_color("white")
                    axes.tick_params(axis="x", colors="white")
                    axes.tick_params(axis="y", colors="white")
                    if scaleTraceMap == False:
                        axes.set_ylim(bottom=min_sig, top=max_sig)

            fig.savefig(os.path.join(self.outputFolder, traceMapFilename))
            plt.close()
            del fig
            del spikeTraceList

            if not returnSpikeTrace:
                del spikeTraceMap
            del spike_count

        if use_sneo:
            del sneo
        del traces
        del blankingIndices
        if returnSpikeTrace:
            return spikeTraceMap

    def loadDataSpikesOnly(
        self,
        blankingBool=False,
        blankBlockBool=False,
        noStimDuration=3000,
        blankingThresh=50,
        blankingWindow=[-10, 10],
    ):
        """
        Loads all the spike times of the h5 file and generates the spike list.

        :param blankingBool: If true, removes all spikes occuring around a detected artefact.
        :param blankBlockBool: If true, removes all spikes occuring around a detected artefact.
                                If the artefacts are close together (this is defined by noStimDuration), all spikes between the
                                artefacts are also removed.
        :param noStimDuration: How many frames of no artefacts it takes in order to not blank in between.
        :param blankingThresh: How large the signal must be in order to be an artefact.
        :param blankingWindow: How many frames before and after the spike are blanked. [-m,n] with non negative n,m integers.
        :return: Saves all spikes to self.spikes. Latter is a list containing the spike times and meta information about the peaks on the channel.
        """

        # Check if auto_parse is on
        if self.auto_parse:
            rawData = h5py.File(os.path.join(self.inputPath, self.filename), "r")
        else:
            try:
                rawData = h5py.File(os.path.join(self.inputPath, self.filename), "r")
            except:
                rawData = h5py.File(self.filename, "r")    

        self.gain = np.asarray(rawData["settings"]["gain"])[0]

        electrodeInfo = np.asarray(rawData["mapping"]["channel", "electrode"])
        mask = [i["electrode"] != -1 for i in electrodeInfo]
        clean_abs_inds = np.asarray(
            [i[0]["electrode"][i[1]] for i in zip(electrodeInfo, mask)], dtype=np.int32
        )
        clean_rel_inds = np.asarray(
            [i[0]["channel"][i[1]] for i in zip(electrodeInfo, mask)], dtype=np.int32
        )
        self.electrodeChannelMapping = np.zeros(
            [2, clean_rel_inds.shape[0]], dtype=np.int32
        )
        self.electrodeChannelMapping[0, :] = np.squeeze(clean_abs_inds)
        self.electrodeChannelMapping[1, :] = np.squeeze(clean_rel_inds)
        del electrodeInfo
        del mask
        del clean_abs_inds
        del clean_rel_inds
        amplitudes = np.squeeze((rawData.get("proc0")["spikeTimes"])["amplitude"])
        spikeTimes = np.squeeze((rawData.get("proc0")["spikeTimes"])["frameno"])
        channels = np.squeeze((rawData.get("proc0")["spikeTimes"])["channel"])
        self.experimentDuration = np.max(spikeTimes) - np.min(spikeTimes)
        spikeTimes = spikeTimes - np.min(spikeTimes)
        self.experimentDurationBlanked = self.experimentDuration

        if blankingBool:
            if self.stimulatedElectrodesList:
                stimulatedChannels = np.squeeze(
                    self.electrodeChannelMapping[
                        1,
                        np.squeeze(
                            np.argwhere(
                                np.in1d(
                                    self.electrodeChannelMapping[0, :],
                                    self.stimulatedElectrodesList,
                                )
                            )
                        ),
                    ]
                )
            else:
                raise ValueError("No stimulation electrode was defined.")

            artefactIndices = np.squeeze(
                np.argwhere(
                    np.logical_and(
                        np.abs(amplitudes) > blankingThresh,
                        np.in1d(channels, stimulatedChannels),
                    )
                )
            )
            spikeTimesArtefact = spikeTimes[artefactIndices]
            if blankBlockBool:
                differences = np.diff(spikeTimesArtefact)
                end = (
                    spikeTimesArtefact[np.argwhere(differences > noStimDuration)]
                    + blankingWindow[1]
                )
                begin = (
                    spikeTimesArtefact[np.argwhere(differences > noStimDuration) + 1]
                    + blankingWindow[0]
                )
                end = np.append(
                    np.squeeze(end), spikeTimesArtefact[-1] + blankingWindow[1]
                ).astype(np.int64)
                begin = np.append(
                    spikeTimesArtefact[0] + blankingWindow[0], np.squeeze(begin)
                ).astype(np.int64)
                blankedSpikeTimes = np.zeros(
                    np.sum(np.squeeze(end - begin)), dtype=np.int64
                )
                index = 0
                for i in range(begin.shape[0]):
                    blankedSpikeTimes[index : index + end[i] - begin[i]] = np.arange(
                        begin[i], end[i]
                    )
                    index += end[i] - begin[i]
                self.blankingEnd = end + 1
                # print(f"Blanking End Times: {np.diff(self.blankingEnd)/20e3}")
                self.experimentDurationBlanked -= np.sum(np.squeeze(end - begin))
                del end
                del begin
                del differences

            else:
                blankedSpikeTimes = np.arange(blankingWindow[1] - blankingWindow[0])
                blankedSpikeTimes = np.tile(
                    blankedSpikeTimes, (spikeTimesArtefact.shape[0], 1)
                )
                blankedSpikeTimes = np.add(
                    np.expand_dims(spikeTimesArtefact + blankingWindow[0], axis=1),
                    blankedSpikeTimes,
                )
                self.blankingEnd = np.squeeze(blankedSpikeTimes[:, -1]) + 1
                blankedSpikeTimes = blankedSpikeTimes.flatten()
                self.experimentDurationBlanked = (
                    self.experimentDurationBlanked
                    - np.sum(artefactIndices.shape * (blankingWindow[1] - blankingWindow[0]))
                )
            blankedSpikeTimes = np.sort(blankedSpikeTimes)
            indices = np.squeeze(np.argwhere(~np.in1d(spikeTimes, blankedSpikeTimes)))
            # indicesDebug = np.squeeze(np.argwhere(np.in1d(spikeTimes, blankedSpikeTimes)))
            # amplitudesDebug = amplitudes[indicesDebug]
            # spikeTimesDebug = spikeTimes[indicesDebug]
            # channelsDebug = channels[indicesDebug]
            # differencesDebug = np.diff(spikeTimesDebug)
            # differencesArtefactsDebug = np.diff(spikeTimesArtefact)

            amplitudes = amplitudes[indices]
            spikeTimes = spikeTimes[indices]
            channels = channels[indices]
            del blankedSpikeTimes
            del spikeTimesArtefact
            del artefactIndices
            del indices

        del rawData

        for i in range(self.electrodeChannelMapping.shape[1]):
            channelIndices = np.argwhere(
                channels == np.squeeze(self.electrodeChannelMapping[1, i])
            )
            peaks = np.squeeze(spikeTimes[channelIndices], axis=1), {}
            peaks[1]["peak_heights"] = np.squeeze(amplitudes[channelIndices], axis=1)
            peaks[1]["nr_of_peaks"] = (peaks[0]).shape[0]
            peaks[1]["electrode"] = self.electrodeChannelMapping[0, i]
            self.spikes.append(peaks)

            del peaks
            del channelIndices
        del spikeTimes
        del channels
        del amplitudes

    def frequencyHeatmap(
        self,
        filename="freq",
        figureSize=(20, 15),
        showPlotBool=False,
        colormap="seismic",
        windowBool=False,
        dpi=100,
        format="png",
        storeMap=False, 
    ):
        """
        Generate a heatmap of the spike frequency.

        :param filename: As what the heatmap should be saved.
        :param figureSize: Tuple which specifies the size of the output.
        :param showPlotBool: If True, shows the matplotlib plot. The plot won't be saved!
        :param colormap: Which matplotlib cmap is used for the plot.
        :param windowBool: If False, all electrodes are plotted.
        :param dpi: Pixels per inch for the plot.
        :param format: Type as what the file should be saved.
        :return: Saves a plot of the heatmap.
        """

        if colormap not in plt.colormaps():
            print("colormap does not exist in Matplotlib.")
            colormap = "seismic"

        heatmap = np.zeros(self.chipHeight * self.chipWidth)
        heatmap[np.asarray([d[1]["electrode"] for d in self.spikes])] = np.asarray(
            [d[1]["nr_of_peaks"] for d in self.spikes]
        )
        heatmap = (
            heatmap.reshape([self.chipHeight, self.chipWidth])
            / self.experimentDurationBlanked
            * self.sampleFrequency
        )

        fig = plt.figure(figsize=figureSize, dpi=dpi)
        if windowBool:
            boundX = [
                min(self.electrodeChannelMapping[0, :] % self.chipWidth),
                max(self.electrodeChannelMapping[0, :] % self.chipWidth) + 1,
            ]
            boundY = [
                int(min(self.electrodeChannelMapping[0, :] / self.chipWidth)),
                int(max(self.electrodeChannelMapping[0, :] / self.chipWidth)) + 1,
            ]
            plt.imshow(
                heatmap[boundY[0] : boundY[1], boundX[0] : boundX[1]], cmap=colormap
            )
            heatmap = heatmap[boundY[0] : boundY[1], boundX[0] : boundX[1]]
        else:
            plt.imshow(heatmap, cmap=colormap)
        plt.colorbar()
        plt.title("Average spiking frequency [Hz]")
        plt.xlabel("Electrode x")
        plt.ylabel("Electrode y")
        if showPlotBool:
            plt.show()
        elif storeMap:
                fig.savefig(
                    os.path.join(self.outputFolder, filename + ".{}".format(format)),
                    dpi=fig.dpi,
                    transparent=True,
                    format=format,
                )
        plt.close()
        del fig
        return heatmap

    def maxAmplitudeHeatmap(
        self,
        filename="max",
        figureSize=(6.4, 4.8),
        showPlotBool=False,
        colormap="seismic",
        windowBool=False,
        dpi=100,
        format="png",
        storeMap=False 
    ):
        """
        Generate a heatmap of the maximum amplitude.

        :param filename: As what the heatmap should be saved.
        :param figureSize: Tuple which specifies the size of the output.
        :param showPlotBool: If True, shows the matplotlib plot. The plot won't be saved!
        :param colormap: Which matplotlib cmap is used for the plot.
        :param dpi: Pixels per inch for the plot.
        :param format: Type as what the file should be saved.
        :return: Saves a plot of the heatmap.
        """

        if colormap not in plt.colormaps():
            print("colormap does not exist in Matplotlib.")
            colormap = "seismic"

        heatmap = np.zeros(self.chipHeight * self.chipWidth)
        for d in self.spikes:
            if len(d[1]["peak_heights"]) != 0:
                heatmap[np.asarray(d[1]["electrode"])] = max(
                    d[1]["peak_heights"].min(), d[1]["peak_heights"].max(), key=abs
                )
        heatmap = heatmap.reshape([self.chipHeight, self.chipWidth])

        fig = plt.figure(figsize=figureSize, dpi=dpi)
        if windowBool:
            boundX = [
                min(self.electrodeChannelMapping[0, :] % self.chipWidth),
                max(self.electrodeChannelMapping[0, :] % self.chipWidth) + 1,
            ]
            boundY = [
                int(min(self.electrodeChannelMapping[0, :] / self.chipWidth)),
                int(max(self.electrodeChannelMapping[0, :] / self.chipWidth)) + 1,
            ]
            plt.imshow(
                heatmap[boundY[0] : boundY[1], boundX[0] : boundX[1]], cmap=colormap
            )
            heatmap = heatmap[boundY[0] : boundY[1], boundX[0] : boundX[1]]
        else:
            plt.imshow(heatmap, cmap=colormap)
        plt.imshow(heatmap, cmap=colormap)
        plt.colorbar()
        plt.title("Maximum spiking amplitude detected [a.u.]")
        plt.xlabel("Electrode x")
        plt.ylabel("Electrode y")
        if showPlotBool:
            plt.show()
        elif storeMap:
            fig.savefig(
                os.path.join(self.outputFolder, filename + ".{}".format(format)),
                dpi=fig.dpi,
                transparent=True,
                format=format,
            )
        plt.close()
        del fig
        return heatmap

    def meanFiringRate(
        self,
        filename="mean",
        figureSize=(6.4, 4.8),
        binSize="minute",
        showPlotBool=False,
        dpi=100,
        format="png",
    ):
        """
        Generate a bar plot of the spike frequency across all channels.

        :param filename: As what the bar plot should be saved.
        :param figureSize: Tuple which specifies the size of the output.
        :param binSize: Size of a bin where the spikes are summarized. Can be 'second', 'minute' or 'hour'.
        :param showPlotBool: If True, shows the matplotlib plot. The plot won't be saved!
        :param dpi: Pixels per inch for the plot.
        :param format: Type as what the file should be saved.
        :return: Saves a bar plot.
        """
        binSizeString = ["s", "second"]
        scaleBins = 1
        if binSize == "minute":
            scaleBins = 60
            binSizeString = ["min", "minute"]
        elif binSize == "hour":
            scaleBins = 3600
            binSizeString = ["h", "hour"]

        bins = np.arange(0, self.experimentDuration, scaleBins * self.sampleFrequency)
        binnedData = np.zeros(bins.shape[0] - 1)
        for d in self.spikes:
            binnedDataDict, temp = np.histogram(np.asarray(d[0]), bins)
            binnedData += binnedDataDict
            del temp
            del binnedDataDict
        binnedData = binnedData / self.electrodeChannelMapping.shape[1]
        fig = plt.figure(figsize=figureSize, dpi=dpi)

        plt.xlabel("Time [{}]".format(binSizeString[0]))
        plt.ylabel("No. of spikes per {}".format(binSizeString[1]))

        plt.bar(x=range(binnedData.shape[0]), height=binnedData)
        plt.title("Mean Firing Rate over time [Hz]")

        if showPlotBool:
            plt.show()
        else:
            fig.savefig(
                os.path.join(self.outputFolder, filename + ".{}".format(format)),
                dpi=fig.dpi,
                transparent=True,
                format=format,
            )
        plt.close()
        del fig
        return range(binnedData.shape[0]), binnedData

    def signalPropagationMovie(
        self,
        startFrame,
        durationInFrames,
        filename="Signal_Propagation.mp4",
        decayFrames=100,
        slowDownFactor=100,
        figureSize=(8, 8),
        showPlotBool=False,
        colormap="seismic",
        dpi=100,
        readRawDataBool=False,
        filterOrder=2,
        cutOffFrequency=200,
        positivNegativePeaks=0,
        maskcolor="grey",
    ):
        """
        Generates a movie of a selected timeframe on how the spikes propagate.
        Use self.blankingEnd or self.spikes for good start frames.

        :param startFrame: The first frame of the measurement which is shown in the movie.
        :param durationInFrames: How many frames should be in the video.
        :param filename: As what the movie should be saved.
        :param decayFrames: How long the linear decay lasts after a spike.
        :param slowDownFactor: How many times the footage should be slowed down.
        :param figureSize: Tuple which specifies the size of the output.
        :param showPlotBool: If True, shows the matplotlib animation. The animation won't be saved!
        :param colormap: Which matplotlib cmap is used for the movie.
        :param dpi: Pixels per inch for the plot.
        :param readRawDataBool: If True, reads the h5 file and generates a movie with the raw data.
        :param filterOrder: Order of the butterworth filter used.
        :param cutOffFrequency: Low frequencies up to this are filtered out by a highpass filter.
        :param positivNegativePeaks: If 0, all values are taken. For 1 only positive, for -1 only negative are shown.
        :param maskcolor: color with which the mask is plotted.
        :return: Saves a movie of the propagating spikes without amplitude information.
        """

        if colormap not in plt.colormaps():
            print("colormap does not exist in Matplotlib.")
            colormap = "seismic"
        if not is_color_like(maskcolor):
            print("Mask color does not exist in Matplotlib.")
            maskcolor = "grey"
        colormap = copy.copy(cm.get_cmap(colormap))
        colormap.set_bad(color=maskcolor)

        boundX = [
            min(self.electrodeChannelMapping[0, :] % self.chipWidth),
            max(self.electrodeChannelMapping[0, :] % self.chipWidth) + 1,
        ]
        boundY = [
            int(min(self.electrodeChannelMapping[0, :] / self.chipWidth)),
            int(max(self.electrodeChannelMapping[0, :] / self.chipWidth)) + 1,
        ]
        columns = np.arange(boundX[0], boundX[1])
        rows = np.arange(boundY[0], boundY[1])
        endFrame = durationInFrames + startFrame
        movieMatrix = np.zeros([durationInFrames, rows.shape[0], columns.shape[0]])
        movieMatrix[:] = np.nan

        if readRawDataBool:
            coeffB, coeffA = butter(
                filterOrder,
                cutOffFrequency / self.sampleFrequency * 2,
                btype="highpass",
            )
            rawData = h5py.File(os.path.join(self.inputPath, self.filename), "r")
            indexX = self.electrodeChannelMapping[0, :] % self.chipWidth - boundX[0]
            indexY = (
                rows.shape[0]
                - 1
                - (self.electrodeChannelMapping[0, :] / self.chipWidth - boundY[0])
            ).astype(np.int32)
            movieMatrix[:, indexY, indexX] = (
                self.microVoltPerBit
                * (
                    lfilter(
                        coeffB,
                        coeffA,
                        (
                            rawData.get("sig")[
                                np.squeeze(self.electrodeChannelMapping[1, :]),
                                int(max(startFrame - 200, 0)) : endFrame,
                            ]
                        ),
                    ).T
                )[int(200 + min(startFrame - 200, 0)) :, :]
            )

            del rawData
        else:
            for i in range(self.electrodeChannelMapping.shape[1]):
                indexX = self.electrodeChannelMapping[0, i] % self.chipWidth - boundX[0]
                indexY = int(
                    rows.shape[0]
                    - 1
                    - (self.electrodeChannelMapping[0, i] / self.chipWidth - boundY[0])
                )
                spikes = (self.spikes[i][0])[
                    np.argwhere(
                        np.logical_and(
                            self.spikes[i][0] < endFrame,
                            self.spikes[i][0] >= startFrame,
                        )
                    )
                ]
                movieMatrix[:, indexY, indexX] = 0
                movieMatrix[spikes - startFrame, indexY, indexX] = 1
                if decayFrames > 0:
                    decayVector = np.arange(decayFrames + 1, 0, -1)
                    movieMatrix[:, indexY, indexX] = convolve(
                        decayVector, np.squeeze(movieMatrix[:, indexY, indexX])
                    )[:-decayFrames]

        fig, ax = plt.subplots(figsize=figureSize, dpi=dpi)
        maxVal = np.max(movieMatrix[~np.isnan(movieMatrix)])
        minVal = np.min(movieMatrix[~np.isnan(movieMatrix)])
        if positivNegativePeaks == -1 and readRawDataBool:
            maxVal = 0
        elif positivNegativePeaks == 1:
            minVal = 0
        else:
            maxVal = max(np.abs(minVal), np.abs(maxVal))
            minVal = -maxVal
        plt.imshow(
            movieMatrix[0], vmin=minVal, vmax=maxVal, origin="lower", cmap=colormap
        )
        if readRawDataBool:
            plt.colorbar()
            plt.title("Signal in µV")
        im = plt.imshow(
            movieMatrix[0], vmin=minVal, vmax=maxVal, origin="lower", cmap=colormap
        )
        del maxVal, minVal

        def update(i):
            im.set_array(movieMatrix[i])
            return (im,)

        if showPlotBool:
            ani = animation.FuncAnimation(
                fig,
                update,
                frames=len(movieMatrix),
                interval=int(1000 / int(self.sampleFrequency / slowDownFactor)),
            )
            plt.show()
        else:
            ani = animation.FuncAnimation(fig, update, frames=len(movieMatrix))
            ani.save(
                os.path.join(self.outputFolder, filename),
                fps=int(self.sampleFrequency / slowDownFactor),
                extra_args=["-vcodec", "libx264"],
            )
        plt.close()
        del endFrame
        del fig
        del movieMatrix
        del boundX, boundY
        del columns, rows

    def signalPropagationPlot(
        self,
        startFrame,
        durationInFrames,
        filename="Signal_Propagation",
        figureSize=(6.4, 4.8),
        showPlotBool=False,
        storePlotBool=False,
        colormap="seismic",
        dpi=100,
        format="png",
    ):
        """
        Generates a plot of a selected timeframe on when the first spike of an electrode hits.
        Use self.blankingEnd or self.spikes for good start frames.

        :param startFrame: The first frame of the measurement which is shown in the plot.
        :param durationInFrames: How many frames should be in the plot.
        :param filename: As what the plot should be saved.
        :param figureSize: Tuple which specifies the size of the output.
        :param showPlotBool: If True, shows the matplotlib plot. The plot won't be saved!
        :param colormap: Which matplotlib cmap is used for the plot.
        :param dpi: Pixels per inch for the plot.
        :param format: Type as what the file should be saved.
        :return: Saves a plot of the first spikes color coded in regards with their delay without amplitude information.
        """

        if colormap not in plt.colormaps():
            print("colormap does not exist in Matplotlib.")
            colormap = "seismic"

        boundX = [
            min(self.electrodeChannelMapping[0, :] % self.chipWidth),
            max(self.electrodeChannelMapping[0, :] % self.chipWidth) + 1,
        ]
        boundY = [
            int(min(self.electrodeChannelMapping[0, :] / self.chipWidth)),
            int(max(self.electrodeChannelMapping[0, :] / self.chipWidth)) + 1,
        ]
        columns = np.arange(boundX[0], boundX[1])
        rows = np.arange(boundY[0], boundY[1])
        endFrame = durationInFrames + startFrame
        heatmap = np.zeros([rows.shape[0], columns.shape[0]])
        heatmap[:] = np.nan

        for i in range(self.electrodeChannelMapping.shape[1]):
            indexX = self.electrodeChannelMapping[0, i] % self.chipWidth - boundX[0]
            indexY = int(
                self.electrodeChannelMapping[0, i] / self.chipWidth - boundY[0]
            )
            spikes = (self.spikes[i][0])[
                np.argwhere(
                    np.logical_and(
                        self.spikes[i][0] < endFrame, self.spikes[i][0] >= startFrame
                    )
                )
            ]
            if spikes.size != 0:
                heatmap[indexY, indexX] = (
                    (spikes[0] - startFrame) * 1000 / self.sampleFrequency
                )
            else:
                heatmap[indexY, indexX] = 0

        fig = plt.figure(figsize=figureSize, dpi=dpi)
        plt.imshow(heatmap, cmap=colormap)
        plt.colorbar()
        plt.title("Delay of the first spike [ms]")
        plt.xlabel("Electrode x")
        plt.ylabel("Electrode y")
        if showPlotBool:
            plt.show()
        elif storePlotBool:
            fig.savefig(
                os.path.join(self.outputFolder, filename + ".{}".format(format)),
                dpi=fig.dpi,
                transparent=True,
                format=format,
            )
        plt.close()
        del fig
        return heatmap

    def stimulationResponseDelayOverTime(
            self, electrodeNumbers, numberOfStimsToSkip = 0, window = 300, figureSize = (6.4,4.8), filename = "Response_Delay", showPlotBool = False, dotSize = 0.3,
            colors = [], alpha = 1.0, autocolor = True, plotAllResponsesBool = False, plotLegendBool = True, yAxisUnit = "Stim", dpi=100, format = 'png'
    ):
        """
        Generates a plot where for each specified electrode, where the delays of the first spike which occurs after
        the signal was blanked are plotted.

        :param electrodeNumbers: The electrodes to be plotted.
        :param numberOfStimsToSkip: How many blanked signals are skipped.
        :param window: Maximum delay in frames, which are still considered.
        :param figureSize: Tuple which specifies the size of the output.
        :param filename: As what the plot should be saved.
        :param showPlotBool: If True, shows the matplotlib plot. The plot won't be saved!
        :param dotSize: How large a point in the scatter plot is.
        :param colors: A list for the colors of the dots of an electrode. Same order as electrodeNumbers.
        :param alpha: transparency of the dots (0: transparent, 1: opaque)
        :param plotAllResponsesBool: If True, plots all spikes which are in between the end of blanking and the end of the window.
        :param plotLegendBool: If True, electrodes are listed with their respective color.
        :param yAxisUnit: Unit for y-axis. Can be 'Stim', 'Hour', 'Minute'
        :param dpi: Pixels per inch for the plot.
        :param format: Type as what the file should be saved.
        :param autocolor: use the RGB color generated in circularcolorCoding-method
        :return: delayMatrix (list of spikes with delay time post stimulus), colors (color values for all electrodes), all_spikes_absolute (list of all spikes occuring within post stim window with absolute time)
        """

        if isinstance(electrodeNumbers, list):
            electrodeNumbers = np.squeeze(np.asarray(electrodeNumbers))
        else:
            electrodeNumbers = np.zeros(1,dtype=np.int32) + electrodeNumbers

        if autocolor:
            sorted_color_electrodes, rgb_cycle,_ = self.circular_color_coding()
            indices_to_plot = []
            colors_to_plot = []

            for el in electrodeNumbers:
                indices_to_plot.append(np.where(sorted_color_electrodes == el)[0][0])

            for index in indices_to_plot:
                colors_to_plot.append(rgb_cycle[index])
            colors = colors_to_plot

        if not np.isin(electrodeNumbers, self.electrodeChannelMapping[0,:]).all():
            print("At least one electrode was not routed.")
            return
        if self.blankingEnd.size == 0:
            print("No blanking was done.")
            return

        stims = np.arange(0,self.blankingEnd.shape[0],numberOfStimsToSkip+1)
        blankingEndTimes = np.squeeze(self.blankingEnd[stims])
        indices = [int(np.argwhere(self.electrodeChannelMapping[0,:]==x)) for x in electrodeNumbers]

        if yAxisUnit == "Hour":
            stims = blankingEndTimes / self.sampleFrequency / 3600
        elif yAxisUnit == "Minute":
            stims = blankingEndTimes / self.sampleFrequency / 60
        else:
            yAxisUnit = "Stim"

        delayMatrix = []

        n=0

        #store the absolute time of the spike within the post stimulus window
        all_spikes_absolute = []

        for i in self.electrodeChannelMapping[1,indices]:
            index = np.argwhere(self.electrodeChannelMapping[1,:]==i).item()
            if plotAllResponsesBool:
                delaysOfElectrode = np.asarray([[np.nan],[0]])
            else:
                delaysOfElectrode = np.zeros([2,blankingEndTimes.shape[0]])

            spikes_absolute = []

            for j in range(blankingEndTimes.shape[0]):
                spikes = ((self.spikes[index][0])[np.argwhere(np.logical_and(self.spikes[index][0] < blankingEndTimes[j]+window, self.spikes[index][0] >= blankingEndTimes[j]))] - blankingEndTimes[j]) * 1000 / self.sampleFrequency
                if spikes.size:
                    spikes_absolute.append((self.spikes[index][0])[np.argwhere(np.logical_and(self.spikes[index][0] < blankingEndTimes[j]+window, self.spikes[index][0] >= blankingEndTimes[j]))])
                if spikes.size != 0 and plotAllResponsesBool:
                    toConcat = np.zeros([2,spikes.shape[0]]) + stims[j]
                    toConcat[0,:] = np.squeeze(spikes)
                    delaysOfElectrode = np.concatenate((delaysOfElectrode,toConcat),axis=1)
                    del toConcat
                elif spikes.size != 0 and not plotAllResponsesBool:
                    delaysOfElectrode[0,j] = spikes[0]
                    delaysOfElectrode[1,j] = stims[j]
                elif plotAllResponsesBool:
                    delaysOfElectrode = np.concatenate((delaysOfElectrode, np.asarray([[np.nan],[stims[j]]])),axis=1)
                else:
                    delaysOfElectrode[0, j] = np.nan
                    delaysOfElectrode[1, j] = stims[j]
            n += 1

            if len(spikes_absolute):
                abs_times_to_append = np.squeeze((np.concatenate(spikes_absolute)))
                nr_peaks = abs_times_to_append.shape
            else:
                abs_times_to_append = np.array([])
                nr_peaks = 0

            if abs_times_to_append.ndim == 0:
                abs_times_to_append = np.expand_dims(abs_times_to_append, axis = 0)

            spike_data = {"peak_heights": np.array([]), "nr_of_peaks": nr_peaks, "electrode": self.electrodeChannelMapping[0][index]}
            all_spikes_absolute.append((np.array(abs_times_to_append), spike_data))


            delayMatrix.append(np.asarray(delaysOfElectrode))
            del delaysOfElectrode


        fig = plt.figure(figsize=figureSize,dpi=dpi)
        plt.ylabel(yAxisUnit)
        plt.xlabel('Delay of spike [ms]')
        n=0
        for i in self.electrodeChannelMapping[1,indices]:
            index = np.argwhere(self.electrodeChannelMapping[1,:]==i)
            color = None
            if n < len(colors):
                color = colors[n]
            plt.scatter(delayMatrix[n][0,:], delayMatrix[n][1,:], marker= 'o', linewidths=0, s=dotSize, color=color, label = "Electrode {}".format(np.squeeze(self.electrodeChannelMapping[0,index])), alpha = alpha)
            n+=1
        plt.xlim(0, window * 1000 / self.sampleFrequency)
        plt.title("Delay of stimulus response over time")
        if plotLegendBool:
            plt.legend()

        if showPlotBool:
            plt.show()
        else:
            fig.savefig(os.path.join(self.outputFolder, filename+".{}".format(format)), dpi=fig.dpi, transparent=True, format=format)
        plt.close()
        del fig
        del blankingEndTimes
        del stims
        del indices
        del n
        return delayMatrix, colors, all_spikes_absolute

    def stimulationResponseCompressTo1D(
        self,
        delayMatrix,
        colors,
        figureSize=(6.4, 4.8),
        filename="Response_Delay_1D",
        window=9000,
        showPlotBool=False,
        kernel=[1],
        alpha=1,
        plotLegendBool=False,
        dpi=100,
        format="png",
    ):
        """
        Compress delay matrix to one row with kernel for intensity.

        :param delayMatrix: n x spikes list of lists with latency, iterations rows (2xnum_spikes matrix)
        :param colors: A list for the colors of the dots of an electrode. Same order as electrodeNumbers.
        :param window: max considered latency, used for maximum length
        :param figureSize: Tuple which specifies the size of the output.
        :param filename: As what the plot should be saved.
        :param showPlotBool: If True, shows the matplotlib plot. The plot won't be saved!
        :param kernel: kernel for smearing of single spikes, centred to middle
        :param alpha: transparency of the kernel (0: transparent, 1: opaque)
        :param plotLegendBool: If True, electrodes are listed with their respective color.
        :param dpi: Pixels per inch for the plot.
        :param format: Type as what the file should be saved.
        """

        numberOfElectrodes = len(delayMatrix)
        delayBinaryMatrix = np.zeros((numberOfElectrodes, window))

        # transform list of spikes into binary matrix of size numElectrodes x samples
        for n, delaysOfElectrode in enumerate(delayMatrix):
            for index in delaysOfElectrode.shape[1]:
                delayBinaryMatrix[[n, delaysOfElectrode[0,index]]] += 1

        # smear spike bins for smoother results
        if kernel != [1]:
            delayProjection = convolve1d(input=delayBinaryMatrix, weights=kernel, mode='constant', cval=0)
            delayProjection = delayProjection[len(kernel):-len(kernel)]

        fig = plt.figure(figsize=figureSize, dpi=dpi)
        plt.ylabel("Number of spikes")
        plt.xlabel("Delay of spike [ms]")

        for n, delayDistributionElectrode in delayProjection:
            plt.plot(np.arange(window), delayDistributionElectrode, color=colors[n], alpha=alpha)

        plt.xlim(0, window * 1000 / self.sampleFrequency)
        plt.title("Delay of stimulus response over time")
        if plotLegendBool:
            plt.legend()

        if showPlotBool:
            plt.show()
        else:
            fig.savefig(
                os.path.join(self.outputFolder, filename + ".{}".format(format)),
                dpi=fig.dpi,
                transparent=True,
                format=format,
            )
        plt.close()
        del fig
        del n
        return delayProjection

    def plotSelectedElectrodes(
        self,
        electrodeNumbers,
        colormap="seismic",
        showPlotBool=True,
        figureSize=(16,9),
        filename="Selected_Electrodes",
        dpi=100,
        format="png",
    ):
        """
        :param electrodeNumbers: The electrodes to be plotted.
        :param colormap: Which matplotlib cmap is used for the plot.
        :param showPlotBool: If True, shows the matplotlib plot. The plot won't be saved!
        :param figureSize: Tuple which specifies the size of the output.
        :param filename: As what the plot should be saved.
        :param dpi: Pixels per inch for the plot.
        :param format: Type as what the file should be saved.
        :return: Saves a plot where the selected electrodes are highlighted.
        """

        if isinstance(electrodeNumbers, int):
            electrodeNumbers = np.zeros(1, dtype=np.int32) + electrodeNumbers
        else:
            electrodeNumbers = np.squeeze(np.asarray(electrodeNumbers))
        if not np.isin(electrodeNumbers, self.electrodeChannelMapping[0, :]).all():
            print("At least one electrode was not routed.")
            return

        if colormap not in plt.colormaps():
            print("colormap does not exist in Matplotlib.")
            colormap = "seismic"

        boundX = [
            min(self.electrodeChannelMapping[0, :] % self.chipWidth),
            max(self.electrodeChannelMapping[0, :] % self.chipWidth) + 1,
        ]
        boundY = [
            int(min(self.electrodeChannelMapping[0, :] / self.chipWidth)),
            int(max(self.electrodeChannelMapping[0, :] / self.chipWidth)) + 1,
        ]
        columns = np.arange(boundX[0], boundX[1])
        rows = np.arange(boundY[0], boundY[1])

        heatmap = np.zeros([rows.shape[0], columns.shape[0]])
        heatmap[:] = np.nan
        for i in range(self.electrodeChannelMapping.shape[1]):
            indexX = self.electrodeChannelMapping[0, i] % self.chipWidth - boundX[0]
            indexY = int(
                self.electrodeChannelMapping[0, i] / self.chipWidth - boundY[0]
            )
            if np.isin(self.electrodeChannelMapping[0, i], electrodeNumbers).all():
                heatmap[indexY, indexX] = 1
            else:
                heatmap[indexY, indexX] = 0

        fig = plt.figure(figsize=figureSize, dpi=dpi)
        plt.imshow(heatmap, cmap=colormap)
        plt.title("Selected Electrodes")
        plt.xlabel("Electrode x")
        plt.ylabel("Electrode y")
        if showPlotBool:
            plt.show()
        else:
            fig.savefig(
                os.path.join(self.outputFolder, filename + ".{}".format(format)),
                dpi=fig.dpi,
                transparent=True,
                format=format,
            )

        del fig

    def mostActiveElectrodes(self, quantity=1):
        """

        :param quantity: How many electrodes should be returned.
        :return: The most active electrodes.
        """
        electrodesAndPeaks = np.zeros(
            self.electrodeChannelMapping.shape, dtype=np.int32
        )
        electrodesAndPeaks[0, :] = np.asarray([i[1]["electrode"] for i in self.spikes])
        electrodesAndPeaks[1, :] = np.asarray(
            [i[1]["nr_of_peaks"] for i in self.spikes]
        )
        electrodesAndPeaks = electrodesAndPeaks[
            0, electrodesAndPeaks[1, :].argsort()[::-1]
        ]
        return np.squeeze(electrodesAndPeaks[0:quantity])

    def circular_color_coding(
        self,
        electrodes=[],
        showPlotBool=False,
        savePlotBool=False,
        figureSize=(8, 8),
        filename="CircularcolorCoding",
        dotSize=0.3,
        dpi=100,
        format="png",
        modulate_dotsize=False,
        dotbase=20,
        highlight_electrode=[], 
    ):
        """
        :param electrodes: Electrodes which have to be color coding. If no input, all routed electrodes are taken.
        :param showPlotBool: If True, shows the matplotlib plot. The plot won't be saved!
        :param figureSize: Tuple which specifies the size of the output.
        :param filename: As what the plot should be saved.
        :param dotSize: How large a point in the scatter plot is.
        :param dpi: Pixels per inch for the plot.
        :param format: Type as what the file should be saved.
        :param modulate_dotsize: Whether relative dotsize should be computed per electrode according to spike frequency
        :param dotbase: in case modulated dotsize is desired
        :return: Returns an array of electrodes and an array with their corresponding colors. Note that the order of the electrodes may differ in the output. Also returns dotsize relative to spike frequency.
        """
        try:
            if len(electrodes) == 0:
                electrodes = self.electrodeChannelMapping[0, :]
        except:
            electrodes = np.zeros(1, dtype=np.int32) + electrodes

        boundX = [
            min(electrodes % self.chipWidth),
            max(electrodes % self.chipWidth) + 1,
        ]
        boundY = [
            int(min(electrodes / self.chipWidth)),
            int(max(electrodes / self.chipWidth)) + 1,
        ]

        Rad = np.zeros(len(electrodes))
        Phi = np.zeros(len(electrodes))

        for i, el in enumerate(electrodes):
            x = electrodes[i] % self.chipWidth - (boundX[1] + boundX[0]) / 2
            y = electrodes[i] / self.chipWidth - (boundY[1] + boundY[0]) / 2
            Rad[i] = np.sqrt(x**2 + y**2)
            if x > 0:
                Phi[i] = np.arctan(y / x)
            elif x < 0 and y >= 0:
                Phi[i] = np.arctan(y / x) + np.pi
            elif x < 0 and y < 0:
                Phi[i] = np.arctan(y / x) - np.pi
            elif x == 0 and y > 0:
                Phi[i] = np.pi / 2
            elif x == 0 and y < 0:
                Phi[i] = -np.pi / 2

        phi = np.linspace(0, 2 * np.pi, len(electrodes))
        rgb_cycle = np.vstack(
            (  # Three sinusoids
                0.5 * (1.0 + np.cos(phi)),  # scaled to [0,1]
                0.5 * (1.0 + np.cos(phi + 2 * np.pi / 3)),  # 120° phase shifted.
                0.5 * (1.0 + np.cos(phi - 2 * np.pi / 3)),
            )
        ).T  # Shape = (60,3)

        sortedIndices = np.arange(len(Phi))[Phi.argsort()]

        self.sorted_color_electrodes = electrodes[sortedIndices]
        self.sorted_colors = rgb_cycle

        if modulate_dotsize:
            sorted_act, sorted_freq = self.select_most_active_el(
                method="most_active", value=len(electrodes[sortedIndices])
            )
            dotsize = []
            for i in range(len(electrodes[sortedIndices])):
                dotsize.append(
                    dotbase
                    * sorted_freq[
                        np.where(sorted_act == electrodes[sortedIndices][i])[0][0]
                    ]
                    / np.max(sorted_freq)
                )
        else:
            dotsize = dotSize
        if showPlotBool or savePlotBool:
            fig = plt.figure(figsize=figureSize, dpi=dpi)
            plt.scatter(
                electrodes[sortedIndices] % self.chipWidth
                - np.min(electrodes[sortedIndices] % self.chipWidth),
                y=np.max((electrodes[sortedIndices] / self.chipWidth).astype(np.int32))
                - (electrodes[sortedIndices] / self.chipWidth).astype(np.int32),
                marker="s",
                c=rgb_cycle,
                s=dotsize,
            )
            plt.gca().set_aspect('equal')
            if len(highlight_electrode):
                
                plt.scatter(
                    highlight_electrode % self.chipWidth
                    - np.min(electrodes[sortedIndices] % self.chipWidth),
                    y=np.max((electrodes[sortedIndices] / self.chipWidth).astype(np.int32))
                    - (highlight_electrode / self.chipWidth).astype(np.int32),
                    marker = 'o',
                    facecolors='none', 
                    edgecolors='black',
                    s=30, 
                    linewidth=15, 
                )
                plt.gca().set_aspect('equal')
            plt.title("Circular color Coding for Electrodes")
            plt.xlabel("Electrode x")
            plt.ylabel("Electrode y")

        if showPlotBool:
            plt.show()
            
        elif savePlotBool:
            fig.savefig(
                os.path.join(self.outputFolder, filename + ".{}".format(format)),
                dpi=fig.dpi,
                transparent=True,
                format=format,
            )
        
        if showPlotBool or savePlotBool:
            del fig
        del Phi
        del Rad
        return electrodes[sortedIndices], rgb_cycle, dotsize

    def electrode_subsampling(
        self, 
        min_dist=5,
        point_limit=None,
        showPlotBool=False,
        figureSize=(16, 9),
        initialSortBy='Activity'
    ):
        """
        :param min_dist: Minimal distance in pixels that a new electrode has to be from already selected ones
        :param point_limit: Maximal number of electrodes the algorithm will select before terminating
        :param showPlotBool: Whether or not a plot should be generated by default
        :param figureSize: Size of the generate figure
        :param initialSortBy: Determines by which metrics the intial electrode list will be sorted
        :Returns list of the electrode numbers chosen during the subsampling as well as their coordinates
        """

        nr_eles = self.electrodeChannelMapping.shape[1]
        if initialSortBy == 'Activity':
            act_elecs = self.mostActiveElectrodes(nr_eles)
        elif initialSortBy == 'Frequency':
            act_elecs = self.mostActiveElectrodes(method="frequency_threshold")

        #last column: 0: still have to look at, 1: chosen as point, 2: discarded
        act_coords = []
        for e in act_elecs:
            x, y = self.convert_elno_to_xy(e)
            act_coords.append([x, y, 0])
        act_coords = np.array(act_coords)

        i = 0
        # Continue while there are electrodes left that haven't been looked at & the point limit hasn't been reached
        while(np.any(act_coords[:, 2] == 0)):
            if i == point_limit:
                break
            # Find next center coord, and set to looked at
            try:
                center_ind_abs = np.argwhere(act_coords[:, 2] == 0)[0]
            except:
                center_ind_abs = np.argwhere(act_coords[:, 2] == 0)
                break
            center_coords = act_coords[center_ind_abs, :2]
            act_coords[center_ind_abs, 2] = 1   

            # Find all points to be discared based on distance
            poi_abs_ind = np.argwhere(act_coords[:, 2] == 0)
            coords_to_look_at = act_coords[poi_abs_ind, :2]

            distances = np.hypot(*(coords_to_look_at - center_coords).T)
            to_discard_rel_ind = np.argwhere(distances <= min_dist)
            if to_discard_rel_ind.size == 0:
                i += 1
                continue
            elif to_discard_rel_ind.size == 1:
                act_coords[poi_abs_ind, 2] = 2
            else:
                act_coords[poi_abs_ind[to_discard_rel_ind], 2] = 2

            i += 1

        selected_coords = act_coords[act_coords[:, 2] == 1, :2]
        selected_electrodes = act_elecs[act_coords[:, 2] == 1]

        if showPlotBool:
            self.plotSelectedElectrodes(selected_electrodes, figureSize=figureSize)

        return selected_electrodes, selected_coords
        
    def convert_xy_to_elno(self, x, y):
        """
        Convert coordinates (x,y) to an electrode number from 0 to 26'399.
        This function uses numpy convention (x is axis 0, y is axis 1)
        Input: x (0-119), y (0-219)
        """
        return x * self.chipWidth + y % self.chipWidth

    def convert_elno_to_xy(self, elno):
        """
        Convert electrode number (0-26399) to electrode coordinates (x,y).
        This function uses numpy convention (x is axis 0, y is axis 1)
        Input: elno (0-26'399)
        """
        x = int(elno / self.chipWidth)
        y = elno % self.chipWidth
        return x, y

    def select_most_active_el(self, method="frequency_threshold", value=0.2):
        """
        Returns a list of the most active electrodes.
        :param method: "frequency_threshold" returns all electrodes with spiking frequency higher than value, "most_active" returns the N most active electrodes (N=value)
        :param value: either frequency threshold or how many electrodes to select
        """
        spike_count = np.asarray(
            [
                self.spikes[i][1]["nr_of_peaks"]
                / self.experimentDurationBlanked
                * self.sampleFrequency
                for i in range(len(self.spikes))
            ]
        )
        electrodes = np.asarray(
            [self.spikes[i][1]["electrode"] for i in range(len(self.spikes))]
        )
        if method == "frequency_threshold":
            threshed_idx = np.asarray(
                [spike_count[idx] > value for idx in range(len(spike_count))]
            )
            return electrodes[threshed_idx], spike_count[threshed_idx]
        elif method == "most_active":
            sorted_idx = np.flip(np.argsort(spike_count))
            return electrodes[sorted_idx[:value]], spike_count[sorted_idx[:value]]
        else:
            raise ValueError("Unknown Method provided.")

    def get_overlap(self, limFrames, return_mapping=True, exclude_repetitions=False, store_times=False):
        results = []
        if not isinstance(limFrames, list):
            limFrames = [-limFrames, limFrames]
        
        # self.spikes is list of peaks with peak[0] spiketimes and peak[1] is dict with "electrode"
        indices = self.electrodeChannelMapping[0, :] # absolute

        numElectrodes = len(indices)
        overlap_matrix = np.empty((numElectrodes,numElectrodes), dtype=object)
        mapping_matrix = np.zeros((numElectrodes,numElectrodes,4))
        STTRP_matrix = np.empty((numElectrodes,numElectrodes), dtype=object)
        for i in np.ndindex(overlap_matrix.shape): overlap_matrix[i] = []
        for i in np.ndindex(STTRP_matrix.shape): STTRP_matrix[i] = []
        # for i in range(self.electrodeChannelMapping.shape[1]):
        #     print(f"{i}: {self.electrodeChannelMapping[0,i]} - {self.electrodeChannelMapping[1,i]}")
        for trig_el_id_rel, trig_el_id_abs in enumerate(self.electrodeChannelMapping[0, :]):
            spike_vec_trig = self.spikes[trig_el_id_rel][0]
            if len(spike_vec_trig):
                if exclude_repetitions:
                    # if same electrode spike occurs on left side of window then exclude spike
                    exclude_mat = (spike_vec_trig - spike_vec_trig[:,None])
                    include_ids = np.logical_not(
                        np.any(
                            np.logical_and(
                                exclude_mat > limFrames[0], 
                                exclude_mat < 0), 
                            axis=1))
                    spike_vec_trig = spike_vec_trig[include_ids]
                for compare_el_id_rel, compare_el_id_abs in enumerate(self.electrodeChannelMapping[0, :]):                    
                    spike_vec_compare = self.spikes[compare_el_id_rel][0]
                    if len(spike_vec_compare):
                        normed_mat = (spike_vec_compare - spike_vec_trig[:,None])
                        valid_ids_mat = np.logical_and(normed_mat > limFrames[0], normed_mat < limFrames[1])
                        threshed_mat = normed_mat[valid_ids_mat]
                        if np.all(threshed_mat==0):
                            overlap_vec = threshed_mat # exception for trigger on same electrode and all centred to zeros
                        else:
                            overlap_vec = threshed_mat[threshed_mat.astype(bool)]
                        if overlap_vec.shape[0]:
                            # print(f"{trig_el_id_rel} - {compare_el_id_rel}: {overlap_vec.shape[0]}/{normed_mat.shape[0]}")
                            overlap_matrix[trig_el_id_rel, compare_el_id_rel] = overlap_vec
                            mapping_matrix[trig_el_id_rel, compare_el_id_rel,:] = [
                                trig_el_id_abs, 
                                compare_el_id_abs, 
                                overlap_vec.shape[0],
                                normed_mat.shape[0]]
                            STTRP_matrix[trig_el_id_rel, compare_el_id_rel] = (spike_vec_trig, normed_mat, valid_ids_mat)
        self.STTH_matrix = overlap_matrix
        if return_mapping:
            if store_times:
                return overlap_matrix, mapping_matrix, indices, STTRP_matrix
            return overlap_matrix, mapping_matrix, indices
        else:
            return overlap_matrix

    def get_boundaries(self):
        """
        Returns the boundaries of the routings for zooming into custom plots.
        """
        boundX = [min(self.electrodeChannelMapping[0, :] % self.chipWidth),
            max(self.electrodeChannelMapping[0, :] % self.chipWidth) + 1]
        boundY = [int(min(self.electrodeChannelMapping[0, :] / self.chipWidth)),
                    int(max(self.electrodeChannelMapping[0, :] / self.chipWidth)) + 1]
        return boundX, boundY

    def get_raw_trace(self, electrode_nr = -1):
        """Returns the raw trace recorded at an electrode or all electrodes.
        :param electrode_nr: which electrode to get the raw signal from, if "-1", then return whole signal
        """
        if self.auto_parse:
            rawData = h5py.File(os.path.join(self.inputPath, self.filename), "r")
        else:
            try:
                rawData = h5py.File(os.path.join(self.inputPath, self.filename), "r")
            except:
                rawData = h5py.File(self.filename, "r")


        if electrode_nr == -1:
            trace = (rawData.get('sig')[:, :])
        else:
            index = self.electrodeChannelMapping[1][np.where(self.electrodeChannelMapping[0] == electrode_nr)[0][0]]
            trace = (rawData.get('sig')[index, :])
        return trace