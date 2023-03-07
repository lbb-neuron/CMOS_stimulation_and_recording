import sys
import os
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.ndimage import convolve
import cv2
import copy
import pickle
import PySimpleGUI as sg
from tqdm import tqdm
import h5py
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

from stimulationPatternClass import stimulationPatternClass
import maxlab
import maxlab.system
import maxlab.chip
import maxlab.util
import maxlab.saving

import AnalyZor_Class as Analyzor
import AnalyZor_Helper

def main():

##########################Placeholders########################################
    vmapFile = ''
    noText = '-'
    defaultElectrode = '0'
    reverseConfString = ''

##########################GUISETTINGS########################################
    textSize = 10
    titleSize = 30
    textFont = ("fixed", textSize)
    titleFont = ("fixed", titleSize)

###############################################codeSettings######################################################
    vmap_cmap = "magma"
    quick_analysis_cmap = "inferno"
    sizeElectrodeArray = np.array([120,220])

    patchHeight = 55
    patchWidth = 55
    record_all_coords = []

    threshold = 30

    cutOffFrequency = 300
    filterOrder = 2
    spikeDistance = 50

    trigger_electrode = None

    minimalDistance = 0
    pointLimit = 200

    windowSize = 15
    preWindow = 0

    reverseConfBool = False
    reverseConfKernelSize = 5

    loadedScopeBool = False
    onlySpikesBool = True
    legacyFormatBool = True
    storeAnalyzorBool = True

    voltageMapsPath = os.path.join(sys.path[0], 'voltageMapArrays')
    recordingsPath = os.path.join(sys.path[0], 'recordings')
    electrodeSelectionPaths = os.path.join(sys.path[0], 'electrodeSelections')
    analyzorObjectPaths = os.path.join(sys.path[0], 'analyzorObjects')

    stimulationBurnInDuration = 0
    nrOfTotalLoops = 1
    nrStimulationLoops = 1
    stimulationIterationDuration = 0
    stimulationBurnOutDuration = 0
    nrOfLastSequences = 0
    recordBurnTimeBool = True
    recordLastStimsBool = True
    recordFirstStimsBool = True

    unitListSmallShort = ['s', 'ms','µs']
    unitListSmall = ['min','s', 'ms','µs']
    unitList = ['h','min','s','ms','µs','ns']
    unitListLarge = ['s','min','h']
    stimulusAmplitude = 0
    sampleRate = 20000
    stimulationEditBool = False
    pulseDuration = 0.0004

    stimFileBool = False

####################################################StoredVariables###############################################################
    stimulationDuration = stimulusDuration(stimulationBurnInDuration, stimulationIterationDuration,stimulationBurnOutDuration, nrOfTotalLoops)
    stimulationPatternList = []
    stimulationElectrodesList = []
    selectedStimulationElectrodeIndex = 0
    selectedPatternIndex = 0

    coords = []
    sampledElectrodes = []

    VMArray = np.zeros(sizeElectrodeArray)
    elecSelection = []

    centerElectrodeAxisX = int(sizeElectrodeArray[1])
    centerElectrodeAxisY = int(sizeElectrodeArray[0])

#################################################CHIP######################################################
    maxlab.util.initialize()

###############################################GUIInit#############################################################################
###################################################################################################################################
    file_list = os.listdir(voltageMapsPath)
    selection_list = os.listdir(electrodeSelectionPaths)
    recordings_list = os.listdir(recordingsPath)
    recordings_list = [s for s in recordings_list if "h5" in s]
    analyzor_list = os.listdir(analyzorObjectPaths)

###############################################PreparationTab#####################################################################

    data_column = [
        [sg.Text("Enter important experiment information below:\nIt will be used for filtering and determine the names of newly generated files.")],
        [sg.Text('Chip ID'),
         sg.InputText(size=(20, 1), key="CHIP_ID", enable_events=True, default_text="0000")],
        [sg.Text('Network Number'),
         sg.InputText(size=(20, 1), key="NETWORK_NUMBER", enable_events=True, default_text="0")],
        [sg.Text("Days in Vitro"),
         sg.InputText(size=(10,1), key="DIV", enable_events=True, default_text=str("0"))],
        [sg.Text("")],
        [sg.Text("Custom Labels:\nYou can enter key - values pairs below which will be inlcuded in the file name")],
        [sg.InputText(size=(20,1), key="CUSTOM_LABEL_NAME_1", enable_events=True,default_text=''),
         sg.InputText(size=(20,1), key="CUSTOM_LABEL_VALUE_1", enable_events=True,default_text='')],
        [sg.InputText(size=(20,1), key="CUSTOM_LABEL_NAME_2", enable_events=True,default_text=''),
         sg.InputText(size=(20,1), key="CUSTOM_LABEL_VALUE_2", enable_events=True,default_text='')],
        [sg.InputText(size=(20,1), key="CUSTOM_LABEL_NAME_3", enable_events=True,default_text=''),
         sg.InputText(size=(20,1), key="CUSTOM_LABEL_VALUE_3", enable_events=True,default_text='')],
        [sg.Text("Storage Location:"),
        sg.InputText(size=(40,1), key="LOCATION", enable_events=True, default_text=str(recordingsPath))],
        [sg.Text("")],
        [sg.Text("File Name to be used:")],
        [sg.Text("ID{}_NW{}_DIV{}_DATE{}_CUSTOMLABELS{}".format("0000", "0", "7", "yyyymmdd", "(...)"), key='FILE_NAME')],
        [sg.Button('Show Current Name')],
    ]

    vmap_column = [
        [sg.Text("We use a voltage map to locate the PDMS structure on the chip and\ndefine our area of interest to record from.")],
        [sg.Text("Available Voltage Maps:")],
        [sg.Listbox(values=file_list, enable_events=True, size=(60, 10), key="FILE_LIST")],
        [sg.Button("Filter by Chip Number"),
         sg.Button("Refresh & Show all", key='RESET_FILTER'),
         sg.Button("Plot Voltage Map")],
        [sg.Text("")],
        [sg.Text('If no voltage map can be found, you can create one now.')],
        [sg.Button("Create New Voltage Map")]
    ]

    selection_column = [
        [sg.Text("We can now load or create an electrode selection based on the selected voltage map.\nIt will define the electrodes which will be recorded from during experiments.")],
        [sg.Text("Select Premade Electrode Selection:")],
        [sg.Listbox(values=selection_list, enable_events=True, size=(60, 10), key="SELECTION_LIST")],
        [sg.Button("Filter by Chip Number", key='FILTER_ELECTRODE_SELECTION'),
         sg.Button("Refresh & Show all", key='RESET_ELECTRODE_FILTER')],
        [sg.Button("Plot Electrode Selection"),
         sg.Button("Load Selection Into Scope")],
        [sg.Text("If no existing electrode selection is available, we can create a new one in two ways:\n1. Using the electrode selection script \n2. Defining a center electrode and a box around it")],
        [sg.Text('1. Electrode Selection Script')],
        [sg.Button("Create New Electrode Selection")],
        [sg.Text("After creating a new Electrode Selection, it can then be found in the list above.")],
        [sg.Text("")],
        [sg.Text('2. Selection via Center Electrode')],
        [sg.Text("Center Electrode X&Y Coordinate"),
         sg.InputText(size=(3, 1), key="ELECTRODE_X", enable_events=True, default_text=str(sizeElectrodeArray[1])),
         sg.InputText(size=(3, 1), key="ELECTRODE_Y", enable_events=True, default_text=str(sizeElectrodeArray[0]))],
        [sg.Button("Select center electrode on Voltage Map"),
         sg.Button("Reset")],
        [sg.Text("Patch Height"),
         sg.InputText(size=(3, 1), key="HEIGHT_PATCH", enable_events=True, default_text=str(patchHeight))],
        [sg.Text("Patch Width"),
         sg.InputText(size=(3, 1), key="WIDTH_PATCH", enable_events=True, default_text=str(patchWidth))],
        [sg.Text("Treshold"),
         sg.InputText(size=(3, 1), key="TRESHOLD", enable_events=True, default_text=str(threshold))],
        [sg.Checkbox('Route Mask Borders', enable_events=True, key='REVERSE_CONF', default=reverseConfBool)],
        [sg.Button("Plot Rectangle Electrode Selection", key="PLOT_RECTANGLE_SELECTION"),
        sg.Button("Load Into Scope")]
    ]

    tab1_layout = [
        [sg.Text("Welcome to the Neurolyzer GUI", justification="c", expand_x=True)],
        [sg.Text("This GUI will allow you to run through all the neccesary steps to successfully record & stimulate your beautiful cultures!", justification="c", expand_x=True)],
        [sg.Column(data_column),
         sg.VSeperator(),
         sg.Column(vmap_column),
         sg.VSeperator(),
         sg.Column(selection_column)]
    ]

###############################################RecordingTab#######################################################################

    record_column = [
        [sg.Text("Here you can record spontaneous spiking activity over a selected part of the array and\ngenerate the necessary files for further analysis.")],
        [sg.Text("")],
        [sg.Text("Declare the recording settings below:")],
        [sg.Text("Seconds"),sg.Slider(range=(0, 59), orientation='h', size=(20, 10), default_value=0, enable_events=True, key="SECONDS")],
        [sg.Text("Minutes"),sg.Slider(range=(0, 59), orientation='h', size=(20, 10), default_value=1, enable_events=True, key="MINUTES")],
        [sg.Text("Hours"),sg.Slider(range=(0, 96), orientation='h', size=(20, 10), default_value=0, enable_events=True, key="HOURS")],
        [sg.Text("The settings below will apply to all recording functionalities on this tab")],
        [sg.Checkbox('Use Legacy Format', enable_events=True, key='LEGACY_FORMAT', default=legacyFormatBool),
         sg.Checkbox('Only Record Spikes', enable_events=True, key='RECORD_SPIKES', default=onlySpikesBool),
         sg.Checkbox('Store as Analyzor Object', enable_events=True, key='STORE_ANALYZOR', default=storeAnalyzorBool)],
        [sg.Text("")],
        [sg.Text("If you have already chosen a electrode selection on the previous tab, we can proceed to record from it here.")],
        [sg.Text("Status: "),
        sg.Text("Scope is not ready yet, please load an electrode selection!", key="SCOPE_READY_TEXT")],
        [sg.Button("Record")],
        [sg.Text("")],
        [sg.Text("It's also possible to predefine multiple electrode selection first and then run\nrecordings on all of them consecutively.")],
        [sg.Text("Nr. of networks to record from:"),
         sg.InputText(size=(5, 1), key="NR_OF_RECORDINGS", enable_events=True, default_text=str(6))],
        [sg.Radio("Define Patches via Center Electrode", "RADIO1", key="RECORD_ALL_SWITCH",default=True)],
        [sg.Radio("Use preexisting Electrode Selections", "RADIO1", default=False)],
        [sg.Button("Record All")]
    ]

    stimulation_column = [
        [sg.Text("We can also select electrodes from those loaded into Scope as stimulation\nelectrodes and run stimultion experiments")],
        [sg.Text("")],
        [sg.Text("Stimulation Settings:")],
        [sg.Text("Repeat Complete Runthrough"),
         sg.InputText(size=(5,1),key="TOTAL_LOOPS",enable_events=True,default_text=str(nrOfTotalLoops)),
         sg.Text("times")],
        [sg.Text("Approximate Duration of Stimulation:"),
        sg.InputText(size=(5, 1), key="ITERATION_DURATION", enable_events=True, default_text=str(stimulationIterationDuration)),
         sg.Combo(unitListLarge,default_value='s',key="DROP_2", enable_events=True)],
        [sg.Text("No Stim Time Before:"),
         sg.InputText(size=(5,1),key="BURN_IN",enable_events=True,default_text=str(stimulationBurnInDuration)),
         sg.Combo(unitListLarge,default_value='min',key="DROP_1", enable_events=True)],
        [sg.Text("No Stim Time After"),
         sg.InputText(size=(5, 1), key="BURN_OUT", enable_events=True, default_text=str(stimulationBurnOutDuration)),
         sg.Combo(unitListLarge,default_value='min',key="DROP_3", enable_events=True)],
        [sg.Text("The Experiment will take {}{}.".format(stimulationDuration,'s'),size=(40, 1),key="STIMULUS_DURATION")],
        [sg.Text("")],
        [sg.Text("Selected Electrodes:")],
        [sg.Listbox(values=stimulationElectrodesList, enable_events=True, size=(60, 10), key="STIMULATION_LIST")],
        [sg.Button("New Stimulation Electrode"),
         sg.ReadFormButton("Edit",bind_return_key=False, key="EDIT"),
         sg.Button("Delete")],
        [sg.Text("Configure Stimulation Electrode:")],
        [sg.Button("Select Electrode"),
         sg.InputText(size=(5, 1), disabled=True, key="ELECTRODE_NUMBER", enable_events=True, default_text=str(defaultElectrode))],
        [sg.Text("Inter Peak Interval:"),
         sg.InputText(size=(5, 1), disabled=True, key="INTER_PEAK_INTERVAL", enable_events=True, default_text=str(noText)), 
         sg.Combo(unitListSmallShort,disabled=True,default_value='ms',key="DROP_5", enable_events=True)],
        [sg.Text("Offset:"), sg.InputText(size=(5, 1), disabled=True, key="OFFSET", enable_events=True, default_text=str(noText)),
         sg.Combo(unitListSmallShort,disabled=True,default_value='µs',key="DROP_6", enable_events=True)],
        [sg.Text("")],
        [sg.Checkbox('Record Non-Stim', enable_events=True, key='RECORD_BURN', default=recordBurnTimeBool),
         sg.Checkbox('Record First Stim Sequences', enable_events=True, key='RECORD_FIRST_STIMS', default=recordLastStimsBool)],
        [sg.Checkbox('Record n Last Stim Sequences', enable_events=True, key='RECORD_LAST_STIMS', default=recordLastStimsBool),
         sg.Text("n="),
         sg.InputText(size=(2, 1), disabled=False, key="NR_OF_LAST_SEQUENCES", enable_events=True, default_text=str(nrOfLastSequences))],
        [sg.Text("Pulse Duration:"),
         sg.InputText(size=(5, 1), disabled=False, key="PULSE_DURATION", enable_events=True, default_text=str(pulseDuration*(1000**2))),
         sg.Text("µs"),
         sg.Text("Amplitude:"),
         sg.InputText(size=(3, 1), disabled=False, key="AMPLITUDE", enable_events=True, default_text=str(stimulusAmplitude)),
         sg.Text("mV")],
        [sg.Text("")],
        [sg.Button("Record Stimulation")],
    ]

    tab2_layout = [[sg.Column(record_column),
                    sg.VSeperator(),
                    sg.Column(stimulation_column)]
    ]

###############################################AnalysisTab##################################################################################

    quick_analysis = [
        [sg.Text("We can load recorded files or extracted Analyzor Objects and run some\nsimple analysis on them to get a feeling for the data.")],
        [sg.Text("")],
        [sg.Text("Available Recordings:")],
        [sg.Listbox(values=recordings_list, enable_events=True, size=(60, 5), key="RECORDING_LIST")],
        [sg.Button("Filter by Chip Number", key='FILTER_RECORDINGS'),
         sg.Button("Refresh & Show all", key='RESET_RECORDINGS')],
        [sg.Text("If raw data was recorded, we can extract more accurate spikedata using a\ncustom spike detection algorithm. This can take a while, so we store the\nAnalyzor Object afterwards for easier further use.")],
        [sg.Text("Spike Detection and Filter Parameters:")],
        [sg.Text("Cutoff Frequency"),
         sg.InputText(size=(5, 1), key="CUTOFF_FREQ", enable_events=True, default_text=str(cutOffFrequency))],
        [sg.Text("Filter Order"),
         sg.InputText(size=(5, 1), key="FILTER_ORDER", enable_events=True, default_text=str(filterOrder))],
        [sg.Text("Spike Distance"),
         sg.InputText(size=(5, 1), key="SPIKE_DISTANCE", enable_events=True, default_text=str(spikeDistance))],
        [sg.Button("Run Spike Detection & Store Analyzor"),
         sg.Button("Store Analyzor")],
        [sg.Text("")],
        [sg.Text("Available Analyzor Objects:")],
        [sg.Listbox(values=analyzor_list, enable_events=True, size=(60, 5), key="ANALYZOR_LIST")],
        [sg.Button("Filter by Chip Number", key='FILTER_ANALYZOR'),
         sg.Button("Refresh & Show all", key='RESET_ANALYZOR')],
        [sg.Text("")],
        [sg.Text("Status: No Data File selected", key="FILE_STATUS")],
        [sg.Text("Quick analysis based on last selected recording or analyzor object.\nThis will show the Mean Firing Rate and Maximal Absolute Amplitude\nover all selected electrodes.")],
        [sg.Button("Quick Analysis")]
    ]

    spontaneous_analysis = [
        [sg.Text("For spontaneous spiking data, we can select an active electrode\nas our trigger electrode and use it to generate raster plots.\n(Most active electrode per default)", key="TRIG_EL_SEL")],
        [sg.Text("")],
        [sg.Text("Select Trigger Electrode:"),
         sg.InputText(size=(5, 1), key="TRIGGER_ELECTRODE", enable_events=True)],
        [sg.Button("Use most active electrode", key="USE_BEST_ELEC_BUTTON"),
         sg.Button("Next")],
        [sg.Text("")],
        [sg.Text("If needed, a custom subsampling algorithm can be used to filter\nelectrodes and improve visualisation.")],
        [sg.Text("Subsampling Parameters:")],
        [sg.Text("Minimal Distance [pixels]"),
         sg.InputText(size=(5, 1), key="MINIMAL_DISTANCE", enable_events=True, default_text=str(minimalDistance))],
        [sg.Text("Point Limit:"),
         sg.InputText(size=(5, 1), key="POINT_LIMIT", enable_events=True, default_text=str(pointLimit))],
        [sg.Button("Show and apply subsampling & Circular color coding")],
        [sg.Text("")],
        [sg.Text("Raster Plot Parameters:")],
        [sg.Text("Window Size [ms]"),
         sg.InputText(size=(5, 1), key="WINDOW_SIZE", enable_events=True, default_text=str(windowSize))],
        [sg.Text("Pre Window [ms]"),
         sg.InputText(size=(5, 1), key="PRE_WINDOW", enable_events=True, default_text=str(preWindow))],
        [sg.Button("Generate Raster Plot")],
    ]

    tab3_layout = [[sg.Column(quick_analysis),
                    sg.VSeperator(), 
                    sg.Column(spontaneous_analysis),]
    ]

###############################################WindowLayout#################################################################################

    tabgrp = [[sg.TabGroup([[
        sg.Tab('Chip Preparation', tab1_layout),
        sg.Tab('Recording & Stimulation', tab2_layout),
        sg.Tab('Data Analysis', tab3_layout)
        ]], 
        tab_location = 'centertop')
    ]]
    window = sg.Window("Neurolyzer", tabgrp, finalize=True, font=textFont)

####################################################################################################################################################
###########################################################################Loop#####################################################################

    while True:
        plt.close("all")
        if len(coords) != 0:
            centerElectrodeAxisX = coords[0]
            centerElectrodeAxisY = coords[1]
            coords = []

        event, values = window.read()

#######################################################GeneralFunctionality################################################################

        if (event == sg.WIN_CLOSED):
            window.close()
            sys.exit()

#######################################################################################################################
#################################################### TAB 1 ############################################################
#######################################################################################################################

#################################################### Data Column ######################################################

        # Show the currently chosen storage name based on all provided parameters 
        elif event == "Show Current Name":
            try:
                currentTime = time.localtime()
                if currentTime[1] < 10:
                    currentMonth = '0{}'.format(currentTime[1])
                else:
                    currentMonth = currentTime[1]
                if currentTime[2] < 10:
                    currentDay = '0{}'.format(currentTime[2])
                else:
                    currentDay = currentTime[2]

                fname = "ID{}_NW{}_DIV{}_DATE{}{}{}".format(values['CHIP_ID'], values['NETWORK_NUMBER'], values['DIV'], str(currentTime[0]),str(currentMonth),str(currentDay))
                if values["CUSTOM_LABEL_NAME_1"] and values["CUSTOM_LABEL_VALUE_1"]:
                    fname += "_{}{}".format(values["CUSTOM_LABEL_NAME_1"], values["CUSTOM_LABEL_VALUE_1"])
                if values["CUSTOM_LABEL_NAME_2"] and values["CUSTOM_LABEL_VALUE_2"]:
                    fname += "_{}{}".format(values["CUSTOM_LABEL_NAME_2"], values["CUSTOM_LABEL_VALUE_2"])
                if values["CUSTOM_LABEL_NAME_3"] and values["CUSTOM_LABEL_VALUE_3"]:
                    fname += "_{}{}".format(values["CUSTOM_LABEL_NAME_3"], values["CUSTOM_LABEL_VALUE_3"])

                window["FILE_NAME"].update(fname) 
            except:
                pass

#################################################### Vmap Column ######################################################

        # Update the currently loaded vmap based on the new selection
        elif event == "FILE_LIST":
            try:
                vmapFile = values["FILE_LIST"][0]
                VMArray = np.load(os.path.join(voltageMapsPath, vmapFile))
            except:
                pass
        # Filter the voltage list by the chip number
        elif event == "Filter by Chip Number":
            try:
                file_list = os.listdir(voltageMapsPath)
                file_list = [s for s in file_list if str(values["CHIP_ID"]) in s]
                window["FILE_LIST"].update(file_list) 
            except:
                pass
        # Reset the voltage list filter
        elif event == "RESET_FILTER":
            try:
                file_list = os.listdir(voltageMapsPath)
                window["FILE_LIST"].update(file_list) 
            except:
                pass
        # Create a new voltage map for the chip in the system
        elif event == "Create New Voltage Map":
            try:
                currentTime = time.localtime()
                if currentTime[1] < 10:
                    currentMonth = '0{}'.format(currentTime[1])
                else:
                    currentMonth = currentTime[1]
                if currentTime[2] < 10:
                    currentDay = '0{}'.format(currentTime[2])
                else:
                    currentDay = currentTime[2]

                fname = "{}_{}_{}_{}".format(str(currentTime[0]),str(currentMonth),str(currentDay), values['CHIP_ID'])
                getVoltageMap(fname, voltageMapsPath)
                file_list = os.listdir(voltageMapsPath)
                window["FILE_LIST"].update(file_list) 
            except:
                pass

        # Show the currently loaded voltage map
        elif event == "Plot Voltage Map":
            try:
                if not (VMArray != 0).any():
                    print("No Voltage Map Selected")
                    continue
                fig, ax = plt.subplots()
                ax.matshow(VMArray, cmap=vmap_cmap)
                plt.show()
            except:
                pass

#################################################### Selection Column ###################################################

        # Update the currently loaded electrode selection
        elif event == "SELECTION_LIST":
            try:
                elecSelIDFile = values["SELECTION_LIST"][0]
                elecSelection = np.load(os.path.join(electrodeSelectionPaths, elecSelIDFile))
            except:
                pass
        # Filter the list of available electrode selection by chip number
        elif event == "FILTER_ELECTRODE_SELECTION":
            try:
                selection_list = os.listdir(electrodeSelectionPaths)
                selection_list = [s for s in selection_list if str(values["CHIP_ID"]) in s]
                window["SELECTION_LIST"].update(selection_list) 
            except:
                pass
        # Reset the electrode selection list filter
        elif event == "RESET_ELECTRODE_FILTER":
            try:
                selection_list = os.listdir(electrodeSelectionPaths)
                window["SELECTION_LIST"].update(selection_list)  
            except:
                pass
        # Plot the selected electrode selection mulitplied by the underlying voltage map
        elif event == "Plot Electrode Selection":
            try:
                if not (VMArray != 0).any():
                    print("No Voltage Map selected")
                mask = mask_from_electrodeSelection(elecSelection, VMArray)
                if reverseConfBool:
                    kernel = np.zeros([reverseConfKernelSize,reverseConfKernelSize]) + 1
                    convolvedMask= convolve(mask,kernel,cval=0,mode='constant')
                    emptySpaceIndices = convolvedMask == 0
                    mask[emptySpaceIndices] = 1
                    mask = np.abs(mask - 1)

                fig, ax = plt.subplots()
                ax.matshow(mask, cmap=vmap_cmap)
                plt.show()
            except:
                print("No Electrode Selection selected")
        # Run the electrode selection code based on the selected voltage map and store and load it afterwards
        elif event == "Create New Electrode Selection":
            try:
                if not (VMArray != 0).any():
                    print("No Voltage Map selected")
                vmapFile = values["FILE_LIST"][0]
                vmap_path = os.path.join(voltageMapsPath, vmapFile)
                elecSelection = getElectrodeSelection(vmap_path, electrodeSelectionPaths, vmapFile, values["NETWORK_NUMBER"])
                selection_list = os.listdir(electrodeSelectionPaths)
                selection_list = [s for s in selection_list if str(values["CHIP_ID"]) in s]
                window["SELECTION_LIST"].update(selection_list)
            except:
                pass
        # Load the selected electrode selection into scope
        elif event == "Load Selection Into Scope":
            try:
                mask = mask_from_electrodeSelection(elecSelection, VMArray)
                if not reverseConfBool:
                    elecSelection = np.argwhere(mask.flatten())
                else:
                    kernel = np.zeros([reverseConfKernelSize,reverseConfKernelSize]) + 1
                    convolvedMask = convolve(mask,kernel,cval=0,mode='constant')
                    emptySpaceIndices = convolvedMask == 0
                    mask[emptySpaceIndices] = 1
                    elecSelection = np.argwhere(mask.flatten() == 0)

                elecSelection = list(np.squeeze(elecSelection))
                loadConfIntoScope(elecSelection)
                loadedScopeBool = True
                window["SCOPE_READY_TEXT"].update("Scope is ready for recording!")
            except:
                pass
        # Reset the center electrodes for the selection rectangle to the default values
        elif event == "Reset":
            try:
                window["ELECTRODE_X"].update(VMArray.shape[1])
                centerElectrodeAxisX = VMArray.shape[1]
                window["ELECTRODE_Y"].update(VMArray.shape[0])
                centerElectrodeAxisY = VMArray.shape[0]
            except:
                pass
        # Flip the bool
        elif event == "REVERSE_CONF":
            try:
                reverseConfBool = not reverseConfBool
            except:
                pass
        # Open an interactive plot of the selected voltage map to chose the new center electrodes 
        elif event == "Select center electrode on Voltage Map":
            try:
                if not (VMArray != 0).any():
                    print("No Voltage Map Selected")
                def selectCoordinates(event):
                    ix, iy = event.xdata, event.ydata
                    window["ELECTRODE_X"].update(int(ix))
                    coords.append(int(ix))
                    window["ELECTRODE_Y"].update(int(iy))
                    coords.append(int(iy))
                    fig.canvas.mpl_disconnect(cid)
                    plt.close()
                    return
                fig, ax = plt.subplots()
                ax.matshow(VMArray, cmap=vmap_cmap)
                cid = fig.canvas.mpl_connect('button_press_event', selectCoordinates)
                plt.show()
            except:
                pass
        # Plot the electrode selection generated via rectangle selection
        elif event == "PLOT_RECTANGLE_SELECTION":
            try:
                mask = cutout_mask(int(values["ELECTRODE_X"]), int(values["ELECTRODE_Y"]), int(values["HEIGHT_PATCH"]), int(values["WIDTH_PATCH"]), VMArray)
                above_threshold_indices = mask > int(values["TRESHOLD"])
                maskThresh = np.zeros(mask.shape)
                maskThresh[above_threshold_indices] = 1
                if reverseConfBool:
                    kernel = np.zeros([reverseConfKernelSize,reverseConfKernelSize]) + 1
                    convolvedMaskThresh = convolve(maskThresh,kernel,cval=0,mode='constant')
                    emptySpaceIndices = convolvedMaskThresh == 0
                    maskThresh[emptySpaceIndices] = 1
                    maskThresh = np.abs(maskThresh - 1)

                maskThresh = maskThresh*VMArray

                fig, ax = plt.subplots()
                ax.matshow(maskThresh, cmap=vmap_cmap)

                plt.show()
            except:
                pass
        # Load the rectangle electrode selection into scope
        elif event == "Load Into Scope":
            try:
                mask = cutout_mask(int(values["ELECTRODE_X"]), int(values["ELECTRODE_Y"]), int(values["HEIGHT_PATCH"]), int(values["WIDTH_PATCH"]), VMArray)
                above_threshold_indices = mask > int(values["TRESHOLD"])
                maskThresh = np.zeros(mask.shape)
                maskThresh[above_threshold_indices] = 1
                if reverseConfBool:
                    kernel = np.zeros([reverseConfKernelSize,reverseConfKernelSize]) + 1
                    convolvedMaskThresh = convolve(maskThresh,kernel,cval=0,mode='constant')
                    emptySpaceIndices = convolvedMaskThresh == 0
                    maskThresh[emptySpaceIndices] = 1
                    elecSelection = np.argwhere(maskThresh.flatten() == 0)
                else:
                    elecSelection = np.argwhere(maskThresh.flatten())

                elecSelection = list(np.squeeze(elecSelection))
                loadConfIntoScope(elecSelection)
                loadedScopeBool = True
                window["SCOPE_READY_TEXT"].update("Scope is ready for recording")
            except:
                pass

#######################################################################################################################
#################################################### TAB 2 ############################################################
#######################################################################################################################

#################################################RecordingColumn####################################################################

        # Run a normal recording for the electrode selection loaded into scope
        elif event=="Record":
            try:
                currentTime = time.localtime()
                if currentTime[1] < 10:
                    currentMonth = '0{}'.format(currentTime[1])
                else:
                    currentMonth = currentTime[1]
                if currentTime[2] < 10:
                    currentDay = '0{}'.format(currentTime[2])
                else:
                    currentDay = currentTime[2]
                reverseConfString = ''
                if reverseConfBool:
                    reverseConfString = "_Borders"
                if loadedScopeBool:
                    fname = "ID{}_NW{}_DIV{}_DATE{}{}{}".format(values['CHIP_ID'], values['NETWORK_NUMBER'], values['DIV'], str(currentTime[0]),str(currentMonth),str(currentDay))
                    if values["CUSTOM_LABEL_NAME_1"] and values["CUSTOM_LABEL_VALUE_1"]:
                        fname += "_{}{}".format(values["CUSTOM_LABEL_NAME_1"], values["CUSTOM_LABEL_VALUE_1"])
                    if values["CUSTOM_LABEL_NAME_2"] and values["CUSTOM_LABEL_VALUE_2"]:
                        fname += "_{}{}".format(values["CUSTOM_LABEL_NAME_2"], values["CUSTOM_LABEL_VALUE_2"])
                    if values["CUSTOM_LABEL_NAME_3"] and values["CUSTOM_LABEL_VALUE_3"]:
                        fname += "_{}{}".format(values["CUSTOM_LABEL_NAME_3"], values["CUSTOM_LABEL_VALUE_3"])
                    fname += reverseConfString
                    window["FILE_NAME"].update(fname) 
                    recordFileName = fname
                    recordDuration = int(values["SECONDS"] + values["MINUTES"] * 60 + values["HOURS"] * 3600)
                    recordInScope(str(values["LOCATION"]), recordFileName, recordDuration, legacyFormatBool, onlySpikesBool)

                    if storeAnalyzorBool:
                        recordFileName += ".raw.h5"
                        analyzorObject = Analyzor.AnalyZor(recordFileName, inputPath=str(values["LOCATION"]))
                        analyzorObject.loadDataSpikesOnly(blankingBool=False)
                        storage_name, _, _ = recordFileName.rsplit(".", 2)
                        storage_name = os.path.join(analyzorObjectPaths ,storage_name + ".pickle")
                        with open(storage_name, "wb") as f:
                            pickle.dump(analyzorObject, f)
                    print("Recording Done!")
                else:
                    print("Load Configuration into Scope.")
            except:
                pass
        # Flip the bools for the recording parameters
        elif event == "LEGACY_FORMAT":
            legacyFormatBool = not legacyFormatBool

        elif event == "RECORD_SPIKES":
            onlySpikesBool = not onlySpikesBool

        elif event == "STORE_ANALYZOR":
            storeAnalyzorBool = not storeAnalyzorBool
            
        # Run a selected number of consecutive recordings for electrode selections chosen from a list or defined via center electrodes
        elif event=="Record All":
            try:
                PATCH_NUM = int(values["NR_OF_RECORDINGS"])
                if values["RECORD_ALL_SWITCH"]:
                    if not (VMArray != 0).any():
                        print("No Voltage Map Selected")
                        continue
                    record_all_coords = []
                    def selectCoordinatesAllRec(event):
                        ix, iy = event.xdata, event.ydata
                        record_all_coords.append((int(ix), int(iy)))
                        fig.canvas.mpl_disconnect(cid)
                        plt.close()
                        return
                    for patch_id in range(PATCH_NUM):
                        # select
                        fig, ax = plt.subplots()
                        ax.matshow(VMArray, cmap=vmap_cmap)
                        cid = fig.canvas.mpl_connect('button_press_event', selectCoordinatesAllRec)
                        plt.show()
                    print(f"Start Recording from {record_all_coords}")
                else:
                    elecSelection_List = []
                    selection_list_ra = os.listdir(electrodeSelectionPaths)
                    for patch_id in range(PATCH_NUM):
                        layout = [
                            [sg.Text('Select a preexisting Electrode Selection:')],
                            [sg.Listbox(values=selection_list_ra, enable_events=True, size=(60, 10), key="ELECTRODE_SELECTION_LIST")],
                            [sg.Button("Filter by Chip Number", key='FILTER_ELECTRODE_SELECTION_RA'),
                            sg.Button("Refresh & Show all", key='RESET_ELECTRODE_FILTER_RA'),
                            sg.Button('Confirm', key='ELECTRODE_SELECTION_CHOICE_CONFIRM')]
                        ]
                        small_window = sg.Window('Select Electrode Selection', layout, finalize=True)

                        while True:
                            event, values_small = small_window.read()
                            if event == sg.WIN_CLOSED:
                                small_window.close()
                                break
                            elif event == "FILTER_ELECTRODE_SELECTION_RA":
                                try:
                                    selection_list_ra = os.listdir(electrodeSelectionPaths)
                                    selection_list_ra = [s for s in selection_list_ra if str(values["CHIP_ID"]) in s]
                                    small_window["ELECTRODE_SELECTION_LIST"].update(selection_list_ra) 
                                except:
                                    pass
                            elif event == "RESET_ELECTRODE_FILTER_RA":
                                try:
                                    selection_list_ra = os.listdir(electrodeSelectionPaths)
                                    small_window["ELECTRODE_SELECTION_LIST"].update(selection_list_ra)  
                                except:
                                    pass
                            elif event == "ELECTRODE_SELECTION_CHOICE_CONFIRM":
                                elecSelection_List.append(values_small["ELECTRODE_SELECTION_LIST"])
                                small_window.close()
                                break

                # independently determine file name
                currentTime = time.localtime()
                if currentTime[1] < 10:
                    currentMonth = '0{}'.format(currentTime[1])
                else:
                    currentMonth = currentTime[1]
                if currentTime[2] < 10:
                    currentDay = '0{}'.format(currentTime[2])
                else:
                    currentDay = currentTime[2]
                reverseConfString = ''
                if reverseConfBool:
                    reverseConfString = "_Borders"

                for patch_id in range(PATCH_NUM):
                    # load into scope
                    if values["RECORD_ALL_SWITCH"]:
                        centerElectrodeAxisX, centerElectrodeAxisY = record_all_coords[patch_id]
                        mask = cutout_mask(int(centerElectrodeAxisX), int(centerElectrodeAxisY), int(values["HEIGHT_PATCH"]), int(values["WIDTH_PATCH"]), VMArray)
                        if not reverseConfBool:
                            elecSelection = np.argwhere(mask.flatten() > int(values["TRESHOLD"]))
                        else:
                            above_threshold_indices = mask > int(values["TRESHOLD"])

                            if above_threshold_indices.shape[0] > 1024:
                                mask_vals = mask[above_threshold_indices]
                                above_threshold_indices = above_threshold_indices[np.flip(np.argsort(mask_vals))[:1024]]

                            maskThresh = np.zeros(mask.shape)
                            maskThresh[above_threshold_indices] = 1
                            kernel = np.zeros([reverseConfKernelSize,reverseConfKernelSize]) + 1
                            convolvedMaskThresh = convolve(maskThresh,kernel,cval=0,mode='constant')
                            emptySpaceIndices = convolvedMaskThresh == 0
                            maskThresh[emptySpaceIndices] = 1
                            elecSelection = np.argwhere(maskThresh.flatten() == 0)
                        elecSelection = list(np.squeeze(elecSelection))
                    else:
                        elecSelIDFile = elecSelection_List[patch_id][0]
                        elecSelection= np.load(os.path.join(electrodeSelectionPaths, elecSelIDFile))
                    
                    print("Recording patch {}".format(patch_id))
                    loadConfIntoScope(elecSelection)

                    # record
                    fname = "ID{}_NW{}_DIV{}_DATE{}{}{}".format(values['CHIP_ID'], patch_id, values['DIV'], str(currentTime[0]),str(currentMonth),str(currentDay))
                    if values["CUSTOM_LABEL_NAME_1"] and values["CUSTOM_LABEL_VALUE_1"]:
                        fname += "_{}{}".format(values["CUSTOM_LABEL_NAME_1"], values["CUSTOM_LABEL_VALUE_1"])
                    if values["CUSTOM_LABEL_NAME_2"] and values["CUSTOM_LABEL_VALUE_2"]:
                        fname += "_{}{}".format(values["CUSTOM_LABEL_NAME_2"], values["CUSTOM_LABEL_VALUE_2"])
                    if values["CUSTOM_LABEL_NAME_3"] and values["CUSTOM_LABEL_VALUE_3"]:
                        fname += "_{}{}".format(values["CUSTOM_LABEL_NAME_3"], values["CUSTOM_LABEL_VALUE_3"])
                    fname += reverseConfString
                    window["FILE_NAME"].update(fname) 
                    recordFileName = fname

                    print("Will be saved as {}".format(recordFileName))
                    recordDuration = int(values["SECONDS"] + values["MINUTES"] * 60 + values["HOURS"] * 3600)
                    recordInScope(str(values["LOCATION"]), recordFileName, recordDuration, legacyFormatBool, onlySpikesBool)

                    try:
                        if storeAnalyzorBool:
                            recordFileName += ".raw.h5"
                            analyzorObject = Analyzor.AnalyZor(recordFileName, inputPath=str(values["LOCATION"]))
                            analyzorObject.loadDataSpikesOnly(blankingBool=False)
                            storage_name, _, _ = recordFileName.rsplit(".", 2)
                            storage_name = os.path.join(analyzorObjectPaths ,storage_name + ".pickle")
                            with open(storage_name, "wb") as f:
                                pickle.dump(analyzorObject, f)
                    except:
                        print("No spikes were recorded, cannot generate an Analyzor Object!")
                print("Recording Done!")
            except:
                pass

############################################StimulationColumn########################################################

        # Update the stimulus parameters & sequence based on the user selection
        elif event in ["TOTAL_LOOPS","BURN_IN","DROP_1","ITERATION_DURATION","DROP_2","BURN_OUT", "DROP_3"]:
            if isfloat(values["TOTAL_LOOPS"]):
                nrOfTotalLoops = int(values["TOTAL_LOOPS"])
            else:
                nrOfTotalLoops = 0
            if isfloat(values["BURN_IN"]):
                stimulationBurnInDuration = durationInSeconds(float(values["BURN_IN"]),values["DROP_1"],unitList)
            else:
                stimulationBurnInDuration = 0
            if isfloat(values["ITERATION_DURATION"]):
                stimulationIterationDuration = durationInSeconds(float(values["ITERATION_DURATION"]),values["DROP_2"],unitList)
            else:
                stimulationIterationDuration = 0
            if isfloat(values["BURN_OUT"]):
                stimulationBurnOutDuration = durationInSeconds(float(values["BURN_OUT"]),values["DROP_3"],unitList)
            else:
                stimulationBurnOutDuration = 0

            stimulationDuration = stimulusDuration(stimulationBurnInDuration,stimulationIterationDuration,stimulationBurnOutDuration,nrOfTotalLoops)
            stimulationDuration,stimulationDurationUnit = smallestTimeUnitConverter(stimulationDuration,unitListLarge)
            stimulationDurationString = "The Experiment will take {}{}.".format(stimulationDuration, stimulationDurationUnit)
            window["STIMULUS_DURATION"].update(stimulationDurationString)

        elif event=="RECORD_BURN":
            recordBurnTimeBool = not recordBurnTimeBool

        elif event == "RECORD_LAST_STIMS":
            recordLastStimsBool = not recordLastStimsBool

        elif event == 'RECORD_FIRST_STIMS':
            recordFirstStimsBool = not recordFirstStimsBool

        elif event == "PULSE_DURATION":
            if isfloat(values["PULSE_DURATION"]):
                duration = durationInSeconds(float(values["PULSE_DURATION"]), "µs", unitList)
                pulseDuration = round(duration,5)
            else:
                pulseDuration = 0

        elif event == "NR_OF_LAST_SEQUENCES":
            if isfloat(values["NR_OF_LAST_SEQUENCES"]):
                nrOfLastSequences = int(values["NR_OF_LAST_SEQUENCES"])
            else:
                nrOfLastSequences = 0

########################RecordingStimulation

        # Run and record a stimulation experiment on the currently loaded electrode selection and user parameters
        elif event=="Record Stimulation" and not stimulationEditBool:
            try:
                currentTime = time.localtime()
                if currentTime[1] < 10:
                    currentMonth = '0{}'.format(currentTime[1])
                else:
                    currentMonth = currentTime[1]
                if currentTime[2] < 10:
                    currentDay = '0{}'.format(currentTime[2])
                else:
                    currentDay = currentTime[2]

                # record
                fname = "ID{}_NW{}_DIV{}_DATE{}{}{}".format(values['CHIP_ID'], values['NETWORK_NUMBER'], values['DIV'], str(currentTime[0]),str(currentMonth),str(currentDay))
                if values["CUSTOM_LABEL_NAME_1"] and values["CUSTOM_LABEL_VALUE_1"]:
                    fname += "_{}{}".format(values["CUSTOM_LABEL_NAME_1"], values["CUSTOM_LABEL_VALUE_1"])
                if values["CUSTOM_LABEL_NAME_2"] and values["CUSTOM_LABEL_VALUE_2"]:
                    fname += "_{}{}".format(values["CUSTOM_LABEL_NAME_2"], values["CUSTOM_LABEL_VALUE_2"])
                if values["CUSTOM_LABEL_NAME_3"] and values["CUSTOM_LABEL_VALUE_3"]:
                    fname += "_{}{}".format(values["CUSTOM_LABEL_NAME_3"], values["CUSTOM_LABEL_VALUE_3"])
                fname += "_STIMV{}".format(int(float(values["AMPLITUDE"])))
                fname += "_STIMEL{}".format(str(stimulationElectrodesList[0]))
                fname += reverseConfString
                window["FILE_NAME"].update(fname) 
                recordFileName = fname
                print("Will be saved as {}".format(recordFileName))

                f = open(os.path.join(recordingsPath, recordFileName + ".txt"), "w+")
                string = "Burn In: " + str(stimulationBurnInDuration) + "s, " + "Burn Out: " + str(stimulationBurnOutDuration) + "s, Iteration Duration: " + str(nrStimulationLoops) + "*" + str(stimulationIterationDuration) + "s, Repeated for " + str(nrOfTotalLoops) + " times\n"
                f.write(string)
                for i in range(len(stimulationPatternList)):
                    j = stimulationPatternList[i]
                    string = (''.join(str(e)+", " for e in j.electrodeNumbers)+str(j.amplitude)+"mV, "+str(pulseDuration)+"s pulse, "+str(j.interPeakInterval*pulseDuration)+"s IPI,"+str(j.offset*pulseDuration)+"s offset\n")
                    f.write(string)
                f.close()

                if not (VMArray != 0).any():
                    print("No Voltage Map selected")
                    continue
                mask = mask_from_electrodeSelection(elecSelection, VMArray)
                if not reverseConfBool:
                    elecSelection = np.argwhere(mask.flatten())
                else:
                    kernel = np.zeros([reverseConfKernelSize,reverseConfKernelSize]) + 1
                    convolvedMask = convolve(mask,kernel,cval=0,mode='constant')
                    emptySpaceIndices = convolvedMask == 0
                    mask[emptySpaceIndices] = 1
                    elecSelection = np.argwhere(mask.flatten() == 0)

                elecSelection = list(np.squeeze(elecSelection))

                stimulationSequenceDuration = blockDuration(stimulationPatternList) * pulseDuration
                nrStimulationLoops = max(int(stimulationIterationDuration / stimulationSequenceDuration+0.5), 1)
                switchList = switchPattern(stimulationPatternList, stimulationElectrodesList, stimulationSequenceDuration / pulseDuration)

                stimulationSequenceDuration = np.sum(np.asarray([i[1] + int(i[1]==0)  for i in switchList]))*pulseDuration
                recordStimulation(switchList,stimulationBurnInDuration,stimulationBurnOutDuration, stimulationSequenceDuration, int(nrStimulationLoops), int(stimulusAmplitude), elecSelection, stimulationElectrodesList,str(values["LOCATION"]), recordFileName, legacyFormatBool, onlySpikesBool, recordBurnTimeBool, recordFirstStimsBool,recordLastStimsBool, nrOfTotalLoops, pulseDuration*sampleRate, nrOfLastSequences)
                print("Stimulation Done")
            except:
                pass

        elif event=="EDIT" and values["STIMULATION_LIST"] and not stimulationEditBool:
            stimulationPatternList[selectedPatternIndex].electrodeNumbers.remove(int(values["STIMULATION_LIST"][0]))
            stimulationPatternList.append(stimulationPatternClass(ID=str(values["CHIP_ID"]), amp=stimulationPatternList[0].amplitude, numb=int(values["STIMULATION_LIST"][0])))
            stimulationPatternList[-1].copyPattern(stimulationPatternList[selectedPatternIndex])
            selectedPatternIndex = len(stimulationPatternList) - 1

            window["EDIT"].update("Done")
            stimulationEditBool = True
            window["INTER_PEAK_INTERVAL"].update(disabled=False)
            window["OFFSET"].update(disabled=False)
            window["ELECTRODE_NUMBER"].update(disabled=False)
            window["STIMULATION_LIST"].update(disabled=True)
            window["PULSE_DURATION"].update(disabled=True)
            window["AMPLITUDE"].update(disabled=True)
            window["DROP_5"].update(disabled=False)
            window["DROP_6"].update(disabled=False)

############################StimulationElectrodesSettings

        # Add a new stimulation electrode
        elif event == "New Stimulation Electrode" and not stimulationEditBool:
            if len(stimulationPatternList) == 0:
                stimulationPatternList.append(stimulationPatternClass(ID=str(values["CHIP_ID"]), amp=stimulusAmplitude, numb=0))
            else:
                newElectrodeNumber = max(
                    [i.largestElectrodeNumber() % (VMArray.shape[0] * VMArray.shape[1] - 1) for i in
                     stimulationPatternList]) + 1
                stimulationPatternList.append(
                    stimulationPatternClass(ID=str(values["CHIP_ID"]), amp=stimulationPatternList[0].amplitude,
                                            numb=newElectrodeNumber))
            stimulationElectrodesList = []
            for i in stimulationPatternList:
                for j in i.electrodeNumbers:
                    stimulationElectrodesList.append(j)
            window["STIMULATION_LIST"].update(stimulationElectrodesList)
        # Deleted the selected stimulation electrode
        elif event == "Delete" and not stimulationEditBool:
            groupIndex = 0
            for i in range(len(stimulationPatternList)):
                if selectedStimulationElectrodeIndex < len(stimulationPatternList[i].electrodeNumbers) + groupIndex:
                    del stimulationPatternList[i].electrodeNumbers[selectedStimulationElectrodeIndex - groupIndex]
                    break
                groupIndex += len(stimulationPatternList[i].electrodeNumbers)

            for i in stimulationPatternList:
                if not i.electrodeNumbers:
                    stimulationPatternList.remove(i)
            stimulationElectrodesList = []
            for i in stimulationPatternList:
                for j in i.electrodeNumbers:
                    stimulationElectrodesList.append(j)
            window["STIMULATION_LIST"].update(stimulationElectrodesList)
        # Select an electrode from the currently loaded voltage map
        elif event == "Select Electrode" and stimulationEditBool:
            stimulationPatternList[selectedPatternIndex].selectCoordinate(
                stimulationPatternList[selectedPatternIndex].electrodeNumbers[0],
                cutout_mask_NoZeros_Treshed(centerElectrodeAxisX, centerElectrodeAxisY, VMArray, patchHeight,
                                            patchWidth, threshold), centerElectrodeAxisX, centerElectrodeAxisY,
                patchWidth, patchHeight)
            window["ELECTRODE_NUMBER"].update(stimulationPatternList[selectedPatternIndex].electrodeNumbers[0])
        # Update the electrode number based on user input
        elif event == "ELECTRODE_NUMBER" and stimulationEditBool:
            if isfloat(values["ELECTRODE_NUMBER"]):
                stimulationPatternList[selectedPatternIndex].electrodeNumbers[0] = int(values["ELECTRODE_NUMBER"])
            else:
                stimulationPatternList[selectedPatternIndex].electrodeNumbers[0] = 0
        # Update the stimulation amplitude based on user input
        elif event == "AMPLITUDE":
            if isfloat(values["AMPLITUDE"]):
                for i in stimulationPatternList:
                    i.setAmplitude(values["AMPLITUDE"])
                stimulusAmplitude = float(values["AMPLITUDE"])
            else:
                for i in stimulationPatternList:
                    i.setAmplitude(0)
                stimulusAmplitude = 0
        # Update the IPI based on user input
        elif (event == "INTER_PEAK_INTERVAL" or event == "DROP_5") and stimulationEditBool:
            if isfloat(values["INTER_PEAK_INTERVAL"]):
                duration = durationInSeconds(float(values["INTER_PEAK_INTERVAL"]), values["DROP_5"], unitList)
                stimulationPatternList[selectedPatternIndex].setInterPulseInterval(duration / pulseDuration)
            else:
                stimulationPatternList[selectedPatternIndex].setInterPulseInterval(0)
        # Update the offset based on user input
        elif (event == "OFFSET" or event == "DROP_6") and stimulationEditBool:
            if isfloat(values["OFFSET"]):
                duration = durationInSeconds(float(values["OFFSET"]), values["DROP_6"], unitList)
                stimulationPatternList[selectedPatternIndex].setOffset(duration / pulseDuration)
            else:
                stimulationPatternList[selectedPatternIndex].setOffset(0)
        # Enable parameter editing for the currently selected stimulation electrode 
        elif event=="EDIT" and stimulationEditBool:
            window["EDIT"].update("Edit")
            stimulationEditBool = False
            window["INTER_PEAK_INTERVAL"].update(disabled=True)
            window["OFFSET"].update(disabled=True)
            window["ELECTRODE_NUMBER"].update(disabled=True)
            window["STIMULATION_LIST"].update(disabled=False)
            window["PULSE_DURATION"].update(disabled=False)
            window["AMPLITUDE"].update(disabled=False)
            window["DROP_5"].update(disabled=True)
            window["DROP_6"].update(disabled=True)
            patternIndices = list(range(len(stimulationPatternList)))
            patternIndices.remove(selectedPatternIndex)
            for i in patternIndices:
                if stimulationPatternList[selectedPatternIndex].equal(stimulationPatternList[i]):
                    stimulationPatternList[i].appendElectrodeNumber(stimulationPatternList[selectedPatternIndex].electrodeNumbers)
                    del stimulationPatternList[selectedPatternIndex]
                    selectedPatternIndex = i
                    break

            for i in stimulationPatternList:
                if not i.electrodeNumbers:
                    stimulationPatternList.remove(i)
            stimulationElectrodesList = []
            for i in stimulationPatternList:
                for j in i.electrodeNumbers:
                    if not j in stimulationElectrodesList:
                        stimulationElectrodesList.append(j)
                    else:
                        i.electrodeNumbers.remove(j)
            window["STIMULATION_LIST"].update(stimulationElectrodesList)

        #This must be last
        elif values["STIMULATION_LIST"]:
            selectedStimulationElectrodeIndex = stimulationElectrodesList.index(values['STIMULATION_LIST'][0])
            groupIndex = 0
            for i in range(len(stimulationPatternList)):
                if selectedStimulationElectrodeIndex < len(stimulationPatternList[i].electrodeNumbers) + groupIndex:
                    selectedPatternIndex = i
                    break
                groupIndex += len(stimulationPatternList[i].electrodeNumbers)
            selectedStimulationPattern = stimulationPatternList[selectedPatternIndex]
            window["AMPLITUDE"].update(selectedStimulationPattern.amplitude)
            window["ELECTRODE_NUMBER"].update(values['STIMULATION_LIST'][0])
            length, unit = largestTimeUnitConverter(selectedStimulationPattern.interPeakInterval*pulseDuration, unitListSmallShort)
            window["INTER_PEAK_INTERVAL"].update(length)
            window["DROP_5"].update(unit)
            length, unit = largestTimeUnitConverter(selectedStimulationPattern.offset*pulseDuration, unitListSmallShort)
            window["OFFSET"].update(length)
            window["DROP_6"].update(unit)

#######################################################################################################################
#################################################### TAB 3 ############################################################
#######################################################################################################################

#################################################QuickAnalysisColumn####################################################################

        # Load the selected recording as a analyzor object and apply subsampling and color mapping, also update the trigger electrode 
        elif event == "RECORDING_LIST":
            recordingFile = values["RECORDING_LIST"][0]
            stimulationTXTString = recordingFile[: len(recordingFile) - 7] + ".txt"
            stimFileBool = check_if_stim_file(recordingsPath, stimulationTXTString)
            if stimFileBool:
                analyzorObject = Analyzor.AnalyZor(recordingFile, inputPath=recordingsPath, auto_parse=True)  
                analyzorObject.loadDataSpikesOnly(blankingBool=True, blankingWindow=[-10, 10])
                window["FILE_STATUS"].update("A stimulation data file has been selected!")
                window["TRIG_EL_SEL"].update("For stimulated spiking data, we can select a stimulation electrode\nas our trigger electrode and use it to generate raster plots.\n(First stimulation electrode per default)")
                window["USE_BEST_ELEC_BUTTON"].update("Use first stimulation electrode")
            else:
                analyzorObject = Analyzor.AnalyZor(recordingFile, inputPath=recordingsPath)
                analyzorObject.loadDataSpikesOnly(blankingBool=False)
                window["FILE_STATUS"].update("A recording data file has been selected!")
                window["TRIG_EL_SEL"].update("For spontaneous spiking data, we can select an active electrode\nas our trigger electrode and use it to generate raster plots.\n(Most active electrode per default)")
                window["USE_BEST_ELEC_BUTTON"].update("Use most active electrode")

            sampledElectrodes, sampledCoords = analyzorObject.electrode_subsampling(min_dist=0, point_limit=1024, figureSize=(9,7))
            coloredElectrodes, rgb_cycle, _ = analyzorObject.circular_color_coding(electrodes=sampledElectrodes, showPlotBool=False, dotSize=50)
            if stimFileBool:
                trigger_electrode = analyzorObject.stimulatedElectrodesList[0]
                window["TRIGGER_ELECTRODE"].update(trigger_electrode)
            else:
                trigger_electrode = analyzorObject.mostActiveElectrodes(1)
                window["TRIGGER_ELECTRODE"].update(trigger_electrode)

        # Filter recordings list by chip id (only show h5 files)
        elif event == "FILTER_RECORDINGS":
            try:
                recordings_list = os.listdir(recordingsPath)
                recordings_list = [s for s in recordings_list if "h5" in s]
                recordings_list = [s for s in recordings_list if str(values["CHIP_ID"]) in s]
                window["RECORDING_LIST"].update(recordings_list) 
            except:
                pass
        # Reset recording list filter
        elif event == "RESET_RECORDINGS":
            try:
                recordings_list = os.listdir(recordingsPath)
                recordings_list = [s for s in recordings_list if "h5" in s]
                window["RECORDING_LIST"].update(recordings_list)  
            except:
                pass
        # Load the selected analyzor object andd apply subsampling and color mapping, also update the trigger electrode 
        elif event == "ANALYZOR_LIST":
            try:
                analyzorObjectFile = values["ANALYZOR_LIST"][0]
                stimulationTXTString = analyzorObjectFile[: len(analyzorObjectFile) - 7] + ".txt"
                stimFileBool = check_if_stim_file(recordingsPath, stimulationTXTString)
                if stimFileBool:
                    window["FILE_STATUS"].update("A stimulation data file has been selected!")
                    window["TRIG_EL_SEL"].update("For stimulated spiking data, we can select a stimulation electrode\nas our trigger electrode and use it to generate raster plots.\n(First stimulation electrode per default)")
                    window["USE_BEST_ELEC_BUTTON"].update("Use first stimulation electrode")
                else:
                    window["FILE_STATUS"].update("A recording data file has been selected!")
                    window["TRIG_EL_SEL"].update("For spontaneous spiking data, we can select an active electrode\nas our trigger electrode and use it to generate raster plots.\n(Most active electrode per default)")
                    window["USE_BEST_ELEC_BUTTON"].update("Use most active electrode")
                with open(os.path.join(analyzorObjectPaths, analyzorObjectFile), "rb") as f:
                    analyzorObject = pickle.load(f)
                sampledElectrodes, sampledCoords = analyzorObject.electrode_subsampling(min_dist=0, point_limit=1024, figureSize=(9,7))
                coloredElectrodes, rgb_cycle, _ = analyzorObject.circular_color_coding(electrodes=sampledElectrodes, showPlotBool=False, dotSize=50)
                if stimFileBool:
                    trigger_electrode = analyzorObject.stimulatedElectrodesList[0]
                    window["TRIGGER_ELECTRODE"].update(trigger_electrode)
                else:
                    trigger_electrode = analyzorObject.mostActiveElectrodes(1)
                    window["TRIGGER_ELECTRODE"].update(trigger_electrode)
            except:
                pass
        # Filter the analyzor list by chip id
        elif event == "FILTER_ANALYZOR":
            try:
                analyzor_list = os.listdir(analyzorObjectPaths)
                analyzor_list = [s for s in analyzor_list if str(values["CHIP_ID"]) in s]
                window["ANALYZOR_LIST"].update(analyzor_list) 
            except:
                pass
        # Reset analyzor list filter
        elif event == "RESET_ANALYZOR":
            try:
                analyzor_list = os.listdir(analyzorObjectPaths)
                window["ANALYZOR_LIST"].update(analyzor_list)  
            except:
                pass
        # Run the custom spike detection algorithm for the selected recording file and store it as an analyzor object afterwards
        elif event == "Run Spike Detection & Store Analyzor":
            try:
                recordingFile = values["RECORDING_LIST"][0]
                stimulationTXTString = recordingFile[: len(recordingFile) - 7] + ".txt"
                stimFileBool = check_if_stim_file(recordingsPath, stimulationTXTString)
                if stimFileBool:
                    analyzorObject = Analyzor.AnalyZor(recordingFile, inputPath=recordingsPath, auto_parse=True)
                    analyzorObject.loadData(blankingBool=True, blankingWindow=[-10, 10], cutOffFrequency=int(values['CUTOFF_FREQ']), filterOrder=int(values['FILTER_ORDER']), spikeTraceBool=False, spikeDistance=int(values['SPIKE_DISTANCE']), use_sneo=False, scaleTraceMap=True, returnSpikeTrace=False, loadingSteps=10)               
                    window["FILE_STATUS"].update("A stimulation data file has been selected!")
                    window["TRIG_EL_SEL"].update("For stimulated spiking data, we can select a stimulation electrode\nas our trigger electrode and use it to generate raster plots.\n(First stimulation electrode per default)")
                    window["USE_BEST_ELEC_BUTTON"].update("Use first stimulation electrode")          
                else:
                    analyzorObject = Analyzor.AnalyZor(recordingFile, inputPath=recordingsPath)
                    analyzorObject.loadData(cutOffFrequency=int(values['CUTOFF_FREQ']), filterOrder=int(values['FILTER_ORDER']), spikeTraceBool=False, spikeDistance=int(values['SPIKE_DISTANCE']), use_sneo=False, scaleTraceMap=True, returnSpikeTrace=False, loadingSteps=10)               
                    window["FILE_STATUS"].update("A recording data file has been selected!")
                    window["TRIG_EL_SEL"].update("For spontaneous spiking data, we can select an active electrode\nas our trigger electrode and use it to generate raster plots.\n(Most active electrode per default)")
                    window["USE_BEST_ELEC_BUTTON"].update("Use most active electrode")
                storage_name, _, _ = recordingFile.rsplit(".", 2)
                storage_name = os.path.join(analyzorObjectPaths ,storage_name + ".pickle")
                with open(storage_name, "wb") as f:
                    pickle.dump(analyzorObject, f)
                sampledElectrodes, sampledCoords = analyzorObject.electrode_subsampling(min_dist=0, point_limit=1024, figureSize=(9,7))
                coloredElectrodes, rgb_cycle, _ = analyzorObject.circular_color_coding(electrodes=sampledElectrodes, showPlotBool=True, dotSize=50)
                if stimFileBool:
                    trigger_electrode = analyzorObject.stimulatedElectrodesList[0]
                    window["TRIGGER_ELECTRODE"].update(trigger_electrode)
                else:
                    trigger_electrode = analyzorObject.mostActiveElectrodes(1)
                    window["TRIGGER_ELECTRODE"].update(trigger_electrode)

                #Update the list
                analyzor_list = os.listdir(analyzorObjectPaths)
                analyzor_list = [s for s in analyzor_list if str(values["CHIP_ID"]) in s]
                window["ANALYZOR_LIST"].update(analyzor_list) 
            except:
                pass
        # Store the currently selected recording file as an a analyzor object (w/o custom spike detection)
        elif event == "Store Analyzor":
            try:
                recordingFile = values["RECORDING_LIST"][0]
                stimulationTXTString = recordingFile[: len(recordingFile) - 7] + ".txt"
                stimFileBool = check_if_stim_file(recordingsPath, stimulationTXTString)
                print(stimulationTXTString, stimFileBool)
                if stimFileBool:
                    analyzorObject = Analyzor.AnalyZor(recordingFile, inputPath=recordingsPath, auto_parse=True)
                    analyzorObject.loadDataSpikesOnly(blankingBool=True, blankingWindow=[-10, 10])
                    window["FILE_STATUS"].update("A stimulation data file has been selected!")
                    window["TRIG_EL_SEL"].update("For stimulated spiking data, we can select a stimulation electrode\nas our trigger electrode and use it to generate raster plots.\n(First stimulation electrode per default)")
                    window["USE_BEST_ELEC_BUTTON"].update("Use first stimulation electrode")
                else:
                    analyzorObject = Analyzor.AnalyZor(recordingFile, inputPath=recordingsPath)
                    analyzorObject.loadDataSpikesOnly(blankingBool=False)
                    window["FILE_STATUS"].update("A recording data file has been selected!")
                    window["TRIG_EL_SEL"].update("For spontaneous spiking data, we can select an active electrode\nas our trigger electrode and use it to generate raster plots.\n(Most active electrode per default)")
                    window["USE_BEST_ELEC_BUTTON"].update("Use most active electrode")
                sampledElectrodes, sampledCoords = analyzorObject.electrode_subsampling(min_dist=0, point_limit=1024, figureSize=(9,7))
                coloredElectrodes, rgb_cycle, _ = analyzorObject.circular_color_coding(electrodes=sampledElectrodes, showPlotBool=False, dotSize=50)
                if stimFileBool:
                    trigger_electrode = analyzorObject.stimulatedElectrodesList[0]
                    window["TRIGGER_ELECTRODE"].update(trigger_electrode)
                else:
                    trigger_electrode = analyzorObject.mostActiveElectrodes(1)
                    window["TRIGGER_ELECTRODE"].update(trigger_electrode)

                storage_name, _, _ = recordingFile.rsplit(".", 2)
                storage_name = os.path.join(analyzorObjectPaths ,storage_name + ".pickle")
                with open(storage_name, "wb") as f:
                    pickle.dump(analyzorObject, f)

                #Update the list
                analyzor_list = os.listdir(analyzorObjectPaths)
                analyzor_list = [s for s in analyzor_list if str(values["CHIP_ID"]) in s]
                window["ANALYZOR_LIST"].update(analyzor_list) 
            except:
                pass
        # Plots MFR and MAA for the most recently selected recording or analyzor file
        elif event == "Quick Analysis":
            try:
                boundx, boundy = analyzorObject.get_boundaries()

                freqMap = analyzorObject.frequencyHeatmap()
                freqMap[freqMap == 0] = np.nan

                ampMap = analyzorObject.maxAmplitudeHeatmap()
                ampMap *= (- analyzorObject.microVoltPerBit)
                ampMap[ampMap == 0] = np.nan

                fig, axes = plt.subplots(2,1, figsize=(16,9))
                try:
                    title = analyzorObjectFile.rsplit(".",2)[0]
                    fig.suptitle(title, fontsize=16)
                except:
                    title = recordingFile.rsplit(".",2)[0]
                    fig.suptitle(title, fontsize=16)

                freq_im = axes[0].imshow(freqMap[boundy[0]:boundy[1], boundx[0]:boundx[1]], cmap=quick_analysis_cmap)
                amp_im = axes[1].imshow(ampMap[boundy[0]:boundy[1], boundx[0]:boundx[1]], cmap=quick_analysis_cmap)

                axes[0].set_title('MFR heatmap')
                axes[1].set_title('Max. amplitude heatmap');

                fig.subplots_adjust(left=0.05)
                cbar_ax = fig.add_axes([0.05, 0.077, 0.02, 0.85])
                colorbar = fig.colorbar(freq_im, cax=cbar_ax, label = r"MFR [Hz]")
                fig.subplots_adjust(right=0.9)
                cbar_ax = fig.add_axes([.9, 0.077, 0.02, 0.85])
                colorbar = fig.colorbar(amp_im, cax=cbar_ax, label = r"Max. Abs. Amp [uV]")
                plt.show()
            except:
                pass
        # Select the most active electrode as the trigger electrode
        elif event == "USE_BEST_ELEC_BUTTON":          
            try:
                if stimFileBool:
                    trigger_electrode = analyzorObject.stimulatedElectrodesList[0]
                else:
                    trigger_electrode = analyzorObject.mostActiveElectrodes(1)
                window["TRIGGER_ELECTRODE"].update(trigger_electrode)
            except:
                pass
        # Select the next most active electrode behind the current one 
        elif event == "Next":
            try:
                if stimFileBool:
                    temp = np.array(analyzorObject.stimulatedElectrodesList)
                    l = len(temp)
                    ind = np.argwhere(temp == trigger_electrode)[0][0] + 1
                    if ind == l:
                        ind = 0
                    trigger_electrode = temp[ind]
                else:
                    temp = analyzorObject.mostActiveElectrodes(50)
                    trigger_electrode = temp[np.argwhere(temp == trigger_electrode)[0]+1][0]
                window["TRIGGER_ELECTRODE"].update(trigger_electrode)
            except:
                pass
        # Apply subsampling and color coding to the current analyzor object
        elif event == "Show and apply subsampling & Circular color coding":
            try:
                sampledElectrodes, sampledCoords = analyzorObject.electrode_subsampling(min_dist=int(values["MINIMAL_DISTANCE"]), point_limit=int(values["POINT_LIMIT"]), figureSize=(9,7), showPlotBool=True)
                coloredElectrodes, rgb_cycle, _ = analyzorObject.circular_color_coding(electrodes=sampledElectrodes, showPlotBool=True, dotSize=50)
            except:
                pass
        # Plot the raster plot for the spontaneous data triggered on the selected trigger electrode
        elif event == "Generate Raster Plot":
            if stimFileBool:
                tempAnalyzorObject = Analyzor.AnalyZor(recordingFile, inputPath=recordingsPath, stimulation_electrodes=[trigger_electrode]) 
                tempAnalyzorObject.loadDataSpikesOnly(blankingBool=True, blankingWindow=[-10, 10])
                spontaneousRasterPlot(analyzor=tempAnalyzorObject, electrodes=coloredElectrodes, colors=rgb_cycle, preWindow = int(values["PRE_WINDOW"])*20, windowSize = int(values["WINDOW_SIZE"])*20, trigger_electrode = int(values["TRIGGER_ELECTRODE"]), isStimBool=stimFileBool)
                del tempAnalyzorObject
            else:
                spontaneousRasterPlot(analyzor=analyzorObject, electrodes=coloredElectrodes, colors=rgb_cycle, preWindow = int(values["PRE_WINDOW"])*20, windowSize = int(values["WINDOW_SIZE"])*20, trigger_electrode = int(values["TRIGGER_ELECTRODE"]), isStimBool=stimFileBool)

##############################################################functions################################################################
#######################################################################################################################################

#Masks voltage map based on an electrode list
def mask_from_electrodeSelection(electrode_arr, matrix):
    electrode_arr = np.array(electrode_arr)
    mask = np.zeros(matrix.shape)
    for i in range(electrode_arr.shape[0]):
        mask[int(electrode_arr[i] / matrix.shape[1]), electrode_arr[i] % matrix.shape[1]] += 1
    cut_matrix = matrix * mask
    return cut_matrix
     
#Sets all elements outside of the defined square window to 0
#X&Y are the coordinates of the center of the window
def cutout_mask(X,Y,height,width,matrix):
    X = int(np.absolute(X))
    Y = int(np.absolute(Y))
    if X >= matrix.shape[1] or Y >= matrix.shape[1]:
        return matrix
    mask = np.zeros(matrix.shape)
    mask[max(Y-int(height/2),0):min(Y+int(height/2),matrix.shape[0]),max(X-int(width/2),0):min(X+int(width/2),matrix.shape[1])] += 1
    cutMatrix = matrix * mask
    return cutMatrix

#Returns a new 0,1 matrix of the specified window of the matrix according to the threshold
#X&Y are the coordinates of the center of the window
def cutout_mask_NoZeros_Treshed(X,Y,matrix,height,width,threshold):
    X = int(np.absolute(X))
    Y = int(np.absolute(Y))
    if X >= matrix.shape[1] or Y >= matrix.shape[1]:
        aboveThreshIndices = matrix > threshold
        mask = np.zeros(matrix.shape)
        mask[aboveThreshIndices] = 1
        return mask
    mask = matrix[max(Y-int(height/2),0):min(Y+int(height/2),matrix.shape[0]),max(X-int(width/2),0):min(X+int(width/2),matrix.shape[1])]
    aboveThreshIndices = mask > threshold
    mask = np.zeros(mask.shape)
    mask[aboveThreshIndices] = 1
    return mask

def isfloat(value):
  if value == '':
      return False
  try:
    float(value)
    return True
  except ValueError:
    return False

#Loads a list of electrode numbers into scope
def loadConfIntoScope(electrodeListUnsaved):
    maxlab.util.initialize()

    array = maxlab.chip.Array('online')
    array.reset()
    array.clear_selected_electrodes()
    array.select_electrodes(electrodeListUnsaved)
    array.route()
    array.download()
    maxlab.util.offset()
    return

# function to generate sinusoidal signal
def sineVari(t_period, amp=10, periods=1):
    """Create an FPGA loop on MidSupply
    """
    sineStr = ""
    n_samples = 20
    for i in range(0,n_samples):
        v = int(-amp*math.sin(periods*2*math.pi /n_samples * i))
        s = int(20e3 * t_period/n_samples)
        factor = 128 # 1.65/(3.0/1024)
        sineStr += str(v+factor) + "/" + str(s) + " "
    maxlab.send_raw("system_loop_sine_onVRef " + sineStr)

# Check if there is a txt file of the same name, meaning we are looking at a stimulation file
def check_if_stim_file(inputPath, stim_txt_file):
    f = os.path.exists(os.path.join(inputPath, stim_txt_file))
    return f

# Function to generate a new voltage map for the currently used chip
def getVoltageMap(fname, voltageMapsPath):
    layout = [
        [sg.Text('Please enter a name for the voltage map.')],
        [sg.InputText(key='VOLTAGE_MAP_ID', default_text=fname, enable_events=False)],
        [sg.Text(size=(25,1), k='-OUTPUT-')],
        [sg.Button('Confirm')]
    ]
    small_window = sg.Window('Get Voltage Map', layout, finalize=True)
    event, values_small = small_window.read()

    if event == sg.WIN_CLOSED:
        small_window.close()
    elif event == "Confirm":
        voltage_map_ID = values_small['VOLTAGE_MAP_ID']
        print("Obtaining voltage map for experiment {}.".format(voltage_map_ID))
        small_window.close()

        # set the switch settings to apply sinusoidal wave to reference electrode
        maxlab.util.initialize()
        time.sleep(1)
        maxlab.send(maxlab.system.Switches(sw_0=1, sw_1=0, sw_2=0, sw_3=1, sw_4=0, sw_5=0, sw_6=0, sw_7=0))
        maxlab.send(maxlab.chip.Amplifier().set_gain(7))

        # perform offset correction
        print("Offset correction...")
        maxlab.send_raw("system_loop_stop")
        time.sleep(1)
        # Program the loop
        sineVari(0.001, 1)
        maxlab.send_raw("system_loop_start")
        maxlab.util.offset()
        maxlab.send_raw("system_loop_stop")

        # iterate over config files, start loop and record
        frequency = 1000
        amp_in_bits = 20
        save_directory = voltageMapsPath
        load_directory = os.path.join(os.getcwd(), 'Configurations_full/')

        # initialize the saver (for some reason doesn't work without)
        Saver = maxlab.saving.Saving()
        Saver.open_directory(save_directory)
        Saver.set_legacy_format(True)
        Saver.group_delete_all()
        wells = range(1)

        for well in wells:
            Saver.group_define(well, "routed")

        print("Obtaining the voltage map...")
        for i in tqdm(range(29)):
            path = load_directory + 'cfg_{:0>3d}.cfg'.format(i)
            array = maxlab.chip.Array('A')
            array.load_config(path)
            array.download()

            maxlab.send_raw("system_loop_stop")

            # Program the loop
            sineVari(0.001, amp_in_bits)

            # Initialize server and start recording
            Saver = maxlab.saving.Saving()
            Saver.open_directory(save_directory)
            Saver.set_legacy_format(True)
            Saver.group_delete_all()
            wells = range(1)

            for well in wells:
                Saver.group_define(well, "routed")

            time.sleep(0.5)
            Saver.start_file('{:0>3d}'.format(i))
            Saver.start_recording(wells)

            # Start the sine loop
            maxlab.send_raw("system_loop_start")

            time.sleep(2.5)

            maxlab.send_raw("system_loop_stop")

            # Stop saving
            Saver.stop_recording()
            Saver.stop_file()
            Saver.group_delete_all()

            maxlab.send_raw("system_loop_stop")
            time.sleep(0.5)

        # initialize the system back to normal state
        maxlab.send(maxlab.system.Switches(sw_0=0, sw_1=0, sw_2=0, sw_3=0, sw_4=0, sw_5=0, sw_6=0, sw_7=0))
        maxlab.send(maxlab.chip.Amplifier().set_gain(512))
        maxlab.util.initialize()

        # Open all H5 datasets and analyze
        try:
            h5paths = sorted([os.path.join(save_directory,f) for f in os.listdir(save_directory) if f.endswith('.raw.h5')]) # load last or first 10
            f = [h5py.File(i, "r") for i in h5paths]
        except:
            print('H5 dataset could not be imported.')

        #Figure out all electrode numbers
        electrode_info = [np.asarray(i['mapping']['channel','electrode']) for i in f]
        mask = [i['electrode']!=-1 for i in electrode_info]
        clean_abs_inds = [i[0]['electrode'][i[1]] for i in zip(electrode_info,mask)]
        clean_rel_inds = [i[0]['channel'][i[1]] for i in zip(electrode_info,mask)]

        #For each recording figure out the x and y coordinates per electrode
        x_clean=[v%220 for v in clean_abs_inds]
        y_clean=[v/220 for v in clean_abs_inds]

        cut_traces = []
        for i,v in enumerate(clean_rel_inds):
            cut_traces.append(np.asarray(f[i]['sig'])[v,900:1000])

        cut_traces_max=np.asarray([np.amax(i,axis=1) for i in cut_traces])
        cut_traces_min=np.asarray([np.amin(i,axis=1) for i in cut_traces])
        cut_traces_amp=cut_traces_max-cut_traces_min

        #For each recordig build the elctrode array for visualization
        el_array = np.zeros((120,220))
        for i,j in enumerate(cut_traces_amp):
            int_indices = [int(float_index) for float_index in y_clean[i]]
            el_array[int_indices, x_clean[i]]=j

        # delete the h5 files
        for item in h5paths:
            if item.endswith('.raw.h5'):
                os.remove(item)

        # save the voltage map
        np.save(os.getcwd() + "/voltageMapArrays/" + voltage_map_ID, el_array)
        plt.imshow(el_array, cmap =vmap_cmap)
        plt.show()

# Function to create a custom electrode selection using the current voltage map and user input
def getElectrodeSelection(voltageMapPath, electrodeSelectionPaths, selection_name, network_number):

    selection_name, _ = selection_name.rsplit(".", 1)
    selection_name = 'electrodes_{}_N{}.npy'.format(selection_name, network_number)

    layout = [[sg.Text('Please enter a name for the electrode selection.')],
          [sg.InputText(size=(60, 1),key='ELECTRODE_SELECTION_ID', enable_events=True, default_text=str(selection_name))],
          [sg.Button('Confirm', key='ELECTRODE_SELECTION_CONFIRM')]]
    small_window = sg.Window('Get Electrode Selection', layout, finalize=True)

    continue_bool = False
    while True:
        event, values_small = small_window.read()

        if event == sg.WIN_CLOSED:
            small_window.close()
            break
        elif event == "ELECTRODE_SELECTION_ID":
            selection_name = values_small['ELECTRODE_SELECTION_ID']
        elif event == "ELECTRODE_SELECTION_CONFIRM":
            electrode_selection_ID = selection_name
            electrode_selection_ID = os.path.join(electrodeSelectionPaths, electrode_selection_ID)
            small_window.close()
            continue_bool = True
            break

    if continue_bool:
        # Parameters for electrode selection functions
        voltage_map = np.load(voltageMapPath)

        scale_factor = 7
        selection_threshold = 40
        selection_threshold_default = selection_threshold
        n_sample = 1000
        color_map = "hot"

        line_thickness = 1
        dot_radius = 2
        alpha = 0.01
        fontSize = 0.5
        windowSize = [220*scale_factor,120*scale_factor]

        #interactive functionality
        global selected_pixels_add
        global selected_pixels_remove
        selected_pixels_add = []
        selected_pixels_remove = []
        selected_electrodes_add = []
        selected_electrodes_remove = []
        selected_electrodes = []
        selected_electrodes_hist = []
        add_remove_hist = []
        selection_threshold_hist = []

        def pixel_to_electrode(selected_pixels,voltage_map,selection_threshold):
            #determine the electrodes corresponding to the marked pixels
            selected_electrodes = []
            selected_electrodes_vertices = [tuple(int(round(selected_pixels[i][j]/scale_factor)) for j in range(len(selected_pixels[i]))) for i in range(len(selected_pixels))]
            x_vals = [selected_electrodes_vertices[i][0] for i in range(len(selected_electrodes_vertices))]
            y_vals = [selected_electrodes_vertices[i][1] for i in range(len(selected_electrodes_vertices))]
            min_x = int(min(x_vals))
            max_x = int(max(x_vals))
            min_y = int(min(y_vals))
            max_y = int(max(y_vals))
            x_arange = np.arange(min_x,max_x)
            y_arange = np.arange(min_y,max_y)
            for i in range(len(x_arange)):
                for j in range(len(y_arange)):
                    if voltage_map[y_arange[j]][x_arange[i]] > selection_threshold:
                        in_poly = pixel_in_selection(x_arange[i],y_arange[j],selected_electrodes_vertices)
                        if in_poly:
                            selected_electrodes.append((x_arange[i],y_arange[j]))
            return selected_electrodes

        def pixel_in_selection(x,y,selected_electrodes_vertices):
            #check if pixel is in selection using the even-odd rule (https://en.wikipedia.org/wiki/Even-odd_rule)
            result = False
            j = len(selected_electrodes_vertices)-1
            for i in range(len(selected_electrodes_vertices)):
                if (x == selected_electrodes_vertices[i][0]) and (y == selected_electrodes_vertices[i][1]):
                    return True
                if ((selected_electrodes_vertices[i][1] > y) != (selected_electrodes_vertices[j][1] > y)):
                    slope = (x-selected_electrodes_vertices[i][0])*(selected_electrodes_vertices[j][1]-selected_electrodes_vertices[i][1])-(selected_electrodes_vertices[j][0]-selected_electrodes_vertices[i][0])*(y-selected_electrodes_vertices[i][1])
                    if slope == 0:
                        return True
                    elif (slope < 0) != (selected_electrodes_vertices[j][1] < selected_electrodes_vertices[i][1]):
                        result = not result
                j = i
            return result

        def plot_electrodes(selected_electrodes,voltage_map_rgb,dot_radius):
            #plot electrodes
            selected_electrodes_plot = [tuple(selected_electrodes[i][j]*scale_factor for j in range(len(selected_electrodes[i]))) for i in range(len(selected_electrodes))]
            for i in range(len(selected_electrodes_plot)):
                cv2.circle(voltage_map_rgb,selected_electrodes_plot[i],radius=dot_radius,color=[0,0,255,1],thickness=-1)
            voltage_map_rgb = cv2.addWeighted(voltage_map_rgb,alpha,copy.deepcopy(initial_map),1-alpha,0)
            return voltage_map_rgb
            
        def selection_pixels(event,x,y,flags,param):
            #add pixels to selection polygon
            if event == cv2.EVENT_LBUTTONDOWN:
                selected_pixels_add.append((x, y))
            if event == cv2.EVENT_RBUTTONDOWN:
                selected_pixels_remove.append((x, y))

        im = plt.imshow(voltage_map,cmap=color_map) #sometimes a specific colormap breaks a voltage map, use a different one if that happens
        voltage_map_rgb = np.array(im.cmap(im.norm(im.get_array()))[:,:,0:3])
        voltage_map_rgb = cv2.resize(voltage_map_rgb,(scale_factor*np.shape(voltage_map_rgb)[1],scale_factor*np.shape(voltage_map_rgb)[0]))
        initial_map = voltage_map_rgb.copy()
        prev_map = voltage_map_rgb.copy()
        voltage_map_rgb_add = voltage_map_rgb.copy()
        voltage_map_rgb_remove = voltage_map_rgb.copy()
        cv2.namedWindow("voltage_map")
        cv2.setMouseCallback("voltage_map", selection_pixels)

        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if selected_pixels_add:
                cv2.fillPoly(voltage_map_rgb_add,[np.array(selected_pixels_add)],color=[0,0,255,0.2])
                voltage_map_rgb_add = cv2.addWeighted(voltage_map_rgb_add,alpha,copy.deepcopy(prev_map),1-alpha,0)
            
            if selected_pixels_remove:
                cv2.fillPoly(voltage_map_rgb_remove,[np.array(selected_pixels_remove)],color=[0,255,0,0.2])
                voltage_map_rgb_remove = cv2.addWeighted(voltage_map_rgb_remove,alpha,copy.deepcopy(prev_map),1-alpha,0)

            if key == ord("a"):
                #add or subtract electrodes based on drawn polygons
                if selected_pixels_add or selected_pixels_remove:
                    if selected_pixels_add:
                        #electrodes in addition polygon
                        selected_electrodes_add = pixel_to_electrode(selected_pixels_add,voltage_map,selection_threshold)
                        selected_electrodes += selected_electrodes_add
                    if selected_pixels_remove:
                        #electrodes in subtraction polygon
                        selected_electrodes_remove = pixel_to_electrode(selected_pixels_remove,voltage_map,selection_threshold)
                        selected_electrodes = [i for i in selected_electrodes if i not in selected_electrodes_remove]
                    add_remove_hist.append(selected_pixels_add)
                    add_remove_hist.append(selected_pixels_remove)
                    selected_electrodes_hist.append(copy.deepcopy(selected_electrodes))
                    selection_threshold_hist.append(0)
                    selected_electrodes_indices = [selected_electrodes[i][1]*220+selected_electrodes[i][0] for i in range(len(selected_electrodes))]
                    selected_electrodes_indices = np.sort(selected_electrodes_indices)
                    selected_electrodes_boolean = np.zeros((120,220))
                    for i in np.asarray(selected_electrodes):
                        selected_electrodes_boolean[i[1],i[0]] = 1
                    np.save(electrode_selection_ID,np.asarray(selected_electrodes_indices))
                    if len(selected_electrodes_hist) >= 2:
                        if len(selected_electrodes_hist[-1]) >= len(selected_electrodes_hist[-2]):
                            print('electrodes added:     ' + str(len(selected_electrodes_hist[-1])-len(selected_electrodes_hist[-2])))
                        elif len(selected_electrodes_hist[-2]) >= len(selected_electrodes_hist[-1]):
                            print('electrodes removed:   ' + str(len(selected_electrodes_hist[-2])-len(selected_electrodes_hist[-1])))
                    else:
                        print('electrodes added:     ' + str(len(selected_electrodes_hist[0])))
                    selected_pixels_add = []
                    selected_pixels_remove = []
                    selected_electrodes_add = []
                    selected_electrodes_remove = []
                    voltage_map_rgb_add = []
                    voltage_map_rgb_remove = []
                    print('electrodes selected: ',len(selected_electrodes),'\n')
                else:
                    print('currently no selection')

            if key == ord("u"):
                #refresh with increased selection threshold
                selection_threshold += 1
                print('selection  threshold: '+str(selection_threshold))
                selection_threshold_hist.append(-1)
                if add_remove_hist:
                    selected_electrodes = []
                    for i in range(len(add_remove_hist)):
                        if add_remove_hist[i]!=[]:
                            if i%2==0:
                                selected_electrodes_add = pixel_to_electrode(add_remove_hist[i],voltage_map,selection_threshold)
                                selected_electrodes += selected_electrodes_add
                            if i%2==1:
                                selected_electrodes_remove = pixel_to_electrode(add_remove_hist[i],voltage_map,selection_threshold)
                                selected_electrodes = [i for i in selected_electrodes if i not in selected_electrodes_remove]
                    selected_electrodes = list(set(selected_electrodes))
                    selected_electrodes_hist.append(copy.deepcopy(selected_electrodes))
                    add_remove_hist.append(add_remove_hist[-2])
                    add_remove_hist.append(add_remove_hist[-2])
                    selected_electrodes_indices = [selected_electrodes[i][1]*220+selected_electrodes[i][0] for i in range(len(selected_electrodes))]
                    selected_electrodes_indices = np.sort(selected_electrodes_indices)
                    selected_electrodes_boolean = np.zeros((120,220))
                    for i in np.asarray(selected_electrodes):
                        selected_electrodes_boolean[i[1],i[0]] = 1
                    np.save(electrode_selection_ID,np.asarray(selected_electrodes_indices))
                    print('electrodes removed:   ' + str(len(selected_electrodes_hist[-2])-len(selected_electrodes_hist[-1])))
                    selected_electrodes_add = []
                    selected_electrodes_remove = []
                else:
                    print(' ')
                print('electrodes selected: ',len(selected_electrodes),'\n')

            if key == ord("i"):
                #refresh with decreased selection threshold
                selection_threshold -= 1
                print('selection  threshold: '+str(selection_threshold))
                selection_threshold_hist.append(1)
                if add_remove_hist:
                    selected_electrodes = []
                    for i in range(len(add_remove_hist)):
                        if add_remove_hist[i]!=[]:
                            if i%2==0:
                                selected_electrodes_add = pixel_to_electrode(add_remove_hist[i],voltage_map,selection_threshold)
                                selected_electrodes += selected_electrodes_add
                            if i%2==1:
                                selected_electrodes_remove = pixel_to_electrode(add_remove_hist[i],voltage_map,selection_threshold)
                                selected_electrodes = [i for i in selected_electrodes if i not in selected_electrodes_remove]
                    selected_electrodes = list(set(selected_electrodes))
                    selected_electrodes_hist.append(copy.deepcopy(selected_electrodes))
                    add_remove_hist.append(add_remove_hist[-2])
                    add_remove_hist.append(add_remove_hist[-2])
                    selected_electrodes_indices = [selected_electrodes[i][1]*220+selected_electrodes[i][0] for i in range(len(selected_electrodes))]
                    selected_electrodes_indices = np.sort(selected_electrodes_indices)
                    selected_electrodes_boolean = np.zeros((120,220))
                    for i in np.asarray(selected_electrodes):
                        selected_electrodes_boolean[i[1],i[0]] = 1
                    np.save(electrode_selection_ID,np.asarray(selected_electrodes_indices))
                    print('electrodes added:     ' + str(len(selected_electrodes_hist[-1])-len(selected_electrodes_hist[-2])))
                    selected_electrodes_add = []
                    selected_electrodes_remove = []
                else:
                    print(' ')
                print('electrodes selected: ',len(selected_electrodes),'\n')

            if key == ord("s"):
                #randomly sample n electrodes from selection
                selection_threshold_hist.append(0)
                if add_remove_hist and len(selected_electrodes) > 1000:
                    selected_electrodes = []
                    for i in range(len(add_remove_hist)):
                        if add_remove_hist[i]!=[]:
                            if i%2==0:
                                selected_electrodes_add = pixel_to_electrode(add_remove_hist[i],voltage_map,selection_threshold)
                                selected_electrodes += selected_electrodes_add
                            if i%2==1:
                                selected_electrodes_remove = pixel_to_electrode(add_remove_hist[i],voltage_map,selection_threshold)
                                selected_electrodes = [i for i in selected_electrodes if i not in selected_electrodes_remove]
                    print('sampling '+str(n_sample)+' electrodes')
                    selected_electrodes = list(set(selected_electrodes))
                    random_indices = np.random.choice(len(selected_electrodes),size=n_sample,replace=False).astype(int)
                    selected_electrodes = np.array(selected_electrodes)[random_indices]
                    selected_electrodes = list(map(tuple,selected_electrodes))
                    selected_electrodes_hist.append(copy.deepcopy(selected_electrodes))
                    add_remove_hist.append(add_remove_hist[-2])
                    add_remove_hist.append(add_remove_hist[-2])
                    selected_electrodes_indices = [selected_electrodes[i][1]*220+selected_electrodes[i][0] for i in range(len(selected_electrodes))]
                    selected_electrodes_indices = np.sort(selected_electrodes_indices)
                    selected_electrodes_boolean = np.zeros((120,220))
                    for i in np.asarray(selected_electrodes):
                        selected_electrodes_boolean[i[1],i[0]] = 1
                    np.save(electrode_selection_ID,np.asarray(selected_electrodes_indices))
                    print('electrodes removed:     ' + str(len(selected_electrodes_hist[-2])-len(selected_electrodes_hist[-1])))
                    selected_electrodes_add = []
                    selected_electrodes_remove = []
                else:
                    print(' ')
                print('electrodes selected: ',len(selected_electrodes),'\n')

            if key == ord("z"):
                #revert selection one step
                if len(selection_threshold_hist) >= 2:
                    selection_threshold += int(selection_threshold_hist[-1])
                    selection_threshold_hist = selection_threshold_hist[:-1]
                elif len(selection_threshold_hist) == 1:
                    selection_threshold = selection_threshold_default
                print('selection  threshold: '+str(selection_threshold))
                if len(selected_electrodes_hist) >= 2:
                    selected_electrodes = copy.deepcopy(selected_electrodes_hist[-2])
                    if len(selected_electrodes_hist[-1]) >= len(selected_electrodes_hist[-2]):
                        print('electrodes unadded:   ' + str(len(selected_electrodes_hist[-1])-len(selected_electrodes_hist[-2])))
                    elif len(selected_electrodes_hist[-2]) >= len(selected_electrodes_hist[-1]):
                        print('electrodes readded:   ' + str(len(selected_electrodes_hist[-2])-len(selected_electrodes_hist[-1])))
                elif len(selected_electrodes_hist) == 1:
                    selected_electrodes = []
                    print('electrodes unadded:   ' + str(len(selected_electrodes_hist[-1])))
                elif not selected_electrodes_hist:
                    print('')
                selected_electrodes_indices = [selected_electrodes[i][1]*220+selected_electrodes[i][0] for i in range(len(selected_electrodes))]
                selected_electrodes_indices = np.sort(selected_electrodes_indices)
                selected_electrodes_boolean = np.zeros((120,220))
                for i in np.asarray(selected_electrodes):
                    selected_electrodes_boolean[i[1],i[0]] = 1
                np.save(electrode_selection_ID,np.asarray(selected_electrodes_indices))
                selected_electrodes_hist = selected_electrodes_hist[:-1 or None]
                add_remove_hist = add_remove_hist[:-2 or None]
                print('electrodes selected: ',len(selected_electrodes),'\n')

            voltage_map_rgb = plot_electrodes(selected_electrodes,voltage_map_rgb,dot_radius)

            if selected_pixels_add and selected_pixels_remove:
                voltage_map_rgb = cv2.addWeighted(voltage_map_rgb_add,0.5,voltage_map_rgb_remove,0.5,0)
            elif selected_pixels_add:
                voltage_map_rgb = voltage_map_rgb_add
                voltage_map_rgb_add = copy.deepcopy(initial_map)
            elif selected_pixels_remove:
                voltage_map_rgb = voltage_map_rgb_remove
                voltage_map_rgb_remove = copy.deepcopy(initial_map)
            else:
                voltage_map_rgb_add = copy.deepcopy(initial_map)
                voltage_map_rgb_remove = copy.deepcopy(initial_map)
            prev_map = voltage_map_rgb

            if key == ord("r"):
                #reset selection
                selected_pixels_add = []
                selected_pixels_remove = []
                selected_electrodes_add = []
                selected_electrodes_remove = []
                selected_electrodes = []
                selected_electrodes_indices = []
                selection_threshold = selection_threshold_default
                selected_electrodes_hist = []
                selection_threshold_hist = []
                add_remove_hist = []
                voltage_map_rgb_add = copy.deepcopy(initial_map)
                voltage_map_rgb_remove = copy.deepcopy(initial_map)
                prev_map = copy.deepcopy(initial_map)
                voltage_map_rgb = copy.deepcopy(initial_map)
                print('selection resetted')

            if key == ord("c"):
                #end script
                cv2.destroyAllWindows()
                break
            cv2.imshow('voltage_map',voltage_map_rgb)
            cv2.putText(voltage_map_rgb,'C: close | A: confirm selection | R: reset selection | U/I: increase/decrease selection threshold | S: sample '+str(n_sample)+' electrodes | Z: revert action | L/R Click: add join/disjoin vertex',(0,int(round(0.995*windowSize[1]))),cv2.FONT_HERSHEY_TRIPLEX,fontSize,color=(255,255,255))
        return selected_electrodes_indices            

#Records electrodes which are routed in scope
#Record time will be accurate up to 1 second and at least as long as given in the gui
def recordInScope(directory, filename, duration, fileformat, onlyRecordSpikes):
    wells = range(1)

    Saver = maxlab.saving.Saving()
    Saver.open_directory(directory)
    Saver.set_legacy_format(fileformat)
    Saver.group_delete_all()

    if not onlyRecordSpikes:
        for well in wells:
            Saver.group_define(well, "routed")

    Saver.start_file(filename)
    Saver.start_recording(wells)

    wait(duration)

    Saver.stop_recording()

    Saver.stop_file()
    Saver.group_delete_all()

#input of a duration in second gets converted to the specified unit
#Check begin of code for the possible units
def smallestTimeUnitConverter(duration, units):
    unit = units[0]
    for i in range(2):
        if (duration / 60 > 1):
            duration /= 60
            unit = units[i+1]
    return str(round(duration, 4)), unit

def largestTimeUnitConverter(duration, units):
    unit = units[0]
    for i in range(len(units)-1):
        if (duration < 1):
            duration *= 1000
            unit = units[i+1]
    return str(round(duration, 4)), unit

#Reverse operation of smallestTimeUnitConverter
def durationInSeconds(duration, unit, units):
    if not unit in units:
        unit = units[2]
    if unit == units[0] or unit == units[1]:
        multiplicator = 60**(2-units.index(unit))
    else:
        multiplicator = 10**(-(units.index(unit) - 2) * 3)
    return duration*multiplicator

def stimulusDuration(burnIn,iteration,burnOut, totalLoop):
    return (burnIn + iteration + burnOut)*totalLoop

def wait(duration):
    if duration <= 0:
        return
    tic = time.time()
    time.sleep(duration)
    toc = time.time() - tic
    print("Slept for {} seconds".format(toc))

#Returns how many frames long a sequence can be without overflowing the hardware.
#The unit is in pulsedurations.
#If monophasic, change 3 to 2
def blockDuration(pattList, sequenceSize = 200):
    periods = np.asarray([x.interPeakInterval for x in pattList])
    nrOfElectrodesPerPeriod = np.asarray([len(x.electrodeNumbers) for x in pattList])

    durationInPulseDur = np.lcm.reduce((periods).astype(int))
    commands = int(np.sum(durationInPulseDur / periods) * 3 + 2 * np.sum(durationInPulseDur / periods * nrOfElectrodesPerPeriod))
    durationInPulseDur = int(sequenceSize / commands) * durationInPulseDur

    return durationInPulseDur

#Takes a list of stimulation patterns (according to class) as the input.
#Generates a list of lists, each sublist contains an array with each element corresponding to a stimulation electrode.
#The order is the same as in the list of the gui. It has a 1 if a pulse is sent and a zero if not.
#The second entry is the duration in pulsedurations set by the user. If it is 0, a pulse must be sent on this frame. Else
#the chip will wait the entry * pulseduration.
def switchPattern(pattList, electrodeList, duration):
    if duration <= 0:
        return []
    onOffTimes = np.zeros([len(electrodeList),int(duration)])
    index = 0
    for i in pattList:
        interPeakInterval = int(abs(i.interPeakInterval) )
        offset = int(abs(int(i.offset)) )
        repeatTimes = int(max(0,(onOffTimes.shape[1] - offset) / max(1,interPeakInterval)))
        repeatArray = np.zeros(interPeakInterval)
        repeatArray[0] = 1
        if not offset >= onOffTimes.shape[1]:
            onOffTimes[index:index+len(i.electrodeNumbers),offset:offset+repeatTimes * interPeakInterval] = np.tile(repeatArray,repeatTimes)
            if offset+repeatTimes * interPeakInterval < onOffTimes.shape[1] and offset != 0:
                onOffTimes[index:index + len(i.electrodeNumbers), offset+repeatTimes * interPeakInterval] = 1
            index += len(i.electrodeNumbers)
    comparisonVector = onOffTimes[:,0]
    switchList = []
    duration = 0
    for index in range(1,onOffTimes.shape[1]):
        if (0==comparisonVector).all() and (0==onOffTimes[:,index]).all():
            duration += 1
        else:
            if (0!=comparisonVector).any():
                duration = 0
            switchList.append([comparisonVector,duration])
            comparisonVector = onOffTimes[:,index]
            duration = 1


        if index == onOffTimes.shape[1] - 1:
            tempVector = comparisonVector
            switchList.append([tempVector,duration])

    return switchList

#Reconnects all configurations in scope and using the stimulationPattern function, it generates sequences to get
#the correct stimulation pattern.
def recordStimulation(switchList, burnInTime, burnOutTime, sequenceDuration, loops, amplitude, electrodesList, stimulationElectrodesList, directory, filename, fileformat, onlyRecordSpikes, recordBurnTime, recordFirstStims, recordLastStims, totalLoops, pulseDuration, nrOfLastSequences):
    maxlab.util.initialize()
    maxlab.send(maxlab.chip.Core().enable_stimulation_power(True))
    array = maxlab.chip.Array('stimulation')
    array.reset()
    array.clear_selected_electrodes()
    array.select_electrodes(electrodesList)
    array.select_stimulation_electrodes(stimulationElectrodesList)
    array.route()

    stimulation_units = []
    stimulation_units_indices = []
    index = 0
    for stim_el in stimulationElectrodesList:
        array.connect_electrode_to_stimulation(stim_el)
        stim = array.query_stimulation_at_electrode(stim_el)
        if stim:
            stimulation_units.append(stim)
            stimulation_units_indices.append(index)
        else:
            print("No stimulation channel can connect to electrode: " + str(stim_el))
        index += 1

    array.download()
    maxlab.util.offset()
    time.sleep(5)

    if np.unique(stimulation_units).shape[0] != len(stimulationElectrodesList):
        print("Either not all electrodes are connected or at least 2 are connected to the same Stimbuffer.")

    for i in stimulation_units:
        stim = maxlab.chip.StimulationUnit(i)
        stim.power_up(True)
        stim.connect(True)
        stim.set_voltage_mode()
        stim.dac_source(0)
        maxlab.send(stim)

        stim = maxlab.chip.StimulationUnit(i)
        stim.power_up(False)
        stim.connect(False)
        maxlab.send(stim)


    def append_full_sequence(seq, delay):
        for k in switchList:
            if int((k[1])) == 0:
                for j,i in enumerate(stimulation_units_indices):
                    if int((k[0])[i]) == 1:
                        stim = maxlab.chip.StimulationUnit(stimulation_units[j])
                        stim.power_up(True)
                        stim.connect(True)
                        stim.set_voltage_mode()
                        stim.dac_source(0)
                        seq.append(stim)
                seq.append(maxlab.chip.DAC(0, 512 - int(amplitude/2.9)))
                seq.append(maxlab.system.DelaySamples(int(delay / 2)))
                seq.append(maxlab.chip.DAC(0, 512 + int(amplitude/2.9)))
                seq.append(maxlab.system.DelaySamples(int((delay + 1)/ 2)))
                seq.append(maxlab.chip.DAC(0, 512))
                for j,i in enumerate(stimulation_units_indices):
                    if int((k[0])[i]) == 1:
                        stim = maxlab.chip.StimulationUnit(stimulation_units[j])
                        stim.power_up(False)
                        stim.connect(False)
                        seq.append(stim)
            else:
                seq.append(maxlab.system.DelaySamples(int(k[1]*delay)))
        return seq

    time.sleep(1)

    seq = maxlab.Sequence()
    append_full_sequence(seq, pulseDuration)
    wells = range(1)

    Saver = maxlab.saving.Saving()
    Saver.open_directory(directory)
    Saver.set_legacy_format(fileformat)
    Saver.group_delete_all()

    if not onlyRecordSpikes:
        for well in wells:
            Saver.group_define(well, "routed")

    Saver.start_file(filename)

    (maxlab.Sequence().append(maxlab.system.DelaySamples(20000))).send()

    isRecording = False
    for i in range(totalLoops):

        if not isRecording and recordBurnTime:
            print("Started To Record")
            Saver.start_recording(wells)
            isRecording = True
        if isRecording and not recordBurnTime:
            print("Stopped Recording")
            Saver.stop_recording()
            isRecording = False
        wait(burnInTime)

        if not isRecording and recordFirstStims:
            print("Started To Record")
            Saver.start_recording(wells)
            isRecording = True
        if isRecording and not recordFirstStims:
            print("Stopped Recording")
            Saver.stop_recording()
            isRecording = False

        for i in range(max(loops-nrOfLastSequences,0)):
            print("Sending Sequence")
            seq.send()
            wait(sequenceDuration)

        if not isRecording and recordLastStims:
            print("Started To Record")
            Saver.start_recording(wells)
            isRecording = True
        if isRecording and not recordLastStims:
            print("Stopped Recording")
            Saver.stop_recording()
            isRecording = False

        for i in range(min(nrOfLastSequences, loops)):
            print("Sending Sequence")
            seq.send()
            wait(sequenceDuration)

        if not isRecording and recordBurnTime:
            print("Started To Record")
            Saver.start_recording(wells)
            isRecording = True
        if isRecording and not recordBurnTime:
            print("Stopped Recording")
            Saver.stop_recording()
            isRecording = False

        if isRecording or i < totalLoops-1:
            wait(burnOutTime)

    if isRecording:
        print("Stopped Recording")
        Saver.stop_recording()
    Saver.stop_file()
    Saver.group_delete_all()

    for i in stimulation_units:
        maxlab.chip.StimulationUnit(i).power_up(False)
        maxlab.chip.StimulationUnit(i).connect(False)
        maxlab.send(maxlab.chip.StimulationUnit(i))

# Create a sponatneous spike data raster plot triggered on a trigger electrode
def spontaneousRasterPlot(analyzor, electrodes, colors, preWindow = 0, windowSize = 300, trigger_electrode = None, isStimBool=False):

    spikes = analyzor.spikes
    mapping = analyzor.electrodeChannelMapping

    if isStimBool:
        trigger_spike_times = analyzor.blankingEnd
    else:
        trigger_spike_times = spikes[np.argwhere(mapping[0,:] == trigger_electrode)[0][0]][0]

    delay_list = []
    start_time_list = []
    for trigger_time in trigger_spike_times:
        delays = []
        end_time = trigger_time - preWindow + windowSize
        start_time_list.append((trigger_time-preWindow)/20000)
        for electrode in electrodes:
            ch = np.argwhere(mapping[0,:] == electrode)[0][0]
            temp_spikes = (spikes[ch][0])[
                np.argwhere(
                    np.logical_and(
                        spikes[ch][0] < end_time, spikes[ch][0] >= (trigger_time - preWindow)
                    )
                )
            ]
            if temp_spikes.size != 0:
                delay = (temp_spikes[0] - (trigger_time - preWindow)) * 1000 / 20000
            else:
                delay = np.nan
            delays.append(delay)
        delay_list.append(delays)

    def boundary_dep_electrode_coords(electrode):
        x, y = AnalyZor_Helper.convert_elno_to_xy(electrode)
        boundX, boundY = analyzor.get_boundaries()
        return x - boundY[0], y - boundX[0]

    coloredCoords = np.zeros((len(electrodes), 2))
    for i, e in enumerate(electrodes):
        x,y = boundary_dep_electrode_coords(e)
        coloredCoords[i, :] = np.array([x,y] )

    x_trig, y_trig = boundary_dep_electrode_coords(trigger_electrode)

    fig, axes = plt.subplots(2,1,figsize=(9,7))
    axes[0].set_title('Sampled Electrodes')
    axes[0].scatter(
        coloredCoords[:, 1],
        coloredCoords[:, 0],
        marker="s",
        c=colors,
        s=30,
    )
    axes[0].scatter(
        y_trig,
        x_trig,
        marker="x",
        c='k',
        label='Trigger Electrode',
        s=100
    )
    axes[0].invert_yaxis()
    axes[0].set_aspect('equal')
    axes[0].legend(loc='upper right')

    for i, delay in enumerate(delay_list):
        axes[1].scatter(
            delay, 
            [start_time_list[i] for x in range(len(delay))],
            marker='s',
            c=colors,
            s=1
        )
        axes[1].set_title('Spontanious Activity Raster Plot')
        axes[1].set_xlabel('Delay [ms]')
        axes[1].set_ylabel('Absolute recording time [s]')
    axes[1].vlines(preWindow*1000/20000, 0, start_time_list[-1], label='Trigger Time', colors="k")
    axes[1].legend(loc='upper right')
    plt.show()
# Execute main function
main()
