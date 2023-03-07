import matplotlib.pyplot as plt
import numpy as np
from AnalyZor_Helper import convert_elno_to_xy, convert_xy_to_elno, constant_getter, generate_mask, generate_grid
from tqdm.notebook import tqdm

from AnalyZor_Class import AnalyZor

class Flow():
    """
    This class implements all the flow analysis of spikes stored in an h5 file obtained using the MaxOne System.
    :param data: An AnalyZor object containing the spike data post blanking
    """

    def __init__(self, analyzor_object=None):
        try:
            self.data = analyzor_object
            print("Flow object created using AnalyZor data from {}.".format(self.data.filename))
        except:
            print("Faulty or no AnalyZor object was given as argument.")

        # Create spike array with one row per spike event and columns: spike_time, spike_amplitude, electrode
        spike_array = []
        for ind in range(len(self.data.spikes)):
            d = self.data.spikes[ind]
            electrode = int(d[1]['electrode'])
            nr_peaks = int(d[1]['nr_of_peaks'])
            for s in range(nr_peaks):
                spike_array.append([d[0][s], d[1]['peak_heights'][s], electrode])
        self.spike_array = np.array(spike_array)

    def fast_single_window_flow(self, start_frame=0, spatial_window=1, vector_weighting='none', temporal_window=10,
                                coincidence_window=0.2, verbose=False, normalize=False, full_output=False):
        """
        Calculates a flow map using all spikes occuring within a predefined temporal window. Uses the new Neuroflow
        algorithm which allows for a spatial coincidence window of any size.

        Parameters
        ----------
        start_frame: First frame number of the window to look at
        spatial_window: Radius of the circle in which electrodes will be considered as spatial related
        vector_weighting: Type of weighting for the vectors: None-> equal weights, temporal: based on time difference of spikes
        temporal_window: total temporal window to be considered after first frame
        coincidence_window: time after a spike in which a following spike will be considered related
        verbose: for detailed algorithm information
        normalize: normalize vector lengths
        full_output: If true, also returns the transfer vector electrodes

        Returns
        -------
        x,y,u,v,c
        Being the coordinates, vector values and delay values for coloring purposes
        """

        # Get electrode list
        all_electrodes = self.data.electrodeChannelMapping[0]
        self.all_electrodes = all_electrodes

        # Process input parameters
        temporal_window_frames = int(temporal_window / 0.05)
        coincidence_window_frames = int(coincidence_window / 0.05)
        stop_frame = start_frame + temporal_window_frames

        # Find all spikes within the time window and sort them by time
        data_to_use = np.squeeze(self.spike_array[np.argwhere(
            np.logical_and(self.spike_array[:, 0] >= start_frame, self.spike_array[:, 0] <= stop_frame)), :])
        try:
            data_to_use = data_to_use[data_to_use[:, 0].argsort()]
        except:
            data_to_use = data_to_use[np.newaxis, :]

        coords = np.array([[int(e / 220), e % 220] for e in data_to_use[:, 2]])

        # initialize the transfer matrix
        transfer_vector_list = []
        transfer_vector_weighting = []
        transfer_delay_list = []
        transfer_electrode_list = []

        # Iterate over all electrodes
        for index, electrode in enumerate(np.unique(data_to_use[:, 2])):
            vector_list = []
            weight_list = []
            delay_list = []

            # Find all spike times for this electrode
            elec_ind = np.squeeze(np.argwhere(data_to_use[:, 2] == electrode))
            electrode_spike_times = data_to_use[elec_ind, 0]
            if elec_ind.shape:
                position = coords[elec_ind[0], :]
            else:
                position = coords[elec_ind, :]

            if verbose:
                print(
                    'Looking at electrode {} with spiketimes {} at position {}'.format(electrode, electrode_spike_times,
                                                                                       position))

            # if single spike convert to list
            if type(electrode_spike_times) == np.float64:
                electrode_spike_times = [electrode_spike_times]
            # Iterate over all spike times for this electrode
            for spike_time in electrode_spike_times:
                if verbose:
                    print('Looking at spike: {}'.format(spike_time))
                # Find spikes within temporal window:
                coinc_spike_ind = np.squeeze(np.argwhere(
                    np.logical_and(data_to_use[:, 0] - spike_time < coincidence_window_frames,
                                   data_to_use[:, 0] > spike_time)))
                if verbose:
                    print('Coinc spike ind {}'.format(coinc_spike_ind))
                # For all spikes within temporal coinc. window, also check if they are within the spatial window
                try:
                    distances = np.hypot(*(coords[coinc_spike_ind] - position).T)
                    if verbose:
                        print('Distances {}'.format(distances))

                    # Catch single spike responses
                    if type(distances) == np.float64:
                        if distances <= spatial_window:
                            coinc_final_ind = coinc_spike_ind
                        else:
                            coinc_final_ind = []
                        if verbose:
                            print('Final ind to keep {}'.format(coinc_final_ind))

                    else:
                        to_keep_rel_ind = np.squeeze(np.argwhere(distances <= spatial_window))
                        if verbose:
                            print('Rel ind to keep {}'.format(to_keep_rel_ind))
                        coinc_final_ind = coinc_spike_ind[to_keep_rel_ind]
                        if verbose:
                            print('Final ind to keep {}'.format(coinc_final_ind))
                    # Finally we have spike times and positions for all spatiotemporally related spikes
                    coinc_spikes = data_to_use[coinc_final_ind, :]
                    coinc_coords = coords[coinc_final_ind, :]
                except:
                    if verbose:
                        print('No temporal or spatial follow up')
                    continue

                # Reformat if necessary
                if coinc_spikes.ndim == 1:
                    coinc_spikes = coinc_spikes[np.newaxis, :]
                    coinc_coords = coinc_coords[np.newaxis, :]

                if verbose:
                    print(coinc_spikes.shape)
                    print('Spikes and coords to consider: \n{}\n{}'.format(coinc_spikes, coinc_coords))

                # Calculate the weights for all spatiotemporally related spikes
                for i in range(coinc_spikes.shape[0]):
                    # Either by temporal weigting (closer in time -> stronger weight)
                    if vector_weighting == 'temporal':
                        weight = 1 - ((coinc_spikes[i][0] + 1 - spike_time) / (
                            coincidence_window_frames))
                    # Or equal weights for all spikes
                    else:
                        weight = 1
                    # Calculate delays for color coding
                    delay_list.append([(coinc_spikes[i][0] - start_frame) / 20000 * 1000])
                    # Save weights for later normalisation (if needed)
                    weight_list.append([weight])
                    # Calculate final vector using the weights from above
                    vector_list.append(
                        [(coinc_coords[i][0] - position[0]) * weight, (coinc_coords[i][1] - position[1]) * weight])
            if verbose:
                print('Delays {}, weights {}, vectors {}'.format(delay_list, weight_list, vector_list))

            transfer_vector_list.append(vector_list)
            transfer_vector_weighting.append(weight_list)
            transfer_delay_list.append(delay_list)
            transfer_electrode_list.append(electrode)

        # generate u,v matrices for vector plot
        chipWidth = constant_getter("chipWidth")
        boundY = [min(all_electrodes % chipWidth),  # this is the column
                  max(all_electrodes % chipWidth) + 1]
        boundX = [int(min(all_electrodes / chipWidth)),  # this is the row
                  int(max(all_electrodes / chipWidth)) + 1]

        self.boundY = boundY
        self.boundX = boundX

        # meshgrid generation based on limits
        x, y = generate_grid(self)

        u = np.zeros(x.shape)
        v = np.zeros(x.shape)
        c = np.zeros(x.shape)

        # For all electrodes with flow (following spikes), store the final flow vector in the uv map
        has_flow = [bool(len(i)) for i in transfer_vector_list]
        for el_idx, el_val in enumerate(transfer_electrode_list):
            # find electrode coordinate
            x_el, y_el = convert_elno_to_xy(el_val)
            x_el = int(x_el - boundX[0])
            y_el = int(y_el - boundY[0])
            if has_flow[el_idx]:
                avg_vecs = np.sum(transfer_vector_list[el_idx], 0)
                weighted_vecs = avg_vecs / np.sum(transfer_vector_weighting[el_idx])
                # If needed we can normalize the vectors
                if normalize:
                    weighted_vecs /= np.linalg.norm(weighted_vecs)
                u[x_el, y_el] = weighted_vecs[0]
                v[x_el, y_el] = weighted_vecs[1]
                c[x_el, y_el] = np.mean(transfer_delay_list[el_idx])
        if full_output:
            return x, y, u, v, c, transfer_electrode_list
        else:
            return x, y, u, v, c

    def spontaneous_pathway_metric_extraction(self, trigger_electrode=None, spike_limit=None, pre_window=0,
                                       total_window_size=300, spatial_window=10, coincidence_window=1, weighting='none',
                                       length_threshold=1, normalize=False, verbose=False):
        """
        Uses the fast_single_window_flow function to extract flow maps around all spike times for the selected trigger
        electrode. For each of these windows, the flow map and some other handpicked maps are returned and can be used
        for clustering of the different pathways.

        Parameters
        ----------
        trigger_electrode: Electrode, around whose spike times the analysis windows will be placed
        spike_limit: Maximal number of spike times to be considered during the analysis
        pre_window: Frames before the trigger spike times which will be considered in the analysis
        total_window_size: Total temporal window to be considered after first frame (including pre window)
        spatial_window: Radius of the circle in which electrodes will be considered as spatially related (in pixels)
        coincidence_window: Time after a spike in which a following spike will be considered related (in ms)
        weighting: Type of weighting for the vectors: None-> equal weights, temporal: based on time difference of spikes
        length_threshold: Length threshold to be used during metric extraction
        normalize: Normalize vector lengths
        verbose: If true, prints detailed algorithm information

        Returns
        -------
        metric_matrix, us, vs, cs
        The metric matrix contains handpicked extracted metrics for all triggered windows. us and vs, contain the flow vectors
        for all electrodes, cs contains the delay information used for coloring.
        """

        # If no trigger electrode is given, we just use the most active one
        if not trigger_electrode:
            trigger_electrode = self.data.mostActiveElectrodes(1)
        # if the user does not define a maximal number of spikes on the trigger electrode to look at,
        # we look at all of them
        trigger_electrode_spikes = self.spike_array[self.spike_array[:, 2] == trigger_electrode, :]

        if not spike_limit:
            spike_limit = trigger_electrode_spikes.shape[0]

        # Matrix which will hold the extracted metrics for every window looked at
        metric_matrix = np.empty((spike_limit, 5))
        metric_matrix[:] = np.nan

        # Prepare list, for all u,v,cs
        us = []
        vs = []
        cs = []

        last_frame = np.max(self.spike_array[:,0])
        start_frames = trigger_electrode_spikes[:,0] - pre_window

        # We iterate over all trigger electrode spike times
        for i in tqdm(range(spike_limit), total=spike_limit):
            # Ensure the window around the current spike doesn't overlap with the last one
            start_frame = start_frames[i]
            if i > 0:
                if start_frame - start_frames[i-1] < total_window_size:
                    if verbose:
                        print('Spikes to close together')
                    us.append([])
                    vs.append([])
                    cs.append([])
                    continue
            # Ensure the start frame makes sense and the window doesn't exeed the last frame
            if start_frame < 0:
                if verbose:
                    print('Invalid start time')
                us.append([])
                vs.append([])
                cs.append([])
                continue
            elif start_frame + total_window_size > last_frame:
                if verbose:
                    print('Invalid end time')
                us.append([])
                vs.append([])
                cs.append([])
                continue

            # calculate the flow for the current window
            x, y, u, v, c, elecs = self.fast_single_window_flow(
                start_frame=start_frame, spatial_window=spatial_window, vector_weighting=weighting,
                temporal_window=total_window_size, coincidence_window=coincidence_window, verbose=verbose,
                normalize=normalize, full_output=True
            )
            us.append(u)
            vs.append(v)
            cs.append(c)

            # Metric extraction
            uv_vecs = np.vstack((u.flatten(), v.flatten())).T
            lenghts = np.array([np.linalg.norm(uv_vecs[i, :]) for i in range(uv_vecs.shape[0])])

            # Number of vectors above length threshold
            try:
                metric_matrix[i, 0] = len(lenghts[lenghts >= length_threshold])
            except:
                pass

            # Total Cumulative vector length
            try:
                metric_matrix[i, 1] = sum(lenghts)
            except:
                pass

            # Cumulative vector distance from first signal
            try:
                metric_matrix[i, 2] = np.linalg.norm(np.sum(uv_vecs, 0))
            except:
                pass

            # Maximal temporal delay
            try:
                metric_matrix[i, 3] = np.max(c) / 20000 * 1000
            except:
                pass

            # Distance between trigger electrode and (temporally) last related electrode
            trigger_position = np.array([int(trigger_electrode/220), trigger_electrode%220])
            temp_coords = [int(np.argmax(c)/c.shape[1]), np.argmax(c)%c.shape[1]]
            temp_coords = np.array([int(temp_coords[0] + self.boundX[0]), int(temp_coords[1] + self.boundY[0])])
            try:
                metric_matrix[i, 4] = np.linalg.norm(trigger_position - temp_coords)
            except:
                pass

        # Remove Nan entries
        mask = np.logical_not(np.isnan(metric_matrix[:, 0]))
        metric_matrix = metric_matrix[mask]

        us = np.array(us, dtype=object)[mask]
        vs = np.array(vs, dtype=object)[mask]
        cs = np.array(cs, dtype=object)[mask]

        return metric_matrix, us, vs, cs

    def stimulated_pathway_metric_extraction(self, spike_limit=None, pre_window=0,
                                       total_window_size=300, spatial_window=10, coincidence_window=1, weighting='none',
                                       length_threshold=1, normalize=False, verbose=False):
        """
        Uses the fast_single_window_flow function to extract flow maps around all blanking window end times.
        For each of these windows, the flow map and some other handpicked maps are returned and can be used
        for clustering of the different pathways.

        Parameters
        ----------
        spike_limit: Maximal number of spike times to be considered during the analysis
        pre_window: Frames before the trigger spike times which will be considered in the analysis
        total_window_size: Total temporal window to be considered after first frame (including pre window)
        spatial_window: Radius of the circle in which electrodes will be considered as spatially related (in pixels)
        coincidence_window: Time after a spike in which a following spike will be considered related (in ms)
        weighting: Type of weighting for the vectors: None-> equal weights, temporal: based on time difference of spikes
        length_threshold: Length threshold to be used during metric extraction
        normalize: Normalize vector lengths
        verbose: If true, prints detailed algorithm information

        Returns
        -------
        metric_matrix, us, vs, cs
        The metric matrix contains handpicked extracted metrics for all triggered windows. us and vs, contain the flow vectors
        for all electrodes, cs contains the delay information used for coloring.
        """

        # For stimulated data, we trigger on the end frames of the blanking windows
        trigger_electrode_spikes = self.data.blankingEnd

        if not spike_limit:
            spike_limit = trigger_electrode_spikes.shape[0]

        # Matrix which will hold the extracted metrics for every window looked at
        metric_matrix = np.empty((spike_limit, 4))
        metric_matrix[:] = np.nan

        # Prepare list, for all u,v,cs
        us = []
        vs = []
        cs = []

        last_frame = np.max(self.spike_array[:,0])
        start_frames = trigger_electrode_spikes - pre_window

        for i in tqdm(range(spike_limit), total=spike_limit):

            start_frame = start_frames[i]
            if i > 0:
                if start_frame - start_frames[i-1] < total_window_size:
                    if verbose:
                        print('Spikes to close together')
                    us.append([])
                    vs.append([])
                    cs.append([])
                    continue

            if start_frame < 0:
                if verbose:
                    print('Invalid start time')
                us.append([])
                vs.append([])
                cs.append([])
                continue
            elif start_frame + total_window_size > last_frame:
                if verbose:
                    print('Invalid end time')
                us.append([])
                vs.append([])
                cs.append([])
                continue

            x, y, u, v, c, elecs = self.fast_single_window_flow(
                start_frame=start_frame, spatial_window=spatial_window, vector_weighting=weighting,
                temporal_window=total_window_size, coincidence_window=coincidence_window, verbose=verbose,
                normalize=normalize, full_output=True
            )
            us.append(u)
            vs.append(v)
            cs.append(c)

            # Metric extraction
            uv_vecs = np.vstack((u.flatten(), v.flatten())).T
            lenghts = np.array([np.linalg.norm(uv_vecs[i,:]) for i in range(uv_vecs.shape[0])])

            # Number of vectors above length threshold
            try:
                metric_matrix[i, 0] = len(lenghts[lenghts >= length_threshold])
            except:
                pass

            # Total Cumulative vector length
            try:
                metric_matrix[i, 1] = sum(lenghts)
            except:
                pass

            # Cumulative vector distance from first signal
            try:
                metric_matrix[i, 2] = np.linalg.norm(np.sum(uv_vecs, 0))
            except:
                pass

            # Maximal temporal delay
            try:
                metric_matrix[i, 3] = np.max(c) / 20000 * 1000
            except:
                pass

        # Remove Nan entries
        mask = np.logical_not(np.isnan(metric_matrix[:, 0]))
        metric_matrix = metric_matrix[mask]

        us = np.array(us, dtype=object)[mask]
        vs = np.array(vs, dtype=object)[mask]
        cs = np.array(cs, dtype=object)[mask]

        return metric_matrix, us, vs, cs
