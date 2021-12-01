

def load_data(experiment, train, batchsize, resample_val, shuffle, seed_val, seq_len, neur_count):

    '''
    INPUT
    experiment     int from 1-4 chooses experiment (1: MC_Maze, 2: MC_RTT, 3: Area2_Bump, 4: DMFC_RSG)
    train          bool true -> train, false -> test
    batchsize      int determines amount of trials at once
    resample_val   int determines factor of resampling,    1 returns original, once resampled need to redownload to get original (delete folder)
    shuffle        bool shuffles trials
    seq_len        int length of individual sequences,     0 for full length
    neur_count     int count of neurons per trial,         0 for full length

    OUTPUT
    neuron_id        2d int-array of neuron ids in data
    trial_id       1d int-array of trial ids in data
    data           3d numpy array of trials x neurons x sequences
    '''

    import os
    import numpy as np
    import pandas as pd

    ## Download dataset and required packages if necessary
    os.system('pip install git+https://github.com/neurallatents/nlb_tools.git')
    os.system('pip install dandi')

    from nlb_tools.nwb_interface import NWBDataset


    if experiment == 1:
        if not os.path.isdir("000128"):
            print("Downloading data")
            os.system('dandi download https://dandiarchive.org/dandiset/000128/draft')

        if train:
            dataset = NWBDataset("000128/sub-Jenkins/", "*train", split_heldout=False)
        else:
            dataset = NWBDataset("000128/sub-Jenkins/", "*test", split_heldout=False)

    elif experiment == 2:
        if not os.path.isdir("000129"):
            print("Downloading data")
            os.system('dandi download https://dandiarchive.org/dandiset/000129/draft')

        if train:
            dataset = NWBDataset("000129/sub-Indy", "*train", split_heldout=False)
        else:
            dataset = NWBDataset("000129/sub-Indy", "*test", split_heldout=False)

    elif experiment == 3:
        if not os.path.isdir("000127"):
            print("Downloading data")
            os.system('dandi download https://dandiarchive.org/dandiset/000127')

        if train:
            dataset = NWBDataset("000127/sub-Han/", "*train", split_heldout=False)
        else:
            dataset = NWBDataset("000129/sub-Indy", "*test", split_heldout=False)

    elif experiment == 4:
        if not os.path.isdir("000130"):
            print("Downloading data")
            os.system('dandi download https://dandiarchive.org/dandiset/000130/draft')

        if train:
            dataset = NWBDataset("000130/sub-Haydn/", "*train", split_heldout=False)
        else:
            dataset = NWBDataset("000129/sub-Indy", "*test", split_heldout=False)

    else:
        print("Experiment only 1-4")


    # Seed generator for consistent plots
    np.random.seed(seed_val)


    dataset.resample(resample_val)

    # Smooth spikes with 50 ms std Gaussian
    dataset.smooth_spk(50, name='smth_50')

    dataset = dataset.make_trial_data()

    trial_ids = np.unique(dataset['trial_id'])
    if shuffle: np.random.shuffle(trial_ids)
    trial_ids = trial_ids[0:batchsize]

    neuron_ids = np.array(dataset['spikes'].keys().tolist())
    np.random.shuffle(neuron_ids)

    if neur_count == 0:
        neur_count = len(neuron_ids)

    neuron_ids = neuron_ids[0:neur_count]


    if seq_len == 0:
        seq_len = len(dataset['spikes'][neuron_ids[0]])


    data = np.zeros((batchsize, seq_len, neur_count))


    for i in range(batchsize):
        trial_data = dataset[dataset['trial_id'] == trial_ids[i]]
        trial_data = trial_data['spikes'][neuron_ids][0:seq_len]

        data[i,:,:] = trial_data

    return data, neuron_ids, trial_ids


if __name__ == "__main__":
    data, neuron_ids, trial_ids = load_data(experiment=3, train=True, batchsize=3, resample_val=1, shuffle=False, seed_val=111, seq_len=10, neur_count=5)

    print(data.shape)
    print(neuron_ids.shape)
    print(trial_ids.shape)
