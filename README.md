﻿
1   Filter the vad 

    -   by running vad_processing/vad_processing.ipynb
    -   fitered vad files should be put in the path preprocess/audio/filter_vad/


2   Using the filtered vad files to generate the samples, ground truth label and write them into csv files
    
    -   by running preprocess/audio/generate_samples.py 
        
        #main(x, y, z, vad_dict) 
            -   x : experiment num (0 indicates generating training samples)
            -   y : window size
            -   z : ratio of positive samples to negative samples

    -   make sure generate training samples first.

3   Making pkl files for training and corresponding experiments' samples

    -   by running data_loading/make_examples.py

        -   generate training samples  
            -   make_all_examples(0, windowSize)

        -   generate samples for experiment 1
            -   make_all_examples(1, windowSize, numberOfExperiment)

        -   generate samples for experiment 2
            -   make_all_examples(2, windowSize, numberOfExperiment)

        -   generate samples for experiment 3
            -   make_all_examples(3, windowSize, numberOfExperiment, 'all_unsuccessful')
        
        -   generate samples for experiment 4
            -   make_all_examples(4, windowSize, numberOfExperiment, 'start')

        -   generate samples for experiment 5
            -   make_all_examples(5, windowSize, numberOfExperiment, 'continue')


4   Execute the training 

    -   running baseline/testTrain.py

        Train:
        -    main(True, 0, windowSize)

        Test for different experiments
        -    main(False, experiment_number, windowSize, number_of_experiments_repeated)





        
