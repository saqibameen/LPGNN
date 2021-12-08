# Using Node Similarity to Achieve Local Differential Privacy in Graph Neural Networks

This repo contains the artifacts for the CMPUT-622 course project. Baseline code forked and adopted from [**Locally Private Graph Neural Networks**](https://arxiv.org/abs/2006.05535). The implementation for DeepWalk is from https://github.com/phanein/deepwalk/tree/master/deepwalk. Finally, the implementation for Node2Vec was installed from https://github.com/eliorc/node2vec. 

### Students: 

{saqib1, emireddy, jdsteven}@ualberta.ca


## Results:

The `/all_results` directory contains all the results (.csv) files for different similarity methods. `graphs-gen.py` can be used to generate plots from those files. All the graphs included in the report are stored in `/graphs` directory.


## Implementation:

`similarity_methods.py` contains all the similarity methods implementations. `models.py` invokes it to get the similarity matrix and perform aggregation steps.

## Running Jobs:

`cora_job_array.sh` at root contains script for running the jobs on CPU in Compute Canada while `cc_scripts/cora_job_array.sh` contains the script for running jobs on GPU in Compute Canada. The flag `-n` in the script can be changed to specify the dataset, i.e., lastfm, cora. To change the similarity method, Line 28 needs to be changed in `modesl.py` to use the desired similarity methods, i.e., `node2vec`, `deepwalk`, `grarep`.

Depending on the similarity method, datset, and hardware being used (GPU/CPU), the job completion time may vary. 
