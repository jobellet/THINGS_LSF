# The Role of Low Spatial Frequencies (LSF) in Shaping Visual Processing

## Background

The magnocellular pathway swiftly transmits low spatial frequency (LSF) and motion information to the visual cortex and also to the prefrontal cortex (PFC), via the superior colliculus and mediodorsal thalamus. Previous research, including my own, has shown that this early burst of activity in the PFC contains significant object-category information.

A central question addressed by this project is whether the category information encoded by the PFC functions as a predictive prior, thereby influencing subsequent visual processing within the ventral visual pathway. If confirmed, this would suggest that category-relevant information derived solely from LSF inputs is transmitted via feedback connections.

We hypothesize that representational geometry related to LSF will emerge as early, or possibly even earlier, in the inferotemporal cortex (IT) compared to intermediate visual regions like V4, or primary visual cortex (V1). Such a finding would challenge the conventional hierarchical view of the ventral visual pathway.

## Methods and Approaches

To investigate this hypothesis, we will utilize EEG, MEG, and neuronal spiking datasets provided by the THINGS initiative ([https://things-initiative.org](https://things-initiative.org)).

### EEG Analysis

Using EEG data, we will perform time-resolved representational geometry analysis to examine whether EEG patterns correlate with representational geometries derived from deep neural network models trained on blurred (LSF-only) images. Our expectation is that this representational similarity (RS) should be significant immediately after image onset and be equal to or stronger than RS derived from early layers of deep networks processing original, high-resolution images. For further validation, we may also compare our results to representational dissimilarity matrices (RDM) obtained from fMRI data recorded in V1, assuming that high-level feedback does not dominate these RDMs.

### MEG Analysis

With MEG data, we will use beamforming localization techniques to study the directionality of category-specific LSF signals along the ventral visual pathway.

### Multi-unit Activity (MUA) Analysis

In the MUA dataset, we will directly compare the timing of LSF signals recorded in IT, V4, and V1 areas to identify the sequence and relative timing of category-specific information across these cortical regions.

