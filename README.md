# MINOR filter implementation

This software implements the MINOR algoritm. Code is written in CUDA.

# Requirements
- CUDA SDK (optional, recommended)
- OpenMP (if CUDA not provided, recommended)
- libpng

# Installation
You can define in Makefile if you will use the CUDA framework or not, by default no. To use the CUDA framework add `CUDA=1` to make command

Run:

```
make lib CUDA=1
make main CUDA=1
```

# Usage
Sample usage:
`./main_minor <reference image { rgb }> <noisy image {rgb}> <block_radius> <alpha> <beta> <sigma> <f>`
where:

`Reference image - original, not noisy image`\
`Noisy image - image which will be filtered`\
`block_radius - radius of the processing block B`\
`alpha - number of first pixels being choosen from patch`\
`beta - number of first pixels being choosen from means`\
`sigma - smoothing parameter`\
`f - patch size`\

# Acknowledgment

This code uses parts of the Fourier 0.8 library by Emre Celebi licensed on GPL avalaible here:

http://sourceforge.net/projects/fourier-ipal

http://www.lsus.edu/faculty/~ecelebi/fourier.htm

