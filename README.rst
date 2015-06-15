eventgen - CUDA event data generator
====================================

Small utility to generate event data from consecutive image frames. The images
are compared pixel by pixel and as soon as a specific threshold in difference is
reached, this pixel will 'emit' an event.

To speed up the process the computation is implemented using CUDA
