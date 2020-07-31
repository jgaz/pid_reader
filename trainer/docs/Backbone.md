# Backbone build

## Objective

The training for a standard EfficientNet-b0 takes ~32 minutes per epoch for 8 V100s,
it reaches ~76.1% within 350 epochs.

As I only have 2 K80, that means approx 2000hrs for training time, so I have
 no way to train the simplest and smallest model possible of EfficientNet.

Getting a trained EfficientNet backbone doesn't seem to make much sense
since I am going to work on images completely different from a regular
picture, these are large diagrams in black and white which should be much easier
to understand.

So I have to train my own and then rearrange the EfficientDet model that
will sit on top of it.


## Modified architecture
