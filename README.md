MASSIVE BIG DATA
=============

Repo for our CS229 project

#General Obeservations
**Video Duration:** 8 seconds of video takes about 5 minutes to process on an MBP Retina

**Video Types:** Matlab's VideoReader works for .mp4 and .mov, but .mov seems to silently break the script and the output will look identical to the input.

**Noise:** (ex. people walking by) seems to have a phase wraparound that fades over time, meaning that people will continue to walk past you over and over gradually fading each time...

**window_size:** the narrower the frequency window is, the less likeley the script will do anything (2: nothing happens, 2.5: nothing happens, 2.8: does nothing, 3: works great, 4: very pronounced). There are some weird effects with the window size not working then suddenly working. We need to figure that out.

#File Descriptions
**daves_notes.m** Scratch paper, largely ignorable

**daves_tests.m** At the top is a working implementation that runs in the videos directory, at the bottom is an attempt to fully deconstruct the processing pipeline. This file will eventually be broken into pieces.


