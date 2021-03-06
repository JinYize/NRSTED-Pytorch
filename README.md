# NR-STED Pytorch Version

NR-STED is an opinion-unaware and distortion-unaware (OU-DU) no reference (NR) video quality assessment algorithm. This method was proposed in the paper: S. Mitra, R. Soundararajan and S. S. Channappayya, "Predicting Spatio-Temporal Entropic Differences for Robust No Reference Video Quality Assessment," in IEEE Signal Processing Letters, vol. 28, pp. 170-174, 2021, doi: 10.1109/LSP.2021.3049682.

This is a pytorch implementation of NR-STED. 

## Preparation

To train from scratch, please follow these steps to prepare for training:
* Download matlabPyrTools/ on https://github.com/LabForComputationalVision/matlabPyrTools.
* Download the CSIQ VQA database.
* Use prepare_CSIQ_SRRED.m and prepare_CSIQ_TRRED.m in the MATLAB/ folder to convert the CSIQ VQA database and compute per-frame SRRED and TRRED indices, and link the converted database to the folder databases/.

## Training NR-SRRED from Scratch

```console
python SRRED.py -n 1 -g 4 -nr 0 --epochs 20
```
## Training NR-TRRED from Scratch
```console
python TRRED.py -n 1 -g 4 -nr 0 --epochs 20
```
