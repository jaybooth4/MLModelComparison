#!/bin/bash

rm ./log/*
python3 ./ArtificialDataNN.py
tensorboard --logdir=./log