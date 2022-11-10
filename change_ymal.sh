#!/bin/bash

filename='/home/561/ts7017/s3prl/s3prl/downstream/ctc/libriphone.yaml'

search=LibriSpeech
replace=$(pwd)

sed -i "s|$search|$replace|" $filename
