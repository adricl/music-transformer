### Adrics Notes


## To Play Midi
TiMidity
or 
wildmidi


## Data
- [ ] [MIDI Dataset from google Maestro](https://magenta.tensorflow.org/datasets/maestro)
- [ ] [MIDI Dataset ADL Piano MIDI](https://github.com/lucasnfe/adl-piano-midi) based of the https://colinraffel.com/projects/lmd/

https://github.com/asigalov61/Tegridy-MIDI-Dataset

https://miditok.readthedocs.io/en/latest/pytorch_data.html for encoding midi for datasets and doing augs, this would be good for extending what we already have




### to pre process data
python preprocessing.py data/adl-piano-midi/ midi_processed/processed_data_adl.pt 100 -v
python preprocessing.py data/adl-piano-midi/ midi_processed/processed_data_adl_1000_tokens.pt 1000 -v
python preprocessing.py data/adl-piano-midi/ midi_processed/processed_data_adl_2000_tokens.pt 2000 -v

python preprocessing.py data/maestro-v3.0.0/ midi_processed/processed_data_masetro_2000_tokens.pt 2000 -v


### train
python train.py midi_processed/processed_data_adl.pt model/cpkt.pt model/save.pt

With data processed_data_adl_100.pt it takes ~12485.14 per epoch or 3.46 hours per epoch on a 4090 before updates to code

After pytorch 2.1 complile update we now take 4183.8s per epoch or 1.16 hours per epoch on a 4090

-bs 40 seems to fit into the 24GB of memory on the 4090

python train.py midi_processed/processed_data_adl.pt model/cpkt_all.pt model/save_all.pt 10 -l -bs 38
python train.py midi_processed/processed_data_adl_2000_tokens.pt model/cpkt_all.pt model/save_2000_all.pt 10 -l -bs 40

Epoch 0 Time taken 4183.8 seconds Train Loss 1.131267336885463 Val Loss 1.0878991668874567
Epoch 1 Time taken 4062.28 seconds Train Loss 1.123966256255311 Val Loss 1.089039704843001
Epoch 2 Time taken 3968.21 seconds Train Loss 1.1182763313470976 Val Loss 1.0785238943533464
Epoch 3 Time taken 4233.83 seconds Train Loss 1.1135272768895128 Val Loss 1.0777936141057447


### generate music with existing music
python generate.py model/save.pt test.midi -v -t 1 -k 250 -f "data/adl-piano-midi/Rap/Alternative Hip Hop/Oliver/As Long As She Loves Me.mid" -ft 150 -c

python generate.py model/save.pt test.midi -v -t 1 -k 250 -i "data/adl-piano-midi/Rap/Alternative Hip Hop/Oliver/As Long As She Loves Me.mid" -it 150 -c






### Installing MidiTok

CXX=/usr/bin/clang++-12 #this is reuqired for the symusic install
pin install miditok

