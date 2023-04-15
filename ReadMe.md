## What Every File Contains:

`Auto.py` - Clear Procced Data Folders, Run All Scripts Automaticlly, Run Model Training

`Convert.py` - Convert Data From MP3 to Wav

`Preprocces.py` - Make All Files in Same Length With The Maximum File (For Ex. all with 4 Second)

`spectrogram.py` - convert all Data TO spectrogram

`cnn.py` - train the CNN Model

`Mic.py` - Run The Mic and Detect The Sample

## Notes:

1. the Dataset in `data` folder in being preproccesd to the Fixed Legnth by adding Silnace to The Original Sound File (For ex. all files are converted to 4 second as the longest Sample).
2. The Preprosseced Data is Being transformed to Spectrograms into `Spectro` Folder
3. Then We Split Manualing the Data into traingngset (in `spectro` Folder as it is), testset in `TestSpectro` Folder and Validationset in `Validat` Folder
4. then the `cnn.py` Auto detect the Classes from traing and test sets, and then Train The Model and Save the Final Trained Model in `Models` Folder
5. the rest of the Folders and Files are just for trying random Things (they are not important)

You Can See The Dataset in Drive: 
[Drive Link](https://drive.google.com/drive/folders/1WR7zvQTTnycdMmPV2PP4CaTFhawdiQp1?usp=sharing){:target="_blank" rel="noopener"}
