   Name		- Number of words (numbers)
1. IAM 		- 85012
2. Camb		- 5260
3. ORAND	- 11719
4. CVL		- 84164
5. Other	- 2460
Total number of words: 188615

All final samples are stored in folders (archives) called 'words-final' under each dataset folder.

The words are stored in form '<word>_<dataset-num>_<timestamp>.png' (Way of labeling can be changed.)
For example: car_1_1528457794.9072268.png
	- file corespons to image of a word 'car' from IAM dataset

The word can contain all english alphabet characters (uppercase, lowercase), 0-9 digits,
and four special characters ('.', '-', "+", "'").
(IAM dataset has some other special characters which can be added.)

If you want to recreate final dataset using Python scripts. You have to download and extract the original dataset files and then run the Python script in same folder.
