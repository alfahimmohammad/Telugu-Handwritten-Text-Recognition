The train, val, test files contain the relative path
to the image and it's corresponding label, seperated
by a space.

The telugu_vocab file contains the mapping b/w all the
unique words present in the dataset and a vocab ID.

The lexicon file contains the lexicon that was used
while testing the test set.

Inside the TeluguSeg folder there are the train, val and
test folders. Inside each of these 3 folders there are 
folders which specify the unique writer ID.

Inside each of the unique writer ID folder, there are folders
going from 1,2,3 ... x, where x is the number of pages that
author wrote. 

Inside each of these folders there is a text file, which tells
what the vocab ID for image "n.jpg" on the line number "n".

From the telugu_vocab file we can get the actual label corr.
to that word.