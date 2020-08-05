environment required: Python3, MacOS

Make sure have python3 installed on a MacOS, input command "pip3 install -r requirements.txt" to configure all libararies required in the project.

"Submission.csv" contains the results of classification, after "running sentiment.py" each time, it will be updated.

To start the learning, run "python3 sentiment.py" in the terminal, it might take a few minutes to finish, after that, a graph will pop out.


Introduction of the problem
The dataset is comprised of tab-separated files with phrases from the Rotten Tomatoes dataset. The train/test split has been preserved for the purposes of benchmarking, but the sentences have been shuffled from their original order. Each Sentence has been parsed into many phrases by the Stanford parser. Each phrase has a PhraseId. Each sentence has a SentenceId. Phrases that are repeated (such as short/common words) are only included once in the data.

train.tsv contains the phrases and their associated sentiment labels. We have additionally provided a SentenceId so that you can track which phrases belong to a single sentence.
test.tsv contains just phrases. You must assign a sentiment label to each phrase.
The sentiment labels are:

0 - negative
1 - somewhat negative
2 - neutral
3 - somewhat positive
4 - positive

