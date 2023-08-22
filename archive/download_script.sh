#!/bin/bash

wget https://raw.githubusercontent.com/luonglearnstocode/Seinfeld-text-corpus/master/corpus.txt

awk '/^JERRY: / { sub(/^JERRY: /, ""); print }' corpus.txt > jerry.txt

rm corpus.txt