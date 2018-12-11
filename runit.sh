#!/bin/bash
echo "enter 1 to run fnc_doc2vec.py "
echo "enter 2 to run fnc_mlp.py "
echo "enter 3 to run fnc_rf.py "
echo "enter 4 to run fnc_tfidf.py "
echo -n "Please enter a number to run the file"
read i
if [ $i -eq 1 ]
then
 python fnc_doc2vec.py train_bodies.csv train_stances.csv test_bodies.csv test_stances_unlabeled.csv
elif [ $i -eq 2 ]
then
 python fnc_mlp.py train_bodies.csv train_stances.csv test_bodies.csv test_stances_unlabeled.csv
elif [ $i -eq 3 ]
then
 python fnc_rf.py train_bodies.csv train_stances.csv test_bodies.csv test_stances_unlabeled.csv
elif [ $i -eq 4 ]
then
 python fnc_tfidf.py train_bodies.csv train_stances.csv test_bodies.csv test_stances_unlabeled.csv
else
 echo "wrong option"
fi
