wget http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip
unzip -qq cornell_movie_dialogs_corpus.zip
rm cornell_movie_dialogs_corpus.zip
mkdir datasets
mv cornell\ movie-dialogs\ corpus/movie_conversations.txt ./datasets
mv cornell\ movie-dialogs\ corpus/movie_lines.txt ./datasets