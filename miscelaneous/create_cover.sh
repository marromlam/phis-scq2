#!/usr/bin/env bash


set -eou pipefail


DOT_FILE=/scratch49/forAsier/asier
DOT_FILE=dag
# snakemake --dag > dat.tex
# sed -i "s/rounded/filled/" $DOT_FILE.tex
# sed -i 's/style="rounded"/style="filled", shape="circle"/g' $DOT_FILE.tex
# 's/style="rounded"/style="filled", shape="circle"/g'
# sed -i "s/margin=0/margin=0, model=\"subset\",/g" $DOT_FILE.tex


# neato -Tpdf $DOT_FILE.tex > dag.pdf

/usr/bin/cat $DOT_FILE.tex | grep -v "> 0\|0\[label = \"(.*)?(tab|fig|plot)(.*)?\"" | neato -Tpdf > dag.pdf

# vim:foldmethod=marker
