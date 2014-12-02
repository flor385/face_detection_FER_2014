#! usr/bin/bash

pdflatex --shell-escape rv_report.tex
bibtex rv_report.aux
pdflatex --shell-escape rv_report.tex
pdflatex --shell-escape rv_report.tex
open rv_report.pdf

#	brisemo sav meta shit
rm *.dv *.bbl *.aux *blg *.dvi *.log *.gz *.toc
