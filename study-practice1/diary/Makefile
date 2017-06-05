
diary.pdf: diary.tex diary.aux
	pdflatex $<

view: diary.pdf
	xdg-open $< || explorer $< &

%.aux: %.tex
	pdflatex $<

clean:
	rm -f *.pdf
	rm -f *.aux
	rm -f *.log
	rm -f *.toc
	rm -f *.out
