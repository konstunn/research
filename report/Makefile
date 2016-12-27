
report.pdf: report.tex report.aux report.toc
	pdflatex $<

view: report.pdf
	xdg-open $< || explorer $< &

%.toc: %.tex
	pdflatex $<

%.aux: %.tex
	pdflatex $<

clean:
	rm -f *.pdf
	rm -f *.aux
	rm -f *.log
	rm -f *.toc
	rm -f *.out
