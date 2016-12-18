
main.pdf: main.ltx main.aux 1est.tex 1-2est.tex
	pdflatex $<

view: main.pdf
	xdg-open $< || explorer $< &

%.aux: %.ltx
	pdflatex $<

clean:
	rm -f *.pdf
	rm -f *.aux
	rm -f *.log
	rm -f *.toc
	rm -f *.out
