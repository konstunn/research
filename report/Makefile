
main.pdf: main.ltx main.aux main.toc
	pdflatex $<

view: main.pdf
	xdg-open $< || explorer $< &

%.toc: %.ltx
	pdflatex $<

%.aux: %.ltx
	pdflatex $<

clean:
	rm -f *.pdf
	rm -f *.aux
	rm -f *.log
	rm -f *.toc
	rm -f *.out
