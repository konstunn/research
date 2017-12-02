
.PHONY: main.pdf

main.pdf: main.tex
	latexmk --shell-escape -pdf $<

view: main.pdf
	xdg-open $< > /dev/null 2>&1 &

clean:
	latexmk -f -C main.pdf
