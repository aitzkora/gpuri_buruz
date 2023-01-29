#Purge implicites rules 
.SUFFIXES:
.SUFFIXES: .pdf .tex

# Tools 
TEX = pdflatex 
BIBMAKE = bibtex
# name of Pdf File 
all:: tp3.pdf

tp3.pdf : tp3.tex 
	$(TEX) $<	

# Clean the directory
clean::clean_pdf clean_aux

clean_pdf: 
	@echo Cleaning pdf File... 
	@rm -f tp3.pdf tp3.out
clean_aux:
	@echo Cleaning Auxiliary files... 
	@rm -f core *.nav *.vrb *.loa *.bbl 
	@rm -f *.aux *.toc *.lof *.log *.snm 
