#Purge implicites rules 
.SUFFIXES:
.SUFFIXES: .pdf .tex

# Tools 
TEX = pdflatex 
BIBMAKE = bibtex
# name of Pdf File 
all:: tp1.pdf

tp1.pdf : tp1.tex 
	$(TEX) $<	

# Clean the directory
clean::clean_pdf clean_aux

clean_pdf: 
	@echo Cleaning pdf File... 
	@rm -f tp1.pdf tp1.out
clean_aux:
	@echo Cleaning Auxiliary files... 
	@rm -f core *.nav *.vrb *.loa *.bbl 
	@rm -f *.aux *.toc *.lof *.log *.snm 
