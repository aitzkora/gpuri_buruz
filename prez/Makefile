#Purge implicites rules 
.SUFFIXES:
.SUFFIXES: .pdf .tex

# Tools 
TEX = pdflatex 
BIBMAKE = bibtex
# name of Pdf File 
all:: c.pdf fortran.pdf

c.pdf : gpu.tex 
	$(TEX) -jobname=c $<	
fortran.pdf : gpu.tex
	$(TEX) -jobname=fortran $<	

# Clean the directory
clean::clean_pdf clean_aux

clean_pdf: 
	@echo Cleaning pdf File... 
	@rm -f c.pdf fortran.pdf
clean_aux:
	@echo Cleaning Auxiliary files... 
	@rm -f core *.nav *.vrb *.loa *.bbl c.out fortran.out
	@rm -f *.aux *.toc *.lof *.log *.snm 
	@rm -f fig/*eps-converted-to.pdf 
