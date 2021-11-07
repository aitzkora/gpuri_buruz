#Purge implicites rules 
.SUFFIXES:
.SUFFIXES: .pdf .tex

# Tools 
TEX = pdflatex 
BIBMAKE = bibtex
# name of Pdf File 


gpu.pdf : gpu.tex 
	$(TEX) $<	
#	$(BIBMAKE) $*
#	$(TEX) $<	
#	$(TEX) $<	

# Clean the directory
clean::clean_pdf clean_aux


clean_pdf: 
	@echo Cleaning pdf File... 
	@rm -f gpu.pdf
clean_aux:
	@echo Cleaning Auxiliary files... 
	@rm -f core *.nav *.vrb *.loa *.bbl gpu.out  
	@rm -f *.aux *.toc *.lof *.log *.snm 
	@rm -f a.out ex_stack ex_thread ex_watch ex_pid ex_core 
	@rm fig/*eps-converted-to.pdf 
	@rm -f dll.so ex_dll ex_struct
