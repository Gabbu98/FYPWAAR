# makefile for ${docname}
docname=sig-alternate

${docname}: ${docname}.tex
	pdflatex --shell-escape ${docname}.tex
	bibtex ${docname}
	pdflatex --shell-escape ${docname}.tex
	pdflatex --shell-escape ${docname}.tex
	@echo ""
	@echo "--------------------------------"
	@echo "${docname}.tex compiled"
	@echo "--------------------------------"
	@echo ""


clean:
	rm -f ${docname}.aux ${docname}.bbl ${docname}.div ${docname}.log ${docname}.nav ${docname}.out ${docname}.snm ${docname}.toc ${docname}.vrb
	@echo "-----------------------------------------------------------------------------------"
	@echo "Intermediate files generated when compiling ${docname}.tex removed"
	@echo "-----------------------------------------------------------------------------------"
	@echo ""

cleanAll: clean
	rm -f ${docname}.pdf
	@echo "-----------------------------------------------------------------------------------"
	@echo "Intermediate and output files generated when compiling ${docname}.tex removed"
	@echo "-----------------------------------------------------------------------------------"
	@echo ""

cleanall: cleanAll

