
# Makefile for LaTeX AFM Report

LATEX = pdflatex
MAIN = complete_report
SRCDIR = src
OUTDIR = output

.PHONY: all clean view

all: $(OUTDIR)/$(MAIN).pdf

$(OUTDIR)/$(MAIN).pdf: $(SRCDIR)/$(MAIN).tex
    @mkdir -p $(OUTDIR)
    cd $(SRCDIR) && $(LATEX) -output-directory=../$(OUTDIR) $(MAIN).tex
    cd $(SRCDIR) && $(LATEX) -output-directory=../$(OUTDIR) $(MAIN).tex

clean:
    rm -rf $(OUTDIR)/*

view: $(OUTDIR)/$(MAIN).pdf
    "$$BROWSER" $(OUTDIR)/$(MAIN).pdf

help:
    @echo "Available targets:"
    @echo "  all    - Build the PDF report"
    @echo "  clean  - Remove output files"
    @echo "  view   - Open the PDF in browser"
    @echo "  help   - Show this help message"
