# LaTeX AFM Report

This project contains a LaTeX document that compiles findings from AFM, histogram, and PASD analyses. The structure of the project is organized to facilitate easy management of sections, figures, tables, and references.

## Project Structure

- **src/**: Contains the main LaTeX source files.
  - **main.tex**: The main entry point for the LaTeX document.
  - **sections/**: Contains individual section files for the report.
    - **introduction.tex**: Outlines the purpose and scope of the findings.
    - **methodology.tex**: Details the methodology used in the analysis.
    - **results.tex**: Presents the results of the findings, including figures and tables.
    - **conclusion.tex**: Summarizes the findings and conclusions drawn from the analysis.
  - **figures/**: Directory for storing figures related to the findings.
    - **afm/**: Figures related to AFM findings.
    - **histograms/**: Figures related to histogram findings.
    - **pasd/**: Figures related to PASD findings.
  - **tables/**: Directory for storing any tables included in the report.
  - **bibliography/**: Contains the bibliography file.
    - **references.bib**: BibTeX file listing all references cited in the report.

- **output/**: Directory for storing compiled output files (e.g., PDF).

- **AFMpgf/**: Contains PGF figures generated from the AFM data.

- **Histogrampgf/**: Contains PGF figures generated from the histogram data.

- **PASDpgf/**: Contains PGF figures generated from the PASD data.

- **Makefile**: Instructions for building the LaTeX document, specifying how to compile the .tex files and manage dependencies.

## Compilation Instructions

To compile the LaTeX document, navigate to the project directory and run the following command:

```
make
```

This will generate the output files in the `output/` directory. Ensure that you have a LaTeX distribution installed (such as TeX Live or MiKTeX) to successfully compile the document.

## Additional Information

For any questions or contributions, please refer to the project maintainers or the documentation provided in the respective section files.