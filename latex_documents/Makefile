.PHONY: clean

.ONESHELL:
.SHELLFLAGS := -ec

# Stray LaTeX artifacts left next to the sources by a hand-run build. Kept on
# purpose: .build/ (latexmk cache), the PDFs and the .synctex.gz.
clean:
	EXTS='aux|log|out|fls|fdb_latexmk|bbl|blg|toc|lof|lot'
	find . -not -path '*/.build/*' -type f -regextype posix-extended -regex ".*\.($$EXTS)$$" -print -delete
