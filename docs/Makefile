# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

APIDOCOPTS = --separate --implicit-namespaces --module-first -d 1 --templatedir source/_templates/autodoc/

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

apidoc:
	@sphinx-apidoc $(APIDOCOPTS) -o source/api_documentation ../hippynn

.PHONY: apidoc

cleanapi:
	@echo "Removing generated API files..."
	@rm -f source/api_documentation/hippynn*.rst
	@rm -f source/api_summary/hippynn*.rst
.PHONY: cleanapi

# make sure api_doc is always built first
html_all: apidoc html
.PHONY: html_all

clean_all: cleanapi clean
.PHONY: clean_all

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD)  -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


