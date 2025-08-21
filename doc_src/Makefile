# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile github reqs

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

github:
	@echo "Building HTML documentation and copying to ../docs..."
	# Delete existing files in the docs directory
	rm -rf ../docs/*
	# Generate new HTML documentation
	$(SPHINXBUILD) -M html $(SOURCEDIR) $(BUILDDIR) $(SPHINXOPTS) $(O)
	# Copy the newly built HTML documentation to the docs directory
	cp -r $(BUILDDIR)/html/* ../docs

reqs:
	@echo "Running build requirements script..."
	@if [ -f "$(SOURCEDIR)/build_reqs.py" ]; then \
		echo "Using $(SOURCEDIR)/build_reqs.py"; \
		python "$(SOURCEDIR)/build_reqs.py" $(ARGS); \
	elif [ ! -z "$(BUILD_REQS)" ]; then \
		echo "Using BUILD_REQS environment variable: $(BUILD_REQS)"; \
		python "$(BUILD_REQS)" $(ARGS); \
	else \
		echo "Error: build_reqs.py not found and BUILD_REQS environment variable not set"; \
		exit 1; \
	fi

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)