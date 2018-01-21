.PHONY: docs, integration

integration:
	nosetests --nocapture -w integration/

docs-auto:
	sphinx-autobuild -z slugnet/ docs-src/ docs/

clean-docs-images:
	rm -rf docs/plot_directive
	rm -rf docs/_images

docs: clean-docs-images
	sphinx-build -a -E docs-src/ docs/
