.PHONY: docs

test:
	nosetests --nocapture

docs-auto:
	sphinx-autobuild -z slugnet/ docs-src/ docs/

clean-docs-images:
	rm -rf docs/plot_directive

docs:
	sphinx-build docs-src/ docs/
