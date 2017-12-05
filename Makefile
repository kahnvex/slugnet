.PHONY: docs

test:
	nosetests --nocapture

docs-auto:
	sphinx-autobuild -z slugnet/ docs-src/ docs/

docs:
	sphinx-build docs-src/ docs/
