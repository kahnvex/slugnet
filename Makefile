.PHONY: docs

test:
	nosetests --nocapture

docs-auto:
	sphinx-autobuild -z slugnet/ docs/source docs/build

docs:
	sphinx-build docs/source docs/build
