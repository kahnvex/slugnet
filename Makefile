.PHONY: docs

test:
	nosetests --nocapture

docs-auto:
	sphinx-autobuild -z slugnet/ docs-src/ docs/

clean-docs:
	rm -rf docs/	

docs:
	sphinx-build docs-src/ docs/
