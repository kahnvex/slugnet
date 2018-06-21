.PHONY: docs

test:
	pipenv run nosetests --nocapture

docs-auto:
	pipenv run sphinx-autobuild -z slugnet/ docs-src/ docs/

clean-docs-images:
	rm -rf docs/plot_directive
	rm -rf docs/_images

docs: clean-docs-images
	pipenv run sphinx-build -a -E docs-src/ docs/
