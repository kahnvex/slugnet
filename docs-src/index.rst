Slugnet
=======

Slugnet is a modest expiremental neural networks library intended to solidify
the author's understanding of deep learning.

The goal of this library is to mathematically document all relevant components
of a working neural networks library. This includes models, layers, optimizers,
loss functions, forward propogation, backward propogation, and more.

Before looking at any code, the following diagram will introduce the notation
styles this library will follow. In general, a neural network tries to
approximate some function :math:`f^*`, where :math:`y = f^*(x)`. The neural
network implements a function :math:`\hat{y} = f(x)`.

.. tikz::

   \tikzset{%
      brace/.style = { decorate, decoration={brace, amplitude=5pt} }
   }

   \draw [brace] (0.5,7)  -- (1.5,7) node[yshift=0.5cm, xshift=-0.5cm] {Input};
   \draw [brace] (3.5,9)  -- (7.5,9) node[yshift=0.5cm, xshift=-1.9cm] {Hidden Layers};
   \draw [brace] (9.5,6)  -- (10.5,6) node[yshift=0.5cm, xshift=-0.5cm] {Output};

   \foreach \x/\n in {2/3, 4/2, 6/1}
      \draw(1,\x) circle(0.5cm)
      node {$x_{\n}$};

   \foreach \x/\n in {0/5, 2/4, 4/3, 6/2, 8/1}
      \draw[fill=gray!30](4, \x) circle(0.5cm)
      node {$h_{\n}^{(1)}$};

   \foreach \x/\n in {1/4, 3/3, 5/2, 7/1}
      \draw[fill=gray!30](7, \x) circle(0.5cm)
      node {$h_{\n}^{(2)}$};

   \foreach \x/\n in {3/2, 5/1}
      \draw (10, \x) circle(0.5cm) node {$y_{\n}$};

   \foreach \x in {2,4,6}
      \foreach \y in {0, 2, 4, 6, 8}
         \draw[-{>[scale=4]}, shorten >= 0.55cm, shorten <= 0.5cm](1,\x) -- (4,\y);

   \foreach \x in {0,2,4,6,8}
      \foreach \y in {1,3,5,7}
         \draw[->, shorten >= 0.55cm, shorten <= 0.5cm](4,\x) -- (7,\y);

   \foreach \x in {1,3,5,7}
      \foreach \y in {3,5}
         \draw[->, shorten >= 0.55cm, shorten <= 0.5cm](7,\x) -- (10,\y);

   :libs: arrows,calc,positioning,shadows.blur,decorations.pathreplacing,arrows.meta


.. rst-class:: caption

   **Figure 1:** A three layer neural network. The first layer has five hidden units.
   The superscript number in parenthesis indicates the layer of the unit. The index
   in subscript represents the unit's index. For example :math:`h_3^{(4)}`
   represents the third unit of the forth layer.


We can write the above network as :math:`\hat{y} = f^{(3)}(f^{(2)}(f^{(1)}(x)))`.
Additionally, we may represent the network with the shorthand diagram below.

.. tikz::

   \tikzset{%
      brace/.style = { decorate, decoration={brace, amplitude=5pt} }
   }

   \draw [brace] (0.5,2)  -- (1.5,2) node[yshift=0.5cm, xshift=-0.5cm] {Input};
   \draw [brace] (3.5,2)  -- (7.5,2) node[yshift=0.5cm, xshift=-1.9cm] {Hidden Layers};
   \draw [brace] (9.5,2)  -- (10.5,2) node[yshift=0.5cm, xshift=-0.5cm] {Output};

   \draw(1,1) circle(0.5cm) node {$\boldmath{x}$};
   \draw(4,1)[fill=gray!30]circle(0.5cm) node {$\boldmath{h}^{(1)}$};
   \draw(7,1)[fill=gray!30] circle(0.5cm) node {$\boldmath{h}^{(2)}$};
   \draw(10,1) circle(0.5cm) node {$\boldmath{y}$};

   \draw[->, shorten >= 0.55cm, shorten <= 0.5cm](1,1) -- (4,1);
   \draw[->, shorten >= 0.55cm, shorten <= 0.5cm](4,1) -- (7,1);
   \draw[->, shorten >= 0.55cm, shorten <= 0.5cm](7,1) -- (10,1);

   :libs: arrows,calc,positioning,shadows.blur,decorations.pathreplacing,arrows.meta,bm

API Documentation
-----------------

.. toctree::
   :maxdepth: 4

   index
   layers
   loss
