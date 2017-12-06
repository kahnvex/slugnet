Slugnet
=======

Slugnet is a modest expiremental neural networks library intended to solidify
the author's understanding of deep learning.

The goal of this library is to mathematically document all relevant components
of a working neural networks library. This includes models, layers, optimizers,
activation functions, loss functions, forward propogation, backward
propogation, and more.

Before looking at any code, the following diagram will introduce the notation
styles this library will follow. In general, a neural network tries to
approximate some function :math:`f^*`, where :math:`y = f^*(x)`. The neural
network implements a function :math:`\hat{y} = f(x)`. We say a neural network
is fully connected if each node in a given layer is connected to every node
in the adjacent layer. For now, we will only consider fully connected neural
networks.

When making predictions, a neural network is said to be operating in
feedforward mode. For now, we will inspect how neural networks operate in
feedforward mode.

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

   **Figure 1:** A three layer, fully connected neural network. The first layer
   has five hidden units. The superscript number in parenthesis indicates the
   layer of the unit. The index in subscript represents the unit's index. For
   example :math:`h_3^{(4)}` represents the third unit of the forth layer.


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


.. rst-class:: caption

   **Figure 2:** The same three layer network as in Figure 1, represented
   in a shorthand form where to units of each layer are collapsed onto
   one circle.

Let's "zoom in" on one of the layers to see what is happening under the hood
when our neural network is running in feedforward mode. The layer
:math:`f^{(i)}(x)` performs the computation

.. math::

   \bm{z} = g(\bm{W}^{(i)^T} \bm{x} + \bm{b}^{(i)}).


In this diagram, :math:`\bm{z}` is output, :math:`g` represents the activation
function, :math:`\bm{W}^{(i)^T}` represents a matrix of weights at this layer,
:math:`\bm{b}^{(i)}` represents a vector of bias terms at this layer, and
:math:`\bm{x}` represents the input at this layer.

Neural networks rely on a nonlinear activation function to learn nonlinear
functions. Without a nonlinear activation function, a neural network is nothing
more than a linear model. There are several choices one can make for activation
functions, including but not limited to tanh, sigmoid, and the rectified linear
unit, or ReLU for short.


API Documentation
-----------------

.. toctree::
   :maxdepth: 4

   index
   layers
   loss
