Slugnet
=======

Slugnet is a modest experimental neural networks library intended to solidify
the author's understanding of deep learning.

The goal of this library is to mathematically document all relevant components
of a working neural networks library. This includes models, layers, optimizers,
activation functions, loss functions, forward propagation, backward
propagation, and more.

The mathematical documentation assumes basic understanding of discriminative
machine learning techniques, linear algebra, probability, and calculus. This
documentation will loosely follow the notation found in *Deep Learning*
(Goodfellow, Bengio, & Courville, 2016).

Before looking at any code, the following sections will introduce the notation
styles this library will follow as well as give a brief mathematical
introduction to neural networks. In general, a neural network tries to
approximate some function :math:`f^*`, where :math:`\bm{y} = f^*(\bm{x})`. The
neural network implements a function :math:`\hat{\bm{y}} = f(\bm{x})`, where
:math:`\hat{\bm{y}}` represents the prediction made by the network, and
:math:`f` represents the model. We say a neural network is fully connected if
each node in every layer is connected to every node in the adjacent layer. For
now, we will only consider fully connected neural networks.


-----------------
Table of Contents
-----------------

.. toctree::
   :maxdepth: 2

   index
   model
   layers
   activation
   loss
   optimizers

----------------
Feedforward Mode
----------------

When making predictions, a neural network is said to be operating in
feedforward mode. For now, we will inspect how neural networks operate in
this mode.

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
      \draw (10, \x) circle(0.5cm) node {$\hat{y}_{\n}$};

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


We can write the network in figure 1 as
:math:`\bm{\hat{y}} = f(\bm{x}) = f^{(3)}(f^{(2)}(f^{(1)}(\bm{x})))`. Each layer
:math:`f^{(i)}` is composed of the layer that came before it,
:math:`f^{(i - 1)}`, the first layer :math:`f^{(1)}` takes the input
:math:`\bm{x}`. Variables that are lowercase and bold represent vectors, and
variables that are capitalized and bold represent matrices. Additionally, we may
represent the network with the shorthand diagram below.

.. tikz::

   \tikzset{%
      brace/.style = { decorate, decoration={brace, amplitude=5pt} }
   }

   \draw [brace] (0.5,2)  -- (1.5,2) node[yshift=0.5cm, xshift=-0.5cm] {Input};
   \draw [brace] (3.5,2)  -- (7.5,2) node[yshift=0.5cm, xshift=-1.9cm] {Hidden Layers};
   \draw [brace] (9.5,2)  -- (10.5,2) node[yshift=0.5cm, xshift=-0.5cm] {Output};

   \draw(1,1) circle(0.5cm) node {$\boldmath{x}_{1:3}$};
   \draw(4,1)[fill=gray!30]circle(0.5cm) node {$\boldmath{h}^{(1)}_{1:5}$};
   \draw(7,1)[fill=gray!30] circle(0.5cm) node {$\boldmath{h}^{(2)}_{1:4}$};
   \draw(10,1) circle(0.5cm) node {$\hat{y}_{1:2}$};

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
:math:`f^{(i)}(x)` performs the computation defined in equation 1.

.. math::

   \bm{a}^{(i)} = \phi(\bm{W}^{(i)^T} \bm{x} + \bm{b}^{(i)})

.. rst-class:: caption

   **Equation 1:** Definition of computation performed in one layer of a
   neural network. In this equation, :math:`\bm{a}^{(i)}` is the activated
   output, :math:`\phi` represents the activation function,
   :math:`\bm{W}^{(i)^T}` represents a learned matrix of weights at this
   layer, :math:`\bm{b}^{(i)}` represents a learned vector of bias terms at
   this layer, and :math:`\bm{x}` represents the input at this layer.

Neural networks rely on a nonlinear activation function to learn nonlinear
relationships. Without a nonlinear activation function, a neural network is
nothing more than a linear model. There are several choices one can make for
activation functions, including but not limited to tanh, sigmoid, and the
rectified linear unit, or ReLU for short.

Upon completion of the feedforward operation, the prediction :math:`\hat{y}`
is output from the final layer.

Slugnet represents a neural network as a :code:`Model`. You can run a
neural network in feedforward mode by calling :code:`model.transform(X)`
on a model, where :code:`X` is a matrix of inputs. In this case :code:`X`
is a matrix to allow users of Slugnet to make several predictions
in one call to :code:`model.transform`. Before you can run a model
in feedforward mode, it must be trained. This leads us to backpropogation and
optimization.

---------------------------------------
Loss, Backpropogation, and Optimization
---------------------------------------

Training a neural network is similar to training traditional discriminative
models such as logistic regression. For instance, we need a loss function, we
must compute derivatives, and we must implement some numerical algorithm to
optimize the model. On the other hand, neural networks are somewhat unique in
that they require us to compute a gradient at each layer with which we may
learn weights. To compute this gradient, we use the backpropogation algorithm.

Before we can run backpropogation, a version of the feedforward algorithm
described earlier must be run, only instead of throwing away the intermediate
outputs at each layer, we store them, knowing that we'll need them later
for backpropogation. Additionally, during training, we require the ground truth
labels or values of each sample. That is, the dataset :math:`\mathcal{D}`
consists of :math:`\{\bm{x}_i, \bm{y}_i\}_{i=1}^N`, where :math:`N` is the
number of samples, and :math:`\bm{y}_n` is the ground truth label or output
value for sample :math:`\bm{x}_n`.

Loss Functions
~~~~~~~~~~~~~~

Upon completion of the forward pass on a batch of inputs, we can compute the
loss for the batch using the predicted outputs, :math:`\hat{\bm{y}}`, and
the ground truth labels or values :math:`\bm{y}`.

.. math::

      \bm{\ell}(\bm{\hat{y}}, \bm{y}) = -\frac{1}{N}
         \sum_{i=1}^N \big[
            \bm{y}_i \log(\hat{\bm{y}}_i) + (1 - \bm{y}_i) \log(1 - \hat{\bm{y}}_i)
         \big]

.. rst-class:: caption

   **Equation 2:** Binary cross entropy loss function.

If the outputs that we are learning are binary labels, then we might use
a binary cross entropy loss function, seen in equation 2. On the other hand, if
we are learning labels with multiple classes, we might use categorical cross
entropy. The resulting loss value will inform us about how our network
performed on the batch it just predicted. We can use this value along with
validation to determine if our model is overfitting or underfitting the data.
In the next section, we'll see that the derivative of our loss function is
used to perform backpropogation.

Backpropogation
~~~~~~~~~~~~~~~

Backpropogation involves computing gradients for the weights :math:`\bm{W}^{(i)}`
and bias :math:`\bm{b}^{(i)}` for all layers :math:`i \in \{1, 2, \dots, l\}`
where :math:`l` is the number of layers in our network. Once we've computed
these gradients, the model can use a numerical optimization method to adjust
weights and bias terms in such a way that error is reduced. Before defining the
gradients of our weights and bias terms, we must define how to compute loss
gradient, and the gradient at each layer.


.. math::

   \bm{g}^{(\text{Loss})} &= \bm{g}^{(\ell)} = \nabla_{\hat{\bm{y}}}\bm{\ell}(\bm{\hat{y}}, \bm{y})

.. rst-class:: caption

   **Equation 3:** Defines how we compute the gradient of the loss function,
   which is the first gradient computed during backpropogation. From this
   gradient, we will compute all other gradients.

Once the gradient of the loss function is calculated, we may begin performing
backpropogation on the layers of our neural network. We start from the "top"
of the network, or the output layer. Using the loss gradient
:math:`\bm{g}^{(L)}` we can compute the gradient of the output layer as
defined in equation 3. The definition given in equation 4 is generalized, that
is, it applies to any hidden layer in the network.

.. math::

   \bm{g}_{\text{activated}}^{(i)} &= \bm{g}_a^{(i)} = \bm{g}^{(i)} \circ \phi'(\bm{a}^{(i)})

.. rst-class:: caption

   **Equation 4:** The definition of our activation gradient at layer :math:`i`.
   The variable :math:`\bm{a}^{(i)}` reprsenets the activated output at layer
   :math:`i` and :math:`\phi'` represents the derivative of the activation
   function. The unfilled dot (:math:`\circ`) represents an item-wise
   multiplication between two vectors. It can also be used to represent item-wise
   multiplication between two matrices.


Now, we have all we need to define the gradients of our weights and bias term.

.. math::

      \nabla_{\bm{W}^{(i)}}\bm{\ell} &= \bm{g}_a^{(i)}\, \bm{h}^{(i-1)^T} \\
      \nabla_{\bm{b}^{(i)}}\bm{\ell} &= \bm{g}_a^{(i)}

.. rst-class:: caption

   **Equation 5:** This equation defines the gradients of weight and bias terms,
   :math:`\bm{W}^{(i)^T}` and :math:`\bm{b}^{(i)}`. In this equation,
   :math:`\bm{h}^{i-1}` is the ouput from layer :math:`i - 1`.

The only part of the computation that is missing is that of
:math:`\bm{g}^{(i+1)}` for the next layer in the backpropogation algorithm.
This is definted in equation 3, and we can now see a recursive method of
computing gradients from layer to layer.

.. math::

      \bm{g}^{(i-1)} = \bm{W}^{(i)^T} \bm{g}_a^{(i)}

.. rst-class:: caption

   **Equation 6:** How to propogate the gradient from layer :math:`i` to layer
   :math:`i-1`.

This is all we need to implement a full backpropogation algorithm. Repeated
application of equations 3, 4, and 5 will give us the weight and bias
gradients :math:`\nabla_{\bm{W}}\bm{\ell}` and :math:`\nabla_{\bm{b}}\bm{\ell}`
at every layer, as indicated backpropogation's pseudocode given in
algorithm 1.

.. _backprop:

.. rst-class:: algo
.. math::
   :nowrap:

   \begin{algorithm}
      \caption{Backward Propogation \newline
      --Modification of source: Goodfellow, Bengio, \& Courville (Deep Learning, 2016)}
      \label{backprop}
      \begin{algorithmic}[1]
         \Procedure{Backpropogation}{$\bm{\ell}, \bm{\hat{y}}, \bm{y}, \bm{h}, \bm{W}$}
            \State $\bm{g} \gets \nabla_{\bm{\hat{y}}}\bm{\ell}(\bm{\hat{y}}, \bm{y})$
            \For{$i=l, l-1, \dots 1$}
               \State $\bm{g} \gets \bm{g} \circ \phi'(\bm{a}^{(i)})$
               \State $\nabla_{\bm{W}^{(i)}}\bm{\ell} = \bm{g}_a^{(i)} \, \bm{h}^{(i-1)}$
               \State $\nabla_{\bm{b}^{(i)}}\bm{\ell} = \bm{g}_a^{(i)}$
               \State $\bm{g} \gets \bm{W}^{(i)^T} \bm{g}$
            \EndFor
            \Return $\langle \nabla_{\bm{W}}\bm{\ell}, \nabla_{\bm{b}}\bm{\ell} \rangle$
         \EndProcedure
      \end{algorithmic}
   \end{algorithm}


Optimization
~~~~~~~~~~~~

Next, we can use the gradients computed in backpropogation (algorithm 1) to
compute weight updates for each layer using a numerical optimization method.

In this section, we will focus on a version of the stochastic gradient descent (SGD)
optimization method called mini-batch gradiant descent. This methods works by
taking a sample of size :math:`m` from the training set
:math:`\{\bm{x}_i, \bm{y}_i\}_{i=1}^N`, computing the gradients with
backpropogation, and applying our update using a learning rate parameter
:math:`\epsilon`. In practice, we must gradually decrease :math:`\epsilon` over
time.

.. rst-class:: algo
.. math::
   :nowrap:

   \setcounter{algorithm}{1}
   \begin{algorithm}
      \caption{Mini-Batch Gradient Descent pseudocode}
      \label{backprop}
      \begin{algorithmic}[1]
         \Procedure{SGD}{$\bm{\ell}, \bm{x}, \bm{y}, m$}
            \State $\bm{W} \gets \text{InitWeights}()$
            \State $\bm{b} \gets \text{InitBias}()$
            \While{not converged}
               \For{i = 0, m, 2m, 3m, \dots, N}
                  \State $\epsilon \gets \text{NextEpsilon}(\epsilon)$
                  \State $\langle \bm{\hat{y}}_{i:m}, \bm{h} \rangle \gets
                     \text{FeedForward}(\bm{x}_{i:m}, \bm{y}_{i:m}, \bm{W}, \bm{b})$
                  \State $\langle \nabla_{\bm{W}}\bm{\ell}, \nabla_{\bm{b}}\bm{\ell} \rangle \gets
                     \frac{1}{m} \sum\limits_{j=i}^{i+m}
                     \text{Backpropogation}(\bm{\ell}, \bm{\hat{y}}_j, \bm{y}_j, \bm{h}, \bm{W})$
                  \For{$k = 1, 2, \dots, l$}
                     \State $\bm{W}^{(k)} \gets \bm{W}^{(k)} -
                        \epsilon \nabla_{\bm{W}^{(k)}}\bm{\ell}$
                     \State $\bm{b}^{(k)} \gets \bm{b}^{(k)} -
                        \epsilon \nabla_{\bm{b}^{(k)}}\bm{\ell}$
                  \EndFor
               \EndFor
            \EndWhile
            \Return $\langle \bm{W}, \bm{b} \rangle$
         \EndProcedure
      \end{algorithmic}
   \end{algorithm}

In practice, we will decouple optimization methods from the backpropogation and
feedforward algorithms in order to make a modular system of components that can
be easily mixed and matched. This process is fairly straightforward and will be
apparent as components are documented.
