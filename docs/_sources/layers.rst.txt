Layers
======

In this section, we will cover all relevant layers implemented
by Slugnet, and their specific use cases. This includes convolutional
neural networks and layers  associated with them.

Fully Connected Neural Networks
-------------------------------

Slugnet implements fully connected neural networks via the :code:`Dense`
layer. When operating in feedforward mode, the dense layer computes the
following term

.. math::
   :nowrap:

   \[
      \bm{a} = \phi(\bm{W}^T \bm{x} + \bm{b})
   \]

where :math:`\bm{a}` is activated output, :math:`\phi`
is the activation function, :math:`\bm{W}` are weights,
:math:`\bm{b}` is the bias term. The dense layer does not
implement any activation function, instead it is injected
at runtime via the :code:`activation` parameter. This mean
that on feedforward, the dense layer is incredibly simple,
it performs matrix multiplication between an input matrix
and a matrix of weights, then adds a bias vector, and
that's it.

On feed backward, or backpropagation, the dense layer is
responsible for calculating two values. The value defined
:math:`\frac{\partial \ell}{\partial \bm{a}^{(i)}}` will
be used to calculate the weight and bias gradient at this
layer. The value
:math:`\frac{\partial \ell}{\partial \bm{W}^{(i)}}`
will be used to calculate gradients at all previous layers.
This process is easy to follow in the
:ref:`backpropagation <backprop>` algorithm
given in the introduction section of this documentation.

.. math::
    :nowrap:

    \begin{flalign}
        \frac{\partial \ell}{\partial \bm{a}^{(i)}} &=
            \Big[ \bm{W}^{(i + 1)^T}
            \frac{\partial \ell}{\partial \bm{a}^{(i + 1)}}\Big]
            \circ \phi'(\bm{a}^{(i)}) \\
        \frac{\partial \ell}{\partial \bm{W}^{(i)}} &=
            \frac{\partial \ell}{\partial \bm{a}^{(i)}} \bm{x}^T
    \end{flalign}

When looking at the implementation of :code:`Dense`, there is a notable absence
of :math:`\bm{W}^{(i + 1)^T}`
and :math:`\frac{\partial \ell}{\partial \bm{a}^{(i + 1)}}`.
This is because their dot product is calculated in the previous layer.
The model propagates that gradient to the current layer.

.. tikz::

   \tikzset{%
      brace/.style = { decorate, decoration={brace, amplitude=5pt} }
   }

   \draw [brace] (6.25,0.25)  -- (1.75,0.25) node[yshift=-0.5cm, xshift=2.25cm] {Input Layer};
   \draw [brace] (0,5)  -- (8,5) node[yshift=0.5cm, xshift=-4cm] {Dense Layer};

   \foreach \y/\n in {2/3, 4/2, 6/1}
      \draw(\y,1) circle(0.5cm)
      node {$x_{\n}$};

   \foreach \y/\n in {0/1, 2/2, 4/3, 6/4, 8/5}
      \draw[fill=gray!30](\y, 4) circle(0.5cm)
      node {$h_{\n}$};

   \foreach \x in {2,4,6}
      \foreach \y in {0, 2, 4, 6, 8}
         \draw[-{>[scale=4]}, shorten >= 0.55cm, shorten <= 0.5cm](\x,1) -- (\y,4);

   :libs: arrows,calc,positioning,shadows.blur,decorations.pathreplacing,arrows.meta


.. rst-class:: caption

   **Figure 1:** A depiction of a five unit dense layer. The dense layer is
   connected to a three unit input layer. The arrows going from the input layer
   to the dense layer represent weights that are multuplied by the
   values given by the input layer. The resulting values are represented by
   the gray nodes in the hidden dense layer.


.. autoclass:: slugnet.layers.Dense
  :show-inheritance:
  :members:

An example of using two dense layers to train a multi-layer neural
network to classify mnist data can be seen below.

.. literalinclude:: ../slugnet/examples/mnist.py
   :language: python


If you have slugnet installed locally, this script can be
executed by running the following command. It will output
training and validation statistics to :code:`stdout` as the
model is trained.

.. code-block:: shell

   $ python3 -m slugnet.examples.mnist


Note this snippet makes use of several components that have not
yet been reviewed, such as loss and optimization functions.
There are corresponding documentation sections for these components, and
jumping ahead to learn about them is encouraged.

Dropout
-------

Dropout is a method of regularization that trains subnetworks by turning
off non-output nodes with some probability :math:`p`.

This approximates bagging, which involves training an ensemble of models
to overcome weaknesses in any given model and prevent overfitting [1]_.

We can formalize dropout by representing the subnetworks created by dropout
with a mask vector :math:`\bm{\mu}`. Now, we note each subnetwork defines a
new probability distribution of :math:`y` as
:math:`\mathds{P}(y | \bm{x}, \bm{\mu})` [1]_. If we define
:math:`\mathds{P}(\bm{\mu})` as the probability distribution of mask vectors
:math:`\bm{\mu}`, we can write the mean of all subnetworks as

.. math::
    :nowrap:

    \[
        \sum_{\bm{\mu}} \mathds{P}(\bm{\mu}) \mathds{P}(y | \bm{x}, \bm{\mu}).
    \]

The problem with evaluating this term is the exponential number of mask
vectors. In practice, we approximate this probability distribution by
including all nodes during inference, multiplying each output by
:math:`1 - p`, the probability that any node is included in the network during
training, and running the feedforward operation just once. This rule is
called the weight scaling inference rule [1]_.

.. tikz::

   \tikzset{%
      brace/.style = { decorate, decoration={brace, amplitude=5pt} }
   }

   %\draw [brace] (6.25,0.25)  -- (1.75,0.25) node[yshift=-0.5cm, xshift=2.25cm] {Input Layer};
   %\draw [brace] (0,5)  -- (8,5) node[yshift=0.5cm, xshift=-4cm] {Dense Layer};

   \foreach \y/\n in {0/1, 2/2, 4/3, 6/4, 8/5}
      \draw[fill=gray!30](\y, 6) circle(0.5cm)
      node {$h_{\n}^{(2)}$};

   \foreach \y/\n/\c in {0/1/gray, 2/2/red, 4/3/gray, 6/4/gray, 8/5/red}
      \draw[fill=\c!30](\y, 3) circle(0.5cm)
      node {$d_{\n}$};

   \foreach \y/\n in {0/1, 2/2, 4/3, 6/4, 8/5}
      \draw[fill=gray!30](\y, 1) circle(0.5cm)
      node {$h_{\n}^{(1)}$};

   \foreach \x in {0, 2, 4, 6, 8}
      \draw[-{>[scale=4]}, shorten >= 0.55cm, shorten <= 0.5cm](\x,1) -- (\x,3);

   \foreach \x in {0, 4, 6}
      \foreach \y in {0, 2, 4, 6, 8}
         \draw[-{>[scale=4]}, shorten >= 0.55cm, shorten <= 0.5cm](\x,3) -- (\y,6);

   :libs: arrows,calc,positioning,shadows.blur,decorations.pathreplacing,arrows.meta

.. rst-class:: caption

   **Figure 2:** A dropout layer between two hidden layers
   of a neural network. Note the nodes :math:`h_2^{(1)}` and :math:`h_5^{(1)}`
   are both excluded from the current subnetwork via dropout units
   :math:`d_2` and :math:`d_5`. On the next feedforward
   operation, a new subnetwork will be randomly generated with each unit
   in the first layer being exluded from the subnetwork with probability
   :math:`p`.

.. autoclass:: slugnet.layers.Dropout
   :show-inheritance:
   :members:

An example of using a :code:`Dropout` layer with slugnet is presented below.

.. literalinclude:: ../slugnet/examples/mnist_dropout.py
   :language: python


If you have slugnet installed locally, this script can be
run by executing the following command. It will output training
and validation statistics to :code:`stdout` as the model
is trained. Note that this model is slower to train than
the model without dropout. This is widely noted in the
literature [2]_.

.. code-block:: shell

   $ python3 -m slugnet.examples.mnist_dropout


Convolutional Neural Networks
-----------------------------

Convolutional neural networks are most often used in image classification tasks.
There are several specialized layers used in these networks. The most obvious is
the convolution layer, less obvious are pooling layers, specifically max-pooling
and mean-pooling. In this section we will mathematically review all these layers
in depth.

Convolution Layer
~~~~~~~~~~~~~~~~~

In the general case, a discrete convolution operation implements
the function:

.. math::
    :nowrap:

    \[s(i) = \sum_{a=-\infty}^\infty x(a) k(i - a)\]

where :math:`x` is the input and :math:`k`
is the kernel, or in some cases the weighting function.

In the case of convolutional neural networks, the input
is typically a two dimensional image :math:`I`, and it
follows that we have a two dimensional kernel :math:`K`.
Now we can write out convolution function with both axes:

.. math::
    :nowrap:

    \[S(i, j) = \sum_m \sum_n I(m, n) K(i - m, j - n).\]

Note that we can write the infinite sum over the domains of
:math:`m` and :math:`n` as discrete sums because we assume
that the kernel :math:`K` is zero everywhere but the set of
points in which we store data [1]_.

The motivation for using the convolution operation in a
neural network is best described using an example of an
image. In a densely connected neural network, each node
at layer :math:`i` is connected to every node at layer
:math:`i + 1`. This does not lend itself to image processing,
where location of a shape relative to another shape is
important. For instance, finding a right angle involves
detecting two edges that are perpendicular, *and* whose
lines cross one another. If we make the nonzero parts of the
kernel smaller than the input image, we can process parts of
the image at a time, thereby ensuring locality of the input
signals. To process the entire image, we slide the kernel over
the input, along both axes. At each step, an output is produced
which will be used as input for the next layer.
This configuration allows us to learn the parameters of the
kernel :math:`K` the same way we'd learn ordinary parameters
in a densely connected neural network.

.. tikz::

   \tikzset{%
      brace/.style = { decorate, decoration={brace, amplitude=5pt} }
   }

   %\draw [brace] (6.25,0.25)  -- (1.75,0.25) node[yshift=-0.5cm, xshift=2.25cm] {Input Layer};
   %\draw [brace] (0,5)  -- (8,5) node[yshift=0.5cm, xshift=-4cm] {Dense Layer};

   \foreach \y/\n in {0/1, 2/2, 4/3, 6/4, 8/5}
      \draw[fill=gray!30](\y, 3) circle(0.5cm)
      node {$h_{\n}^{(2)}$};

   \foreach \y/\n in {-2/1, 0/2, 2/3, 4/4, 6/5, 8/6, 10/7}
      \draw[fill=gray!30](\y, 1) circle(0.5cm)
      node {$h_{\n}^{(1)}$};

   \draw[-{>[scale=4]}, shorten >= 0.55cm, shorten <= 0.5cm](-2,1) -- (0,3);
   \draw[-{>[scale=4]}, shorten >= 0.55cm, shorten <= 0.5cm](0,1) -- (0,3);
   \draw[-{>[scale=4]}, shorten >= 0.55cm, shorten <= 0.5cm](0,1) -- (2,3);
   \draw[-{>[scale=4]}, shorten >= 0.55cm, shorten <= 0.5cm](2,1) -- (0,3);
   \draw[-{>[scale=4]}, shorten >= 0.55cm, shorten <= 0.5cm](2,1) -- (2,3);
   \draw[-{>[scale=4]}, shorten >= 0.55cm, shorten <= 0.5cm](2,1) -- (4,3);
   \draw[-{>[scale=4]}, shorten >= 0.55cm, shorten <= 0.5cm](4,1) -- (2,3);
   \draw[-{>[scale=4]}, shorten >= 0.55cm, shorten <= 0.5cm](4,1) -- (4,3);
   \draw[-{>[scale=4]}, shorten >= 0.55cm, shorten <= 0.5cm](4,1) -- (6,3);
   \draw[-{>[scale=4]}, shorten >= 0.55cm, shorten <= 0.5cm](6,1) -- (6,3);
   \draw[-{>[scale=4]}, shorten >= 0.55cm, shorten <= 0.5cm](6,1) -- (4,3);
   \draw[-{>[scale=4]}, shorten >= 0.55cm, shorten <= 0.5cm](6,1) -- (8,3);
   \draw[-{>[scale=4]}, shorten >= 0.55cm, shorten <= 0.5cm](8,1) -- (6,3);
   \draw[-{>[scale=4]}, shorten >= 0.55cm, shorten <= 0.5cm](8,1) -- (8,3);
   \draw[-{>[scale=4]}, shorten >= 0.55cm, shorten <= 0.5cm](10,1) -- (8,3);

   :libs: arrows,calc,positioning,shadows.blur,decorations.pathreplacing,arrows.meta

.. rst-class:: caption

   **Figure 3:** Note how the convolution layer connects nodes that are close
   to one another. This closeness is determined by the size of the kernel. In
   this case we have an input in :math:`\mathds{R}^7`, a kernel in :math:`\mathds{R}^3`,
   and an output in :math:`\mathds{R}^5`.

From figure 3, we can see that the output size can be determined from the input
size and kernel size. The equation is given by

.. math::
   :nowrap:

   \[d_{\text{out}} = d_{\text{in}} - d_{\text{kernel}} + 1.\]

Figure 3 features a one dimensional input and output. As we mentioned earlier,
most convolutional neural networks feature two dimensional inputs and outputs,
such as images. In figure 4, we show how the convolution operation behaves
when we are using two dimensional inputs, kernels, and outputs.

.. tikz::

    \def\input {
        0/2.4/a,
        1.2/2.4/b,
        2.4/2.4/c,
        3.6/2.4/d,
        0/1.2/e,
        1.2/1.2/f,
        2.4/1.2/g,
        3.6/1.2/h,
        0/0/i,
        1.2/0/j,
        2.4/0/k,
        3.6/0/l
    }

    \def\kernel {
        0/1.2/w,
        1.2/1.2/x,
        0/0/y,
        1.2/0/z
    }

    \def\output {
        0/-4.6/aw + bx + ey + fz,
        3.4/-4.6/bw + cx + fy + gz,
        6.8/-4.6/cw + dx + gy + hz,
        0/-8/ew + fx + iy + jz,
        3.4/-8/fw + gx + jy + kz,
        6.8/-8/gw + hx + ky + lz
    }

    \draw (0.5,3.8) node {Input};
    \foreach \x/\y/\l in \input
        \draw (\x,\y) -- (\x,\y + 1) -- (\x + 1,\y + 1) -- (\x + 1,\y) -- (\x,\y)
        node[anchor=south west]{$\l$};

    \draw (8.1,2.6) node {Kernel};
    \foreach \x/\y/\l in \kernel
        \draw (\x + 7.6,\y) -- (\x + 7.6, \y + 1) -- (\x + 8.6, \y + 1) -- (\x + 8.6, \y) -- (\x + 7.6, \y)
        node[anchor=south west]{$\l$};

    \draw (0.7,-1.3) node {Output};
    \foreach \x/\y/\l in \output
        \draw (\x,\y) -- (\x,\y + 3) -- (\x + 3,\y + 3) -- (\x + 3, \y) -- (\x,\y)
        node[xshift=1.5cm, yshift=1.5cm]{\footnotesize $\l$};

    \draw [line width=0.4mm](2.3,3.5) -- (4.7, 3.5) -- (4.7, 1.1) -- (2.3, 1.1) -- (2.3, 3.5);
    \draw [line width=0.4mm](7.5,2.3) -- (9.9, 2.3) -- (9.9, -0.1) -- (7.5, -0.1) -- (7.5, 2.3);
    \draw [line width=0.4mm](6.7,-1.5) -- (9.9, -1.5) -- (9.9, -4.7) -- (6.7, -4.7) -- (6.7, -1.5);

    \draw [line width=0.4mm, -|>] (4.7, 2.3) -- (7.0, 2.3) -- (7.0, -1.4);
    \draw [line width=0.4mm, -|>] (8.7, -0.1) -- (8.7, -1.4);

.. rst-class:: caption

    **Figure 4:** An example of a two dimension convolution operation. The
    input is an image in :math:`\mathds{R}^{3 \times 4}`, and the kernel is
    in :math:`\mathds{R}^{2 \times 2}`. As the kernel is slid over the input
    with a stride width of one, an output in
    :math:`\mathds{R}^{2 \times 3}` is produced. In the example, the arrows
    and boxes demonstrate how the upper-right portion of the input image
    are compbined with the kernel parameters to produce the upper right
    unit of output.

    --Modified from source: Goodfellow, Bengio, Courville (Deep Learning,
    2016, Figure 9.1).

The stride width determines how far the kernel moves at each step. Of
course, to learn anything interesting, we require multiple kernels at
each layer. These are all configurable hyperparameters that can be set
upon network instantiation. When the network is operating in feedforward
mode, the output at each layer is a three dimensional tensor, rather than
a matrix. This is due to the fact that each kernel produces its own
two dimensional output, and there are multiple kernels at every layer.

.. autoclass:: slugnet.layers.Convolution
   :show-inheritance:
   :members:

Pooling
~~~~~~~

Mean pooling is a method of downsampling typically used in convolutional
neural networks. Pooling makes the representations at a subsequent layer
approximately invariant to translations of the output from the previous
layer [1]_. This is useful when we care about the presence of some feature
but not necessarily the exact location of the feature within the input.

The mean pooling operation implements the function

.. math::
    :nowrap:

    \[
        \frac{1}{s_m s_n} \sum_{i \in m} \sum_{j \in n} I_{i,j}
    \]

where :math:`m, n` are input ranges along both axes, and :math:`s_n, s_m`
define the size of both ranges. This operation is depicted in figure 2.

.. tikz::

   \def\input {
      0/4/4,
      1/4/7,
      2/4/1,
      3/4/2,
      0/3/6,
      1/3/3,
      2/3/0,
      3/3/8,
      0/2/9,
      1/2/1,
      2/2/6,
      3/2/0,
      0/1/6,
      1/1/4,
      2/1/1,
      3/1/7,
   }

   \def\output {
      6/3/5,
      7/3/2.75,
      6/2/5,
      7/2/3.5
   }

   \draw (0.5, 5.4) node {Input};

   \fill[blue!40!white] (0, 3) rectangle (2,5);
   \fill[red!40!white] (0, 1) rectangle (2,3);
   \fill[green!40!white] (2, 3) rectangle (4,5);
   \fill[orange!40!white] (2, 1) rectangle (4,3);

   \foreach \x/\y/\l in \input
      \draw (\x,\y) -- (\x,\y + 1) -- (\x + 1,\y + 1) -- (\x + 1,\y) -- (\x,\y)
      node[anchor=south west]{$\l$};

   \draw (6.7, 4.4) node {Output};

   \fill[blue!40!white] (6, 3) rectangle (7,4);
   \fill[red!40!white] (6, 2) rectangle (7,3);
   \fill[green!40!white] (7,3) rectangle (8,4);
   \fill[orange!40!white] (7,2) rectangle (8,3);
   \foreach \x/\y/\l in \output
      \draw (\x,\y) -- (\x,\y + 1) -- (\x + 1,\y + 1) -- (\x + 1,\y) -- (\x,\y)
      node[anchor=south west]{$\l$};

   \draw (4.9, 2.95) [align=center] node {Mean \\ pooling};
   \draw [line width=0.4mm, -|>] (4, 3) -- (5.8, 3);

.. rst-class:: caption

   **Figure 5:** A visual representation of the mean pooling
   operation. Color coded patches are combined via arithmetic
   average and included in an output matrix.

The max-pooling operation implements the function

.. math::
   :nowrap:

   \[
      \max_{i \in m, j \in n} I_{i,j}
   \]

where :math:`m, n` are input ranges along both axes. This operation
is depicted in figure 3.

.. tikz::

   \def\input {
      0/4/4,
      1/4/7,
      2/4/1,
      3/4/2,
      0/3/6,
      1/3/3,
      2/3/0,
      3/3/8,
      0/2/9,
      1/2/1,
      2/2/6,
      3/2/0,
      0/1/6,
      1/1/4,
      2/1/1,
      3/1/7,
   }

   \def\output {
      6/3/7,
      7/3/8,
      6/2/9,
      7/2/7
   }

   \draw (0.5, 5.4) node {Input};

   \fill[blue!40!white] (0, 3) rectangle (2,5);
   \fill[red!40!white] (0, 1) rectangle (2,3);
   \fill[green!40!white] (2, 3) rectangle (4,5);
   \fill[orange!40!white] (2, 1) rectangle (4,3);

   \foreach \x/\y/\l in \input
      \draw (\x,\y) -- (\x,\y + 1) -- (\x + 1,\y + 1) -- (\x + 1,\y) -- (\x,\y)
      node[anchor=south west]{$\l$};

   \draw (6.7, 4.4) node {Output};

   \fill[blue!40!white] (6, 3) rectangle (7,4);
   \fill[red!40!white] (6, 2) rectangle (7,3);
   \fill[green!40!white] (7, 3) rectangle (8,4);
   \fill[orange!40!white] (7, 2) rectangle (8,3);
   \foreach \x/\y/\l in \output
      \draw (\x,\y) -- (\x,\y + 1) -- (\x + 1,\y + 1) -- (\x + 1,\y) -- (\x,\y)
      node[anchor=south west]{$\l$};

   \draw (4.9, 2.95) [align=center] node {Max \\ pooling};
   \draw [line width=0.4mm, -|>] (4, 3) -- (5.8, 3);

.. rst-class:: caption

   **Figure 6:** A visual representation of the max pooling
   operation. Color coded patches are downsampled by taking
   the maximum value found in the patch.


.. autoclass:: slugnet.layers.MeanPooling
   :show-inheritance:
   :members:

We have now documented all the necessary parts of a convolutional
neural network. This makes training one to classify mnist data simple.

.. literalinclude:: ../slugnet/examples/mnist_conv.py
   :language: python

Note that because Slugnet is implemented using numpy, and thus
runs on a single CPU core, training this model is very slow.

.. [1] Goodfellow, Bengio, Courville (2016), Deep Learning, Chapter 9,
      http://www.deeplearningbook.org

.. [2] S. Wang and C. D. Manning. Fast dropout training. In *Proceedings of the 30th International
   Conference on Machine Learning*, pages 118–126. ACM, 2013.
