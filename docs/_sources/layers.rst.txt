Layers
======

In this section, we will cover all relevant layers implemented
by slugnet, and their specific use cases. This includes convolutional
neural networks and layers  associated with them.

Fully Connected Neural Networks
-------------------------------

Slugnet implements fully connected neural networks via the :code:`Dense`
layer. The :code:`Dense` layer implements the feed forward operation

.. math::
   :nowrap:

   \[
      \bm{a} = \phi(\bm{W}^T \bm{x} + \bm{b})
   \]

where :math:`\bm{a}` is activated output, :math:`\phi`
is the activation function, :math:`\bm{W}` are weights,
:math:`\bm{b}` is our bias.

On feed backward, or backpropogation, the :code:`Dense` layer
calculates two values as follows

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

When looking at the source, there is a notable absence of
:math:`\bm{W}^{(i + 1)^T}`
and :math:`\frac{\partial \ell}{\partial \bm{a}^{(i + 1)}}`.
This is because their dot product is calculated in the previous layer.
The model propogates that gradient to this layer.

.. autoclass:: slugnet.layers.Dense
  :show-inheritance:
  :members:

Dropout
-------

Dropout is a method of regularization that trains subnetworks by turning
off non-output nodes with some probability :math:`p`.

This approximates bagging, which involves training an ensemble of models
to overcome weaknesses in any given model [1]_.

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
including all nodes during inference, and multiplying each output by
:math:`1 - p`, the probability that any node is included in the network during
training. This rule is called the weight scaling inference rule [1]_.

.. autoclass:: slugnet.layers.Dropout
   :show-inheritance:
   :members:

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

    \[s(i) = \sum_{a=-\infty}^\infty x(a) w(i - a)\]

where :math:`x` is the input and :math:`w`
is the kernel, or in some cases the weighting function.

In the case of convolutional neural networks, the input
is typically two dimensional image :math:`I`, and it
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
lines cross one another. If we make the kernel smaller
than the input image, we can process parts of the image
at a time, thereby ensuring locality of the input signals.
To process the entire image, we slide the kernel over the
input, along both axes. At each step, an output is produced
which will be used as input for the next layer.
This configuration allows us to learn the parameters of the
kernel :math:`K` the same way we'd learn ordinary parameters
in a densely connected neural network.

.. tikz::

    \def\input {
        0/2.4/a,
        1.2/2.4/b,
        2.4/2.4/c,
        0/1.2/d,
        1.2/1.2/e,
        2.4/1.2/f,
        0/0/g,
        1.2/0/h,
        2.4/0/i
    }

    \def\kernel {
        0/1.2/w,
        1.2/1.2/x,
        0/0/y,
        1.2/0/z
    }

    \def\output {
        0/-4.6/aw + bx + dy + ez,
        3.4/-4.6/bw + cx + ey + fz,
        0/-8/dw + ex + gy + hz,
        3.4/-8/ew + fx + hy + iz
    }

    \draw (0.5,3.8) node {Input};
    \foreach \x/\y/\l in \input
        \draw (\x,\y) -- (\x,\y + 1) -- (\x + 1,\y + 1) -- (\x + 1,\y) -- (\x,\y)
        node[anchor=south west]{$\l$};

    \draw (5.5,2.6) node {Kernel};
    \foreach \x/\y/\l in \kernel
        \draw (\x + 5,\y) -- (\x + 5, \y + 1) -- (\x + 6, \y + 1) -- (\x + 6, \y) -- (\x + 5, \y)
        node[anchor=south west]{$\l$};

    \draw (0.7,-1.3) node {Output};
    \foreach \x/\y/\l in \output
        \draw (\x,\y) -- (\x,\y + 3) -- (\x + 3,\y + 3) -- (\x + 3, \y) -- (\x,\y)
        node[xshift=1.5cm, yshift=1.5cm]{\footnotesize $\l$};

    \draw [line width=0.4mm](1.1,3.5) -- (3.5, 3.5) -- (3.5, 1.1) -- (1.1, 1.1) -- (1.1, 3.5);
    \draw [line width=0.4mm](4.9,2.3) -- (7.3, 2.3) -- (7.3, -0.1) -- (4.9, -0.1) -- (4.9, 2.3);
    \draw [line width=0.4mm](3.3,-1.5) -- (6.5, -1.5) -- (6.5, -4.7) -- (3.3, -4.7) -- (3.3, -1.5);

    \draw [line width=0.4mm, -|>] (3.5, 2.3) -- (4.0, 2.3) -- (4.0, -1.4);
    \draw [line width=0.4mm, -|>] (6, -0.1) -- (6, -1.4);

.. rst-class:: caption

    **Figure 1:** An example of a two dimension convolution operation. The
    input is an image in :math:`\mathds{R}^{3 \times 3}`, and the kernel is
    in :math:`\mathds{R}^{2 \times 2}`. As the kernel is slid over the input
    with a stride width of one, an output in
    :math:`\mathds{R}^{2 \times 2}` is produced. In the example, the arrows
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

   **Figure 2:** A visual representation of the mean pooling
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

   **Figure 3:** A visual representation of the max pooling
   operation. Color coded patches are downsampled by taking
   the maximum value found in the patch.


.. [1] Goodfellow, Bengio, Courville (2016), Deep Learning, Chapter 9,
      http://www.deeplearningbook.org
