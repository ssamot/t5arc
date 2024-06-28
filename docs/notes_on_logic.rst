
Notes on Logic
===============

Objects
-------

The :ref:`ConceptARC` describes a set of core ideas with which to build ARC samples.
But as they mention it is not feasible to build automatic generators to create samples
from these ideas. I believe this is because they do not have a generic enough generator
that understands what an object is. All of their concepts are based on the fact that
someone understands the different objects in the images and then applies the core concept
to the objects.

**Assumption #1:**
Without correct object identification (or generation) no other logic works.

The idea of objectness is given as one of the possible High-level knowledge priors in
:ref:`MeasureOfIntelligence`:

*High-level knowledge priors regarding objects and phenomena in our external environment.
This may include prior knowledge of* :strong:`visual objectness` *(what defines an object),
priors about orientation and navigation in 2D and 3D Euclidean spaces, goal directedness
(expectation that our environment includes agents that behave according to goals), innate
notions about natural numbers, innate social intuition (e.g. theory of mind), etc.*

Yet we propose that there is a hierarchy of priors where some priors can only be defined if
others already exist. We also propose that for the given ARC tasks the basis prior is
objectness and once this is understood (i.e. an algorithm can tell apart the objects appropriate
for the task) then all other required priors can be built on top.

The idea here is to first create networks that can identify those (task appropriate) objects.
Then create a DSL using those as basic inputs (priors) thus defining other concepts like affine
transformations, natural numbers, ordering etc. Finally train networks that can put those DSL
primitives together correctly.

What are (task appropriate) objects?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**Requirements for objectness**

#. A task appropriate object exists as a group of pixels only as far as this group is same/similar to another group of pixels in the same or another picture.
#. Hello


The whole point of being shown three or four examples per task is to first use the similarities
of groups of pixels between these to identify what constitutes useful objects.

So the idea of an object is fully hinged upon the ideas of sameness and similarity.

**Assumption #2:**
If we identify all of the principles humans use to understand sameness/similarity and are
able to create generators following these principles then we can train networks to correctly
identify objects.


Defining sameness and similarity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Tricks:

#. Same shape and colour
#. Same shape different colour
#. Same colour different shape
#. Same shape after affine transform (translate, rotate, scale)
#. Almost same colour (some noise)
#. Almost same shape (some noise)
#. Inverting colours gives same shape (really difficult)
#. Existing after other object subtraction (really difficult)
#. Same symmetries (need to be able to detect pattern repetition in all scales)
#. Same colour patterns (could that be a compound trick)
#. Combinations of the above. E.g . Scale and almost same colors (large shape made of smaller ones).


Symmetries
^^^^^^^^^^^^
An idea that might come in handy is to also define axes of symmetry for objects that have it. Algorithms
that do this are had to put together (even for the case of simple images like the ones in ARC
so it might be better if we train networks not only to detect an object but also generate its axes of symetry.


What should the networks output (as objects)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The network should be trained to output the following:
#. The bounding box of an object on all images.
#. Any axes of symmetry on all images.



Growth
------

Another basic prior seems to be the concept of growth. Lines grow along their axes, 2D shapes
grow along one or more of their own axes of symmetry or along an axes of symmetry of the image.
This cannot be encompassed by the idea of objects and we need a separate set of networks to detect it.

The growth idea requires detection of:

#. Directions, magnitudes and points of origin of growth. This requires the detection of vectors with a starting point along which the growth can happen. Such vectors need to be learned by the network.
    #. Axes of symmetry
    #. Lines between two points
#. Tiles the growth uses (objects that repeat to generate the growth). Those will be denoted by some code given by a combination of markers (at least one denoting the shape of the tile and one the color pattern).
