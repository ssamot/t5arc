
Class Pixel
A Pixel is a square that can have one of 10 different colours (1 to 10). 1 is Black and is considered Not Coloured.
0 is considered Not Existent.

Class Object.
An Object is a Class that defines a group of Coloured Pixels (called the actual_pixels array) that can exist in one or
multiple Canvasses and that can change through certain allowed Transformations.
The Object has a canvas_pos providing the coordinates of the bottom, left Pixel of the Object in the Canvas it is in.
It also has a bounding_box defined as the top. left and bottom right coordinates of the box that fully encapsulates the
Object's Coloured Pixels.
It also has dimensions defining the length of the bounding_box in the x and y axis.
Finally is has a colour property that defines the colour of its majority Pixels.

The Transformations allowed for the Object are the following:
1) Translation along the x and y axis of the Canvas.
2) Rotation by 90 degrees (so 4 rotations will cause no change).
3) Scale by an integer amount (called a factor which can be positive or negative). If it is positive then each Pixel
gets multiplied on both x and y axis by that factor. If it is negative then for every factor pixels, factor -1 get deleted.
4) Shear, which shears the Object by a factor between 0 and 1.
5) Flip, where the Object gets flipped in place either along the x or the y axis.
6) Mirror where the Object gets copied, the copy Flipped and moved along the axis of the Flip, thus creating a symmetric
new Object made of two mirror images of the original Object connected together.
7) Change of the Colour of the Coloured Pixels.
8) Addition or subtraction of Coloured Pixels (thus changing the Shape of the Object).


Class Primitive.
The Object Class is a Parent to a subclass called Primitive.

Primitive Subclasses.
There are several Classes that are a subclass of the Primitive that define Primitives (Objects) with specific Shapes.
These are:
1) Dot (a single Pixel).
2) Random (a random group of Pixels, coloured randomly which define no obviously recognisable Shape).
3) Parallelogram.
4) Cross.
5) Hole (a Parallelogram with a Black hole inside it).
6) Pi (the Shape pi).
7) Inverse Cross (a Cross rotated 45 degrees and filled partially with coloured pixels between its arms).
8) Angle (a rotated or flipped L Shape).
9) Diagonal (a Line of Pixels that is at 45 or 135 degrees to the x axis).
10) Steps (a 90 degrees equilateral triangle where the vertices of the 90 degrees angle are parallel to the x, y axis).
11) Fish (a [[0, C, C], [C, C, C], [0, C, 0]] pattern of Coloured Pixels that look like a fish).
12) Bolt (a [[C, C, 0], [C, 0, C], [0, C, C]] pattern).
13) Tie (a [[C, 0, 0], [0, C, C], [0, C, C]] pattern).
14) Spiral (a single Line Spiral).
15) Pyramid (a 90 degrees equilateral triangle where the vertex opposite the 90 degrees angle is parallel to the x axis).

Class Canvas
A Canvas is a 32 x 32 grid of Pixels. A N x M subgroup of these, always starting at 0, 0 have Pixels that are
Coloured from 1 to 10. The remaining Pixels are set to 0. The M x N subgroup is considered the Canvas actual_pixels array
while the whole Canvas grid is the full_pixels array. The M and N numbers define the size of the Canvas in the x and y
direction (always in the order, first x then y).

The Canvas can hold a number of Objects (in its objects list) and will paint its actual_array Pixels according to
which Objects it holds and what are their canvas_pos coordinates. SO it will show the Objects it holds onto its Pixel
grid. It has methods to:
1) Add Objects.
2) Remove Objects.
3) Order the Objects by size (area or length or width).
4) Split Objects according to their Coloured Pixels creating from one Object multiple new ones with uniform Colours.
4) Position an Object onto a new position.
5) Group Objects by their colour property.
6) Return the Object with specific colour properties.
7) Return all of the Pixel coordinates that are Coloured in the actual_pixels array of teh Canvas.

Class Task.
An Task is a Class that consists of a number of Input - Output pairs of Canvasses. These are saved in the
input_canvasses and output_canvasses lists of the Task. There is a specific logic common to all the pairs which
allows the transformation of each Input Canvas to its corresponding Output Canvas.

The Task Logic.
Each Task's Logic has one or two elements.
First, the Transformations. These are Object Transformations of some of the Objects in an Input Canvas that lead to
the Input Canvas looking like the Output one.
Secondly (but not always) the triggers. These are attributes or combinations of attributes of the Input Canvas and
(much more usually) the Input Canvas Objects. These attribute (combinations) inform the Logic which of all the possible
Object Transformations should be achieved and what should be those Transformations' arguments.