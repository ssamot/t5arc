s

Features of Objects
====================

Features need to be single numbers or constant length lists.

They need to describe the object itself as well as relationships between the object and other objects.

Individual features (ints unless otherwise specified)
----------------------------------------------------
1. Canvas Position X
2. Canvas Position Y
3. Canvas Position Z

4. Dimension X
5. Dimension Y

6. Number of Colours
7. Number of Pixels with Colour
8. Most common colour
9. Used colours [one hot len 10 list (because there are 10 colours)]

10. Type (of Primitive)
11. Symmetries  # Not Done
12. 2x2 Shape Index
13. 3x3 Shape Index
14. Number of holes
15. Holes sizes [list size 10 (assume max 10 holes per object)]

16. Partially shown on sides  # Not Done
17. Background Object  # Not Done

Relational features [list size 10 (assume max 10 objects per canvas)]
----------------------------------
18. Touches Objects
19. Overlaps Objects
20. Underlaps Objects
21. Fits into Object  # Not Done
22. Matches Objects
23. Matches Objects Only By Shape
24. Matches Objects if Scaled
25. Matches Objects if Scaled Only By Shape
26. Matches Objects if Rotated
27. Matches Objects if Rotated Only By Shape
28. Matches Objects if Flipped
29. Matches Objects if Flipped Only By Shape
30. Matches Objects if Inverted
31. Matches Objects if Inverted Only By Shape

32. Along X with Objects
33. Along Y with Objects
34. Along XY with Objects
35. Along XminusY with Objects
36. Over Objects
37. Under Objects
38. Left of Objects
39. Right of Objects


