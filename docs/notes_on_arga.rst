
1. Abstraction processes.
 There are only two such processes used:
  a. non-background single-color connected pixels
  b. non-background single-color vertically-connected pixels
 My assumption is that there will be many tens that would cover all of the 800 Tasks.
 Take for example the first Task 007bbfb7. There defining the whole input canvas as an object makes the problem tracktable.

2. Overlapping objects.
 The paper allows object overlapping only on the Output Canvas, because only through a known transformation the algorithm
 will understand that an Object behind another is still a whole Object. We really need to be able to go both ways, where
 an Object overlapped in the Input is understood as such by the fact that it is not so (or the overlap has changed)
 in the Output (or even in other pairs of Canvasses).

3. Full operation.
 That is where the first loop is hiding. Each full operation is a filter, a parameter bindng and a transformation combo.
 It gets applied over all objects.
 The parameter binding operation is where the second loop is hiding. It gets applied over all relevant to the object parameters.
