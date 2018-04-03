Creative Diversity
===
Simple setup to optimise for diversity whilst maintaining uniqueness.

Here's a [demo](https://www.youtube.com/watch?v=b_T1iQYkQRk) of it getting a bunch of arms to dance uniquely to one another  
Here's a [demo](https://www.youtube.com/watch?v=SvkLbaj4irA) of it's internal structure 

FYI, it can be tweaked to do a lot more than get the arms to dance. For instance, it can be used to create doodles, make funny facial expressions, etc. I would love to see someone take it for a spin! Or maybe even help out with the To Do List.

How it works (Geek only section)
---
Take any environment where you can perform an action that has an observable effect on the environment.  
Build a model that acts on this environment in n different ways such that it itself can easily distinguish between these n ways of interacting given the observable effects it has on the environment.

Thus, it should work well irrespective of what the environment represents, bet it typing, painting, animating, navigating, etc.

To Do List
 ---
 - Use the image as input instead of the internal state (arm joint states) to optimise for **visually** distinct dancing rather than quantitatively **distinct** dancing! After all, this project is more about the eye candy ;)
 - Use a variational auto-encoder instead to increase diversity