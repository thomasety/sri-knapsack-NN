# sri-knapsack-NN
This is a esearch project for the Harvard student research institute summer program for high schoolers.

Our goal during this research project is to find out how well neural networks can perform on “non-deterministic polynomial time” (NP) problems compared to standard algorithms. 
If a problem is NP, that means we do not know how to find a solution in polynomial time, but we can verify a solution in polynomial time. 
This is why NP problems are so hard as non polynomial time is extremely slow and in certain cases simply takes too long even for powerful computers to compute a solution 
(If we used all the computing power on earth to try to solve the knapsack problem through brute force, it would take tens of thousands of years)

We will study the knapsack problem, which itself is a NP complete problem.
The knapsack problem is an optimization problem, where we are given a knapsack bag that can carry a maximum amount of weight, and a list of items who each have a given weight and value. 
Our goal is to maximize the total value of items we put in the bag without going over the weight limit of the knapsack bag.
We will use the 0/1 version of this problem which means any item can only be taken once or not at all.

Normally, this problem is solved using a tabulation algorithm which consistently finds the exact optimal solution, 
but this can take a large amount of time if there is a large amount of item to choose from. 
Our goal is to see if neural networks can also find optimal solutions while taking less time for larger sets of items. 
To do so, we first need to train the neural network with both features (the max weight of the bag, the value and weight of each item) and the optimal output (which items we put in the bag).
We will generate both features and output using the tabulation algorithm (as it gives the exact optimal solution)
Finally, we will compare the outputs of the neural networks with the tabulation algorithm and see how it performs.
Provided there is time, we will  add some noise to the item values and weights to see how neural networks are affected compared to a tabulation algorithm. 
