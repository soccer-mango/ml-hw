# Assignment 4 - Machine Learning!

  

## Part 3 - Written Questions

  

1.  **Question**: Explain your K-Means algorithm. What are the parts of your algorithm, and how do they work together to divide your samples into different clusters?

	**Answer**: *The algorithm revolves around the process of organizing each data point into groups (k number of groups) based on the proximity of the values comprising each data point. The components of the algorithm are ultimatelt its functions, the two most important being closest_centroids and compute_centroids. The first finds the closest group for each data point, the latter then computes the values which define each individual group. These processes utilize the euclidean distance function in order to quantify the proximity of data points, and uses an initializing function to automatically generate groups at random.*

------------------------------  

2.

- **Question**: What is each data point that is being divided into clusters? what does each cluster represent?

	 **Answer**: *In the image compression context, data points are pixels... represented by the values for R, G, and B. Each cluster represents the colors which are present in the picture.*

  

- **Question**: How does changing the number of clusters impact the result of the song dataset and the image compression dataset?

	**Answer**: *Changing the number of clusters in the song dataset creates more defined "genres." You could think of 8 clusters including just "rock," per se, whereas 16 clusters may include "soft rock" and "hard rock." In the image compression dataset, the number of clusters is the number of colors present in the final image. The more clusters, the closer the ouptut image will match the original. The downfall of more clusters is that the process will be much slower and require more storage.*

------------------------------

3.

- **Question**: What is the difference between supervised classification, supervised regression, and unsupervised learning?

	**Answer**: *Supervised classification is aimed at predicting labels of groups for datasets. The classification uses examples given to it in order to create groups. Examples include K-Neareast Neighbor, Decision Tree, Naive Bayesm etc.
	Supervised regression is used to predict a continuous value. The algorithms have a loss function which is minimized through analysis of training data. Common alogithms are Linear Regression, Logistic Regression, Elastic Net Regression.
	Unsupervised learning aims to find representations in the data given no guidance. *

- **Question**: Give an example of an algorithm under each, specifying the type of data that the algorithm takes in and the goal of the algorithm, and an explanation for why they are a supervised classification/supervised regression/unsupervised algorithm.

	**Answer**: *One unsupervised learning algorithm is K-Nearest Neighbor. This assignment focuses on K-Nearest Neighbor which takes in data points and seeks to group them based on k number of groups and characteristics which define the data points. An algorithm for supervised regression is linear regression which aims to predict how changes in one variable effect another variable. An supervised learning algorithm is a decision tree. Decision trees take in often tabular data and "branch" into a series of nodes an leafs. Based on one characteristic, a data point will follow a certain branch until another characteristic is considered, and so on. This is more or less a sorting algorithm for prediction.*

------------------------------

4. **Question**: Give an overview of how you would modify your Kmeans class to implement Fair K-Means in  `kmeans.py`. Describe any methods you would add and where you would call them. You don’t need to understand the mathematical details of the Fair K-Means algorithm to have a general idea of the key changes necessary to modify your solution.

	**Answer**: *I would create a method called sensitive_attributes() which would evaluate the sensitiveness of certain attributes. The values computed in this function would be designed to ensure that in the process of clustering, sensitive attributes have far less of an effect.*

------------------------------

5. **Question**:  How does the Fair K-means algorithm define fairness? Describe a situation or context where this definition of fairness might not be appropriate, or match your own perception of fairness.

	**Answer**: *The fairness of Fair K-means is measured through both clustering quality and fair representation of sensitive attribute groups. This is a good way to measure because it keeps the rigorous element of using the hard data to get results, while also measuring the success in the form of equal representation, which is what the algorithm is trying to achieve.*

------------------------------

6. **Question**: Are there any situations in which even a perfectly fair ML system might still cause or even exacerbate harm? Are there other metrics or areas of social impact you might consider? Justify your opinion.

	**Answer**: *There will almost always be situations in which a perfeclty fair ML system could cause problems or exacerbate harm. Many people have different opinions of what constitues a problem, or have different opinions about fairness. "Perfectly fair" is an ambiguous term because most people would disagree on what consitutes a universal fairness. There are also many factors the makeup the composition of a person besides the one mentioned in the article. For example, the article does not touch upon mental health, diagnosed or undiagnosed which could require strong consideration in an algorithm like Fair K-Means.*

------------------------------

7. **Question**:
	Based on the text, “Algorithms, if left unchecked, can create highly skewed and homogenous clusters that do not represent the demographics of the dataset. ”

	a. Identify a situation where an algorithm has caused significant social repercussions (you can talk about established companies or even any algorithms that you have encountered in your daily life).

	b. Based on your secondary knowledge of the situation what happened, why do you think it happened, and how would you do things differently (if you would). Justify your choices.

	**Answer**: *Many credit score / banking algorithms have been infamous for social repercussions. Because a demographics access to capital is so vital to its upward mobility, any unfairness in a credit algorithm has a multiplying affect on ongoing inequality. Since companies are all but driven solely by profit, in making a credit score algorithm, companies are not incentivized to consider designing it in such a way that does not contribute to ongoing inequality. One solution, how I would go about it, is to consider the long term effects of designing a fair algorithm and also to consider the societal impact ones company can make.*


------------------------------


8. **Question**:
	Read the article and answer the following questions:

	a. How did Google respond to this? 

	b. Experiment with the autocomplete feature of Google now. What do you notice now? How do you feel about the way Google handled this solution? List one area in which Google can still improve.

	**Answer**: *Google responded to this by removing autocomplete predictions that imply harmful assumptions about groups of people. What I notice in google's autocomplete now is very bipartisan suggestions, often lacking in adjectives. Most of the autocomplete suggestions have to do with geographical questions or data related questions, rather than adjective driven stereotypes. I think this is a good way to handle the situation because autocomplete is such a customer facing part of the search algorithm. Even if people are not searching harmful queries, they may see the harmful query suggestion in the process of writing their query. Google can still improve on this by designing the search results page in a way that highlights good amongst various demographics rather than negative content.*