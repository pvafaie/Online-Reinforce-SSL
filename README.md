This is the github repository for the paper titled Online semi-supervised learning from evolving data streams with meta-features and deep reinforcement learning. Please note that this page is still in progress and might change in the future. The paper will be published in the precedings of LOD conference 2021. This work can also be found in chapter 4 of my thesis. 

The flowchart of the Online Reinforce algorithm can be seen below:

https://ruor.uottawa.ca/handle/10393/42636

![image](https://user-images.githubusercontent.com/56241887/134709083-4a38ef7a-3374-43f3-8da7-ac29d9811fa7.png)

In this method, we need to first train the meta-RL model. The first step for training the meta-RL is to create the meta-dataset. The datasets can be created with the files in the creating training data. Afterwards, the meta-RL model can be trained based on the created datasets.

After the meta-RL model is trained, we can directly use it for pseudo-labelling in our target data. 
