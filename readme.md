This repository collects all informations, codes, and presentations related to the workshop I presented at the 5th "[CSMA (Computational Structural Mechanics Association) Junior held in Porquerolles Island, France, on 14-15 May 2022](https://csma.asso.univ-lorraine.fr/csma-juniors/)".   

# Workshop: Deep learning, real-time simulation and model-order reduction


The rise of innovative structures in engineering is today hampered by our ability to predict the response of complex systems, and more precisely of the materials that compose them. The behavior of the latter is indeed very difficult to apprehend with empirical models, due to the lack of underlying complexity and lack of knowledge of physical phenomena.

To this end, data-driven approaches have emerged in recent years with the aim of providing enriched descriptions of material behavior.
A paradigm shift in traditional modeling thus occurred: data, rather than being used solely as verification tools, began to become the stimulus in the quest for new constitutive representations.
However, data-driven approaches have major weaknesses that hinder their application. The absence of a rigorous framework based on physics (“black box” model effect) is one of the main problems threatening the generalization of these methods to solve problems in engineering.

In this mini-course, we will look at physics-driven data-driven approaches to modeling the behavior of complex materials and the response of heterogeneous structures. In particular, neural networks based on thermodynamics, also called TANN, will be analyzed in detail. These networks, thanks to the integration of the first and second principles of thermodynamics, guarantee results that are always thermodynamically admissible and open up new possibilities in numerical and experimental mechanics, in particular through the use of numerical twins and reduction approaches. models.

Practical session. We will take a closer look at the theoretical and numerical implementation of neural networks for modeling the behavior of materials. Through motivating examples, we will study the architecture of classical networks and TANN networks. The attention will be put on the comprehension of the numerical diagrams and the training of the neural networks, but also the guided programming of certain tasks.
At the end of this practical session, we will analyze the benefits of physics-driven data-driven approaches compared to standard approaches.


  - Masi, Stefanou, Vannucci, and Maffi-Berthier (2020). "[Thermodynamics-based Artificial Neural Networks for constitutive modeling](https://doi.org/10.1016/j.jmps.2020.104277)". Journal of the Mechanics and Physics of Solids, 104277.
  
  - Masi, Stefanou, Vannucci, and Maffi-Berthier (2021). "[Material modeling via Thermodynamics-based Artificial Neural Networks](https://franknielsen.github.io/SPIG-LesHouches2020/Masi-SPIGL2020.pdf)". In: Barbaresco F., Nielsen F. (eds) Proceedings of SPIGL'20: Geometric Structures of Statistical Physics, Information Geometry, and Learning. Springer.



## Citation


    @article{masi2020thermodynamics,
     title={Thermodynamics-based Artificial Neural Networks for constitutive modeling},
     author={Masi, Filippo and Stefanou, Ioannis and Vannucci, Paolo and Maffi-Berthier, Victor},
     journal={Journal of the Mechanics and Physics of Solids},
     pages={104277},
     year={2020},
     publisher={Elsevier}
     }
     
     
    @InProceedings{material2021modeling,
     author={Masi, Filippo and Stefanou, Ioannis and Vannucci, Paolo and Maffi-Berthier, Victor},
     editor={Barbaresco, Frédéric and Nielsen, Frank},
     title={Material modeling via Thermodynamics-based Artificial Neural Networks},
     booktitle={Proceedings of SPIGL'20: Geometric Structures of Statistical Physics, Information Geometry, and Learning},
     year={2021},
     publisher={Springer}
     }

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4482669.svg)](https://doi.org/10.5281/zenodo.4482669)
