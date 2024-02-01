# Zeroth-Order FW Variants for Adversarial Attacks
<p align="center">  
<b>Frank-Wolfe Variants for Adversarial Attacks. Final Project for Optimization for Data Science Course, UniPD</b>
</p>

<p align="center">  
<b>Author: Marco Uderzo</b>
</p>

</br>


<p align="center">
  
<img src="/Sample-Output/ZOSVRG-Sample-1/0004.png" width="70" height="70">
<img src="/Sample-Output/ZOSVRG-Sample-1/0006.png" width="70" height="70">
<img src="/Sample-Output/ZOSVRG-Sample-1/0019.png" width="70" height="70">
<img src="/Sample-Output/ZOSVRG-Sample-1/0024.png" width="70" height="70">
<img src="/Sample-Output/ZOSVRG-Sample-1/0027.png" width="70" height="70">
<img src="/Sample-Output/ZOSVRG-Sample-1/0033.png" width="70" height="70">
<img src="/Sample-Output/ZOSVRG-Sample-1/0042.png" width="70" height="70">
<img src="/Sample-Output/ZOSVRG-Sample-1/0048.png" width="70" height="70">
<img src="/Sample-Output/ZOSVRG-Sample-1/0049.png" width="70" height="70">
<img src="/Sample-Output/ZOSVRG-Sample-1/0056.png" width="70" height="70">

</p>

<p align="center">

<img src="/Sample-Output/ZOSVRG-Sample-1/Adv_id4_Orig4_Adv9.png" width="70" height="70">
<img src="/Sample-Output/ZOSVRG-Sample-1/Adv_id6_Orig4_Adv8.png" width="70" height="70">
<img src="/Sample-Output/ZOSVRG-Sample-1/Adv_id19_Orig4_Adv2.png" width="70" height="70">
<img src="/Sample-Output/ZOSVRG-Sample-1/Adv_id24_Orig4_Adv9.png" width="70" height="70">
<img src="/Sample-Output/ZOSVRG-Sample-1/Adv_id27_Orig4_Adv9.png" width="70" height="70">
<img src="/Sample-Output/ZOSVRG-Sample-1/Adv_id33_Orig4_Adv2.png" width="70" height="70">
<img src="/Sample-Output/ZOSVRG-Sample-1/Adv_id42_Orig4_Adv9.png" width="70" height="70">
<img src="/Sample-Output/ZOSVRG-Sample-1/Adv_id48_Orig4_Adv9.png" width="70" height="70">
<img src="/Sample-Output/ZOSVRG-Sample-1/Adv_id49_Orig4_Adv9.png" width="70" height="70">
<img src="/Sample-Output/ZOSVRG-Sample-1/Adv_id56_Orig4_Adv9.png" width="70" height="70">

</p>

</br>

## Project Description

 The goal of this project is to compare the behaviour and performance
 of two three Zeroth-Order variants of the Frank-Wolfe Algorithm, aimed at 
 solving constrained optimization problems with a better iteration complexity,
 expecially with respect to oracle queries.
 
 We take into consideration: Faster Zeroth-Order Conditional Gradient Sliding (FZCGS)
 (Gao et al., 2018) and Stochastic Gradient Free Frank Wolfe 
 (SGFFW) (Sahu et al., 2019). The latter algorithm branches off into three slightly different ones,
 depending on the Stochastic Approximation Technique used, namely: classical Kiefer-Wolfowitz
 Stochastic Approximation (KWSA) (Kiefer and Wolfowitz, 1952), Random Directions Stochastic Approximation
 (RDSA) (Nesterov and Spokoiny, 2011; Duchi et al., 2015), and an Improvised RDSA (IRDSA). 

 The theory behind these algorithms is presented, with an emphasis on proving that the performance are guaranteed. 
 Then, the aforementioned algorithms are tested on a black-box adversarial attack on the MNIST dataset. 

## Base Repositories

-  [IBM/ZOSVRG-BlackBox-Adv](https://github.com/IBM/ZOSVRG-BlackBox-Adv/tree/master) : base repository used in the Gao et al. paper, which is used as a framework to implement the optimization algorithms and test them on MNIST Adversarial Attacks.
-  [carlini/nn_robust_attacks](https://github.com/carlini/nn_robust_attacks) : repository used by the IBM repo as a base for the Adversarial Attacks framework.


