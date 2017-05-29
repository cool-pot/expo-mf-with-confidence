# ExpoMF with confidence

A modified version of expo-mf, which has a better performance than expo-mf when facing implicit data indicating "frequence or intensity".

This repository contains the experienmental study notebooks in my greduate thesis.

This repository is largely influenced by the paper ["Modeling User Exposure in Recommendation"](http://arxiv.org/abs/1510.07025) (WWW'16) and dawenl's repository["expomf"](https://github.com/dawenl/expo-mf).

## Binarized Implicit Feedback

![performance_steam](https://github.com/cool-pot/expo-mf-with-confidence/blob/master/pics/performance_steam.png)

## Non-binarized Implicit Feedback indicating "frequence or intensity"

![performance_tps](https://github.com/cool-pot/expo-mf-with-confidence/blob/master/pics/performance_tps.png)

## Datasets
- [Taste Profile Subset](http://labrosa.ee.columbia.edu/millionsong/tasteprofile)
- [Movielens](https://grouplens.org/datasets/movielens/)
- [Steam Vedio Games](https://www.kaggle.com/tamber/steam-video-games)

## Contrast
I used the weighted matrix factorization (WMF) implementation in [Implicit](https://github.com/benfred/implicit). 
