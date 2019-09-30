# Status
On going:
- implementare LeNet (Vanilla)
- implementare MC Dropout per LeNet
- processo raccolta statistiche per Max Softmax Prob.


## Teoria
Sono stati individuati 3 metodi base per la stima dell'incertezza nei modelli:
- **Maximum Softmax Probability** \
(*Hendrycks, D. and Gimpel, K. A Baseline for Detecting Misclassified and Out-of-Distribution
Examples in Neural Networks.*)
- **Concrete Dropout** \
(*Gal, Y. and Ghahramani, Z. Dropout as a Bayesian approximation: Representing model uncertainty in deep learning.*)
- **Stochastic Variational Inference** \
(*Blundell, C., Cornebise, J., Kavukcuoglu, K., and Wierstra, D. Weight uncertainty in neural networks.*)


## Esperimenti
La fase sperimentale punta a quantificare la capacità dei vari metodi sopracitati di valutare il grado di incertezza legato ad una predizione da parte di un modello di apprendimento automatico.

Il processo di valutazione è suddiviso nelle seguenti fasi:
1. selezione del dataset di riferimento
2. selezione modello *baseline*
3. per ogni tecnica di valutazione dell'incertezza generare il relativo modello *modified*
4. addestrare i modelli modificati
5. testare le performance *classiche* dei modelli
6. testare i modelli con le metriche di incertezza

Per maggiori dettagli fare riferimento al paper \
[Uncertaintly Essay - EvaluatingPredictive Uncertainty Under Dataset Shift](papers/Uncertaintly%20Essay%20-%20EvaluatingPredictive%20Uncertainty%20Under%20Dataset%20Shift.pdf).

### Datasets
I dati utilizzati durante la fase di testing sono divisi in tre categorie distinte:
- **in-distribution data**
- **shifted data**
- **out-of-distribution data**

e ad ogni categoria verrà associato un particolare dataset.

Come dataset *in-distribution* verranno utilizzati:
- **CIFAR10**
- **MNIST**

Come dataset *out-of-distribution* verranno invece utilizzati:
- **SVHN** \
(*Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., and Ng, A. Y. Reading Digits in Natural
Images with Unsupervised Feature Learning.*)
- **Not-MNIST** \
(*Bulatov, Y. NotMNIST dataset, 2011. http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html.*)


### Modelli
Per ogni tipologia di dataset verranno presentati modelli *ad hoc*:
- *MNIST* : **LeNet** \
(*LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradient-based learning applied to document
recognition.*)
- *CIFAR10* : **ResNet20**





### Metriche di valutazione
Oltre alle solite metriche indipendenti dal concetto di incertezza saranno utilizzate:
- Negative Log-Likelihood (NLL).
- Brier Score.
- Expected Calibration Error (ECE).


