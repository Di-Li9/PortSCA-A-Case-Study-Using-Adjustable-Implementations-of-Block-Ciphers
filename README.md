# Portability of Profiling Side-channel Analysis: A Case Study Using Adjustable Implementations of Block Ciphers

This repository contains the implementation and extensions of the paper *“Portability of Profiling Side-channel Analysis: A Case Study Using Adjustable Implementations of Block Ciphers”*.
![pic2](https://github.com/Di-Li9/PortSCA-A-Case-Study-Using-Adjustable-Implementations-of-Block-Ciphers/blob/master/overall_architecture.png)

## Getting started

To avoid any conflicts, we recommend using a virtual environment.  You can get started by running the following command:

```bash
git clone https://github.com/Di-Li9/PortSCA-A-Case-Study-Using-Adjustable-Implementations-of-Block-Ciphers
```

Create a new Conda environment :

```bash
conda create -n dl_sca python=3.8
```

Activate the environment:

```bash
conda activate dl_sca
```

Install the required packages. We provide a `requirements.txt` file that lists all necessary dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

Here is a non-exhaustive list of the required packages with their default versions provided in the `requirements.txt` file:

- `tensorflow-gpu` (2.2.0)
- `keras` (2.3.1)
- `numpy` (1.22.0)

Additionally, our platform is equipped with CUDA 12.0 and cuDNN 8.1.

## Method

Since the **SKINNY** cipher's tweak parameter is known, the attack procedure is consistent with that of a standard profiling SCA. The methods for recovering secret components and key-dependent S-boxes can be divided into two categories:

### 1) Equation-Based Method

This method requires high accuracy in recovering the mapping $\phi: p \mapsto v$ and is suitable for scenarios without protection countermeasures, where model performance is not affected by adjustable implementation parameters.

| Target                                       | Expression                                                   |
| -------------------------------------------- | ------------------------------------------------------------ |
| Customized S-box                             |$\begin{array}{l}\Delta_{0,1} = k_1 \oplus k_0 = p_{r,0} \oplus p_{s,1} \\\Delta_{0,2} = k_2 \oplus k_0 = p_{r,0} \oplus p_{t,1} \\\vdots \\\Delta_{0,15} = k_{15} \oplus k_0 = p_{r,0} \oplus p_{u,15}\end{array}$|
| MDS Matrix (with 1–15 plaintext bytes fixed) | $v = a \cdot v_s \oplus b$                                 |
| Key-dependent S-box                          | $v_{rk^0,rk^1} = S^{rk^1}(p_i \oplus rk^0)$                  |

### 2) Guessing Entropy-Based Method

This method requires lower accuracy in recovering the mapping $\phi: p \mapsto v$ and is suitable for scenarios with protection countermeasures, or when model performance is affected by adjustable implementation components (i.e., accuracy decreases but remains above random probability).

| Target                                       | Expression                                                   |
| -------------------------------------------- | ------------------------------------------------------------ |
| Customized S-box                             | $d(\Delta)=\sum_{i=1}^{N}\log\big(f(t_i)[\Delta\oplus v_s]\big),\quad \Delta=v_r\oplus v_s$ |
| MDS Matrix (with 1–15 plaintext bytes fixed) | $d[a,k,b]=\sum_{i=1}^{n_a}\log\!\left(f(t_i)\left[a\cdot S(p_i\oplus k)\oplus b\right]\right)$ |
| Key-dependent S-box                          | $d(rk^0,rk^1)=\sum_{i=1}^{N}\log\big(f(t_i)[S^{rk^1}(p_i\oplus rk^0)]\big)$ |

## Examples

To facilitate the reproduction of our experiments, **executable examples** are provided for each dataset, with configurations strictly matching the hyperparameters reported in the paper.

Each example is organized in a dedicated directory corresponding to the target cipher (`Customized_AES`, `Pilsung`, `SKINNY`).

For instance, examples for the Pilsung dataset are located in the `Pilsung/` directory, while those for the **SKINNY** dataset are in the `SKINNY/` directory.

The `Customized_AES/` directory contains analyses for the Customized S-box and Customized MDS, with deeper subdirectories distinguishing experiments conducted on **different devices** and **same devices**.

The `util/` directory for each dataset includes the core implementation files used in the experiments, including:

- `CLR.py`: implements OneCycleLR, *code adapted from https://github.com/fastai/fastai*  
- `LoadData.py`: provides dataset loading functionality

These components collectively ensure the full reproducibility of the experimental results and support extending the methods to other datasets.

The `Model/` directory contains the models used in the experiments along with their trained parameters.

## Additional Notes

In **Figure 12(b)**, the legend labels for $Sbox(P\oplus K)$, $Sbox(P\oplus K)\oplus Mask[0]$, and $Mask[0]$ are incorrectly assigned. Specifically, the labels for $Sbox(P\oplus K)$ and $Mask[0]$ should be swapped to maintain consistency with **Figure 12(a)**.
![pic2](https://github.com/Di-Li9/PortSCA-A-Case-Study-Using-Adjustable-Implementations-of-Block-Ciphers/blob/master/SNR_PoI.png)
