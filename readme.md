<div style="text-align: justify">
  
# Robust Quantization and Processing of One-Bit Signals

**Authors:**  
Florian Mayer – FH JOANNEUM University of Applied Sciences, Graz, Austria  
Christian Vogel – FH JOANNEUM University of Applied Sciences, Graz, Austria

---
## Introduction

One-bit signals are binary, meaning that each single sample is *on* or *off* at any given time.  
This characteristic makes it easier to encode, decode, and process signals with higher efficiency, as the two-level signal significantly reduces storage as well as complexity costs [1]. As such, they have practical applications in many areas, including digital signal processing, compressive sensing, analog-to-digital conversion (ADC), and communication systems.

In compressive sensing, one-bit quantization has significantly enhanced sparse signal recovery, efficiently managing dense signals and extensive data [2]. Further, using deep learning frameworks and model-based deep learning architectures enhances the ability to recover signals from quantized data by learning optimal reconstruction algorithms [3,4]. Especially in audio processing, one-bit signals allow for higher sampling rates, improving sound quality and efficiency by reducing the need for complex filtering and signal processing [5,6,7].

Using one-bit signals in burst-mode RF transmitters allows power amplifiers to reach peak efficiency and avoid power waste during low-level periods, resulting in higher average efficiency compared to conventional linear power amplifiers [8–13].
Applying one-bit quantization further enhances machine learning by enabling smaller and more efficient network structures suitable for edge computing and IoT devices with constrained computational resources [14]. Research on the impact of low-precision ADCs on communication performance highlights the trade-off between precision and efficiency [15,16]. One-bit signal processing is further empowered by advances in CMOS technology, especially for wireless sensor networks [17], and the development of time-encoding ADCs [18,19].

---

One-bit signals represent an extreme form of quantization, where each sample is reduced to a binary value of either `+1` or `−1`. This binary nature offers attractive advantages in terms of hardware efficiency, signal transmission, and low power consumption. One-bit quantization has proven useful in areas such as digital signal processing, analog-to-digital conversion (ADC), compressive sensing, burst-mode RF systems, and energy-aware machine learning.

However, these benefits come at the cost of severe information loss. In contrast to traditional multi-bit representations, one-bit signals require fundamentally new algorithms for signal recovery and processing, especially under realistic noise, dynamic range, and bandwidth constraints.

This research investigates whether there exists an **optimal one-bit representation** of a given real-valued signal that minimizes reconstruction error under filtering constraints. Instead of heuristic or fixed quantizers, the quantization is formulated as a constrained optimization problem:

$$
\min_{\mathbf{b} \in \{-1, +1\}^N} \left\| \mathbf{x} - \mathbf{R} \cdot \mathbf{b} \right\|_2^2
$$

Here, \( \mathbf{x} \in \mathbb{R}^N \) is the original signal, \( \mathbf{b} \) the one-bit vector, and \( \mathbf{R} \in \mathbb{R}^{N \times N} \) a fixed reconstruction filter (e.g., FIR). The task is to find a binary sequence \( \mathbf{b} \) that best preserves the signal structure after reconstruction.

---

## Research Objectives

This project addresses the following core questions:

- **Optimal Representation:** How can the one-bit quantizer \( F(\cdot) \) be improved to yield minimal reconstruction error for a given real-valued input signal?

- **Spectral Shaping:** Is it possible to design one-bit quantizers that enforce a specific spectral energy distribution \( \tilde{E}(\omega) \), without relying on oversampling?

- **Sequential and Block-Based Methods:** Can the quantization process be split into sequential or block-wise segments to reduce the computational complexity, while preserving fidelity?

- **Robustness:** How can one-bit quantization be made resilient to noise and uncertainty, especially in real-world and embedded applications?

- **Arithmetic Operations:** Can arithmetic (e.g., \( F(x_1) + F(x_2) \approx F(x_1 + x_2) \)) be performed directly on quantized signals, and under what conditions does this hold?

---

## Methodology

We propose a structured, optimization-based framework that generalizes the quantization process across application domains. The quantization is performed block-wise to reduce complexity, using overlapping filters and minimum-phase systems for optimal spectral behavior. The block-structure allows the reuse of previous results and enables streaming applications.

The core components include:

- **Filter Design:** Minimum-phase FIR filters for targeted noise shaping.
- **Block Optimization:** Independent optimization per block with propagated error history.
- **Evaluation Metrics:** Signal-to-error ratio (SER), spectral flatness, and robustness to variation.

This block-wise method is referred to as **OBBQ – Optimization-Based Block Quantization**, and generalizes previous work on one-bit quantization into a scalable and adaptive form.

---

## Outlook

By transforming one-bit quantization from a static encoding into a dynamic optimization problem, this framework offers flexibility for a wide range of signal types and use-cases. Potential applications include:

- Real-time digital signal processing on embedded systems
- Robust low-resolution ADCs for RF frontends
- Frequency-selective signal shaping without oversampling
- Adaptive one-bit learning in energy-constrained neural systems

Future work will explore higher-dimensional signals, real-time implementations, and combinations with deep-learning-based reconstruction methods.

---

## Further Project Modules

- [ISCAS_2024](./ISCAS_2024/README.md)
- [Proposal_Documentation](./Proposal_Documentation/README.md)
- [Sequential_Prototype](./Sequential_Prototype/README.md)

---

## Acknowledgment

This research is funded by the Austrian Science Fund (FWF) [10.55776/DFH 5] within the DENISE project and supported by the province of Styria.

---

## © 2025, Florian Mayer and Christian Vogel

---

## Selected References

[1] Z. Li et al., “A survey on one-bit compressed sensing: Theory and applications,” *Front. Comput. Sci.*, vol. 12, no. 2, 2018.  
[2] P. T. Boufounos and R. G. Baraniuk, “1-Bit Compressive Sensing,” *CISS*, IEEE, 2008.  
[3] F. Mayer and C. Vogel, “An Optimization-Based Approach to One-Bit Quantization,” *ISCAS 2024*.  
[4] S. Khobahi et al., “Deep Signal Recovery with One-Bit Quantization,” *ICASSP*, IEEE, 2019.  
[5] R. Schreier et al., *Understanding Delta-Sigma Data Converters*, Wiley, 2017.

</div>
