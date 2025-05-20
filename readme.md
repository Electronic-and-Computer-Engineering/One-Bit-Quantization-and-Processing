<div style="text-align: justify">
  
# Robust Quantization and Processing of One-Bit Signals

**Authors:**  
Florian Mayer – FH JOANNEUM University of Applied Sciences, Graz, Austria  
Christian Vogel – FH JOANNEUM University of Applied Sciences, Graz, Austria

---
## Introduction

One-bit signals are binary, meaning that each single sample is *on* or *off* at any given time (`+1` or `−1`).  
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
## References

[1] Z. Li, W. Xu, X. Zhang, and J. Lin, "A survey on one-bit compressed sensing: theory and applications," *Frontiers of Computer Science*, vol. 12, no. 2, pp. 217–230, 2018. [Online]. Available: https://doi.org/10.1007/s11704-017-6132-7

[2] P. T. Boufounos and R. G. Baraniuk, "1-Bit compressive sensing," in *Proc. 42nd Annual Conf. Information Sciences and Systems (CISS)*, Princeton, NJ, USA, Mar. 2008, pp. 16–21. [Online]. Available: https://doi.org/10.1109/CISS.2008.4558487

[3] S. Khobahi, N. Naimipour, M. Soltanalian, and Y. C. Eldar, "Deep Signal Recovery with One-Bit Quantization," in *Proc. ICASSP*, May 2019, pp. 2987–2991.

[4] S. Khobahi and M. Soltanalian, "Model-Based Deep Learning for One-Bit Compressive Sensing," *IEEE Transactions on Signal Processing*, vol. 68, pp. 5292–5307, 2020.

[5] D. Reefman and E. Janssen, "One-bit Audio: An Overview," *Journal of the Audio Engineering Society*, vol. 52, Mar. 2004.

[6] J. D. Reiss, "Understanding Sigma-Delta Modulation: The Solved and Unsolved Issues," *AES: Journal of the Audio Engineering Society*, vol. 56, no. 1, 2008.

[7] S. M. Kershaw and M. B. Sandler, "Sigma-delta modulation for audio DSP," in *IEE Colloquium on Audio DSP – Circuits and Systems*, 1993, pp. 1/1–1/6.

[8] K. Hausmair, S. Chi, P. Singerl, and C. Vogel, "Aliasing-Free Digital Pulse-Width Modulation for Burst-Mode RF Transmitters," *IEEE Transactions on Circuits and Systems I*, vol. 60, no. 2, pp. 415–427, Feb. 2013. [Online]. Available: https://doi.org/10.1109/TCSI.2012.2215776

[9] K. Hausmair, P. Singerl, and C. Vogel, "Multiplierless Implementation of an Aliasing-Free Digital Pulsewidth Modulator," *IEEE Transactions on Circuits and Systems II*, vol. 60, no. 9, pp. 592–596, Sep. 2013. [Online]. Available: https://doi.org/10.1109/TCSII.2013.2268431

[10] K. Hausmair, S. Chi, and C. Vogel, "How to reach 100% coding efficiency in multilevel burst-mode RF transmitters," in *Proc. ISCAS*, Beijing, 2013, pp. 2255–2258. [Online]. Available: https://doi.org/10.1109/ISCAS.2013.6572326

[11] S. Chi, P. Singerl, and C. Vogel, "Coding efficiency optimization for multilevel PWM based switched-mode RF transmitters," in *Proc. MWSCAS*, Seoul, 2011, pp. 1–4. [Online]. Available: https://doi.org/10.1109/MWSCAS.2011.6026539

[12] S. Chi, K. Hausmair, and C. Vogel, "Coding efficiency of bandlimited PWM based burst-mode RF transmitters," in *Proc. ISCAS*, Beijing, 2013, pp. 2263–2266. [Online]. Available: https://doi.org/10.1109/ISCAS.2013.6572328

[13] S. Chi, C. Vogel, and P. Singerl, "The frequency spectrum of polar modulated PWM signals and the image problem," in *Proc. ICECS*, Athens, Greece, 2010, pp. 679–682. [Online]. Available: https://doi.org/10.1109/ICECS.2010.5724603

[14] T. Hoefler, D. Alistarh, T. Ben-Nun, N. Dryden, and A. Peste, "Sparsity in Deep Learning: Pruning and growth for efficient inference and training in neural networks," arXiv preprint arXiv:2102.00554, 2021. [Online]. Available: http://arxiv.org/abs/2102.00554

[15] J. Singh, O. Dabeer, and U. Madhow, "On the limits of communication with low-precision analog-to-digital conversion at the receiver," *IEEE Transactions on Communications*, vol. 57, no. 12, pp. 3629–3639, Dec. 2009.

[16] J. Prainsack and K. Witrisal, "Optimum receiver based on single bit quantization," in *Proc. SPAWC*, Marrakech, Morocco, Jun. 2010, pp. 1–5. [Online]. Available: https://doi.org/10.1109/SPAWC.2010.5671062

[17] H. A. Hjortland, *Sampled and Continuous-Time 1-Bit Signal Processing in CMOS for Wireless Sensor Networks* (PhD Thesis), [n.d.].

[18] G. G. E. Gielen, L. Hernandez, and P. Rombouts, "Time-Encoding Analog-to-Digital Converters – Part 1: Basic Principles," *IEEE Solid-State Circuits Magazine*, vol. 12, no. 2, pp. 47–55, 2020. [Online]. Available: https://doi.org/10.1109/MSSC.2020.2987536

[19] G. G. E. Gielen, L. Hernandez, and P. Rombouts, "Time-Encoding Analog-to-Digital Converters – Part 2: Architectures and Circuits," *IEEE Solid-State Circuits Magazine*, vol. 12, no. 3, pp. 18–27, 2020. [Online]. Available: https://doi.org/10.1109/MSSC.2020.3002144
</div>
