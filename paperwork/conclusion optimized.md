```latex
\section{Conclusion}
\noindent
This paper investigates the dynamic deployment problem of DAG-structured DNN inference tasks in cloud-edge-end collaborative environments. We formulate an online deployment model that jointly characterizes DNN dependency structures, time-varying resource states, reliability degradation, and energy consumption. Based on this model, the deployment problem is formulated as a multi-objective optimization problem that jointly considers service reliability, inference accuracy, inference delay, and total energy consumption.

To solve this problem, we propose DAREED, a DAG-Aware Reliable and Energy-Efficient Evolutionary Deployment algorithm. DAREED combines evolutionary multi-objective search with DAG-aware operators, resource-aware search guidance, constraint handling, and preference-based selection. These mechanisms enable the algorithm to exploit DNN topology and dynamic resource states while maintaining population diversity and reducing infeasible exploration.

The experimental design evaluates DAREED across Pareto quality, online deployment performance, robustness, scalability, and preference adaptability. The complete experimental results are expected to demonstrate that DAREED can generate high-quality deployment candidates and achieve a favorable balance among reliability, accuracy, delay, and energy under online DNN inference workloads. In future work, we will further investigate learning-assisted preference adaptation and extend the deployment framework to support additional objectives such as monetary cost, carbon-aware resource usage, and heterogeneous accelerator scheduling.


\begin{thebibliography}{1}
\bibliographystyle{IEEEtran}

\bibitem{ref1}
L. Jiao, D. Wang, Y. Bai, et al., ``Deep learning in visual tracking: A review,'' \textit{IEEE Transactions on Neural Networks and Learning Systems}, vol. 34, no. 9, pp. 5497--5516, 2021.

\bibitem{ref2}
W. Samek, G. Montavon, S. Lapuschkin, et al., ``Explaining deep neural networks and beyond: A review of methods and applications,'' \textit{Proceedings of the IEEE}, vol. 109, no. 3, 2021, doi: 10.1109/JPROC.2021.3060483.

\bibitem{ref3}
Y. Tang, C. Zhao, J. Wang, et al., ``Perception and navigation in autonomous systems in the era of learning: A survey,'' \textit{IEEE Transactions on Neural Networks and Learning Systems}, 2022.

\bibitem{ref4}
N. Pathak, S. Misra, A. Mukherjee, et al., ``UAV virtualization for enabling heterogeneous and persistent UAV-as-a-service,'' \textit{IEEE Transactions on Vehicular Technology}, vol. 69, no. 6, pp. 6731--6738, 2020.

\bibitem{ref5}
V. Sze, Y. H. Chen, T. J. Yang, et al., ``Efficient processing of deep neural networks: A tutorial and survey,'' \textit{Proceedings of the IEEE}, vol. 105, no. 12, pp. 2296--2319, 2017, doi: 10.1109/JPROC.2017.2761740.

\bibitem{ref6}
S. S. Gill, M. Golec, J. Hu, et al., ``Edge AI: A taxonomy, systematic review and future directions,'' \textit{Cluster Computing}, vol. 28, no. 1, p. 18, 2025.

\bibitem{ref7}
M. Patel, B. Naughton, C. Chan, N. Sprecher, S. Abeta, and A. Neal, ``Mobile-edge computing introductory technical white paper,'' \textit{White Paper, Mobile-edge Computing (MEC) Industry Initiative}, 2014.

\bibitem{ref8}
X. Wang, Y. Han, V. C. M. Leung, et al., ``Convergence of edge computing and deep learning: A comprehensive survey,'' \textit{IEEE Communications Surveys \& Tutorials}, vol. 22, no. 2, pp. 869--904, 2020.

\bibitem{ref9}
H. Ma, R. Li, X. Zhang, et al., ``Reliability-aware online scheduling for DNN inference tasks in mobile edge computing,'' \textit{IEEE Internet of Things Journal}, 2023.

\bibitem{ref10}
S. Teerapittayanon, B. McDanel, and H. T. Kung, ``Distributed deep neural networks over the cloud, the edge and end devices,'' 2017, doi: 10.48550/arXiv.1709.01921.

\bibitem{ref11}
X. Deng, J. Yin, P. Guan, et al., ``Intelligent delay-aware partial computing task offloading for multiuser industrial Internet of Things through edge computing,'' \textit{IEEE Internet of Things Journal}, 2023.

\bibitem{ref12}
M. Min, D. Xu, L. Xiao, et al., ``Learning-based computation offloading for IoT devices with energy harvesting,'' 2017, doi: 10.48550/arXiv.1712.08768.

\bibitem{ref13}
Y. Huang, F. Wang, F. Wang, et al., ``DeePar: A hybrid device-edge-cloud execution framework for mobile deep learning applications,'' in \textit{IEEE INFOCOM 2019 - IEEE Conference on Computer Communications Workshops (INFOCOM WKSHPS)}, 2019, doi: 10.1109/INFCOMW.2019.8845240.

\bibitem{ref14}
T. Mohammed, C. Joe-Wong, R. Babbar, and M. D. Francesco, ``Distributed inference acceleration with adaptive DNN partitioning and offloading,'' in \textit{Proc. IEEE INFOCOM}, 2020, pp. 854--863.

\bibitem{ref15}
J. Li, W. Liang, Y. Li, et al., ``Throughput maximization of delay-aware DNN inference in edge computing by exploring DNN model partitioning and inference parallelism,'' \textit{IEEE Transactions on Mobile Computing}, vol. 22, no. 5, pp. 3017--3030, 2021.

\bibitem{ref16}
X. Xu, K. Yan, S. Han, B. Wang, X. Tao, and P. Zhang, ``Learning-based edge-device collaborative DNN inference in IoVT networks,'' \textit{IEEE Internet of Things Journal}, vol. 11, no. 5, pp. 7989--8004, Mar. 2024, doi: 10.1109/JIOT.2023.3317785.

\bibitem{ref17}
J. Xu, H. Yao, R. Zhang, T. Mai, and M. Guizani, ``Low latency and accuracy-guaranteed DNN inference for UAV-assisted IoT networks,'' \textit{IEEE Transactions on Cognitive Communications and Networking}, doi: 10.1109/TCCN.2025.3542443.

\bibitem{ref18}
Z. Zhuang et al., ``DECC: Delay-aware edge-cloud collaboration for accelerating DNN inference,'' \textit{IEEE Transactions on Emerging Topics in Computing}, vol. 13, no. 2, pp. 438--450, Apr.-Jun. 2025, doi: 10.1109/TETC.2024.3404551.

\bibitem{ref19}
B. Lin, Y. Huang, J. Zhang, et al., ``Cost-driven off-loading for DNN-based applications over cloud, edge, and end devices,'' \textit{IEEE Transactions on Industrial Informatics}, vol. 16, no. 8, pp. 5456--5466, 2020, doi: 10.1109/TII.2019.2961237.

\bibitem{ref20}
Z. Xu et al., ``Energy-aware inference offloading for DNN-driven applications in mobile edge clouds,'' \textit{IEEE Transactions on Parallel and Distributed Systems}, vol. 32, no. 4, pp. 799--814, Apr. 2021.

\bibitem{ref21}
W. Jiang, H. Han, Q. Wang, L. Qian, F. Wei, and G. Feng, ``Energy-efficient resource allocation for accuracy-aware cooperative DNN inference in IoT,'' in \textit{IEEE INFOCOM 2025 - IEEE Conference on Computer Communications Workshops (INFOCOM WKSHPS)}, London, United Kingdom, 2025, pp. 1--6, doi: 10.1109/INFOCOMWKSHPS65812.2025.11152853.

\bibitem{ref22}
M. Mounesan, X. Zhang, and S. Debroy, ``Infer-EDGE: Dynamic DNN inference optimization in just-in-time Edge-AI implementations,'' in \textit{NOMS 2025 - 2025 IEEE Network Operations and Management Symposium}, Honolulu, HI, USA, 2025, pp. 1--9, doi: 10.1109/NOMS57970.2025.11073623.

\bibitem{ref23}
Z. Zhang, Y. Zhao, M.-C. Chang, C. Lin, and J. Liu, ``E4: Energy-efficient DNN inference for edge video analytics via early-exit and DVFS,'' 2025, arXiv:2503.04865.

\bibitem{ref24}
H. Liu, L. Cao, T. Pei, Q. Deng, and J. Zhu, ``A fast algorithm for energy-saving offloading with reliability and latency requirements in multi-access edge computing,'' \textit{IEEE Access}, vol. 8, pp. 151--161, 2020.

\bibitem{ref25}
J. Zilic, A. Aral, and I. Brandi, ``EFPO: Energy efficient and failure predictive edge offloading,'' in \textit{Proc. 12th IEEE/ACM International Conference on Ubiquitous Computing and Communications (UCC)}, 2019, pp. 165--175.

\bibitem{ref26}
K. Liu, C. Liu, G. Yan, V. C. S. Lee, and J. Cao, ``Accelerating DNN inference with reliability guarantee in vehicular edge computing,'' \textit{IEEE/ACM Transactions on Networking}, vol. 31, no. 6, pp. 3238--3253, Dec. 2023, doi: 10.1109/TNET.2023.3279512.

\bibitem{ref27}
M. Taheri et al., ``Exploration of activation fault reliability in quantized systolic array-based DNN accelerators,'' in \textit{2024 25th International Symposium on Quality Electronic Design (ISQED)}, San Francisco, CA, USA, 2024, pp. 1--8, doi: 10.1109/ISQED60706.2024.10528372.

\bibitem{ref28}
G. Yan, C. Liu, and K. Liu, ``ASPM: Reliability-oriented DNN inference partition and offloading in vehicular edge computing,'' in \textit{2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC)}, Bilbao, Spain, 2023, pp. 3298--3303, doi: 10.1109/ITSC57777.2023.10422172.

\bibitem{ref29}
M. Deutel, G. Kontes, et al., ``Multi-objective Bayesian optimization with reinforcement learning for edge deployment of DNNs on microcontrollers,'' in \textit{Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO '25 Companion)}, ACM, New York, NY, USA, pp. 19--20.

\bibitem{ref30}
Y. Du, S. Xu, and Y. Yu, ``Joint scheduling of sensing data offloading and edge inference for multi-UAV networks,'' 2026, arXiv:2605.03898.

\bibitem{ref31}
J. Cao, X. Li, Q. Liu, T. Han, N. Zhang, and W. Shi, ``EdgeServing: Deadline-aware multi-DNN serving at the edge,'' 2026, arXiv:2605.05527.

\bibitem{ref32}
J. Zhou, X. Hou, Y. Zeng, et al., ``Quality of experience and reliability-aware task offloading and scheduling for multi-user mobile-edge computing systems,'' \textit{IEEE Transactions on Services Computing}, 2025.

\bibitem{ref33}
A. M. Rasouli, M. Esnaashari, and M. Ansari, ``RASOUL: A reliability-aware task allocation strategy to improve success rate and energy saving in mobile-edge computing,'' \textit{IEEE Internet of Things Journal}, 2025.

\bibitem{ref34}
H. Hao, C. Xu, W. Zhang, et al., ``Reliability-aware optimization of task offloading for UAV-assisted edge computing,'' \textit{IEEE Transactions on Computers}, 2025.

\bibitem{ref35}
Z. Zhou, J. Han, L. Mo, et al., ``Reliability-aware real-time task mapping for multi-server edge computing systems with DVFS,'' in \textit{2025 40th Youth Academic Annual Conference of Chinese Association of Automation (YAC)}, IEEE, 2025, pp. 1994--2000.

\bibitem{ref36}
J. Zhou, K. Cao, J. Sun, et al., ``A framework to solve the energy, makespan and lifetime problems in reliability-driven task scheduling,'' in \textit{2019 International Conference on Internet of Things (iThings) and IEEE Green Computing and Communications (GreenCom) and IEEE Cyber, Physical and Social Computing (CPSCom) and IEEE Smart Data (SmartData)}, IEEE, 2019, pp. 608--614.

\bibitem{ref37}
K. Ergun, R. Ayoub, P. Mercati, et al., ``Dynamic reliability management of multigateway IoT edge computing systems,'' \textit{IEEE Internet of Things Journal}, vol. 10, no. 5, pp. 3864--3889, 2022.

\bibitem{ref38}
O. Shafi, M. K. Pandit, A. Gujarati, et al., ``IceEdge: Thermal-aware machine learning inference serving for emerging edge applications,'' \textit{ACM Transactions on Sensor Networks}, vol. 22, no. 4, pp. 1--26, 2026.

\bibitem{ref39}
K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, ``A fast and elitist multiobjective genetic algorithm: NSGA-II,'' \textit{IEEE Transactions on Evolutionary Computation}, vol. 6, no. 2, pp. 182--197, Apr. 2002, doi: 10.1109/4235.996017.

\bibitem{ref40}
C. A. C. Coello and M. S. Lechuga, ``MOPSO: A proposal for multiple objective particle swarm optimization,'' in \textit{Proc. 2002 Congress on Evolutionary Computation (CEC'02)}, 2002, pp. 1051--1056.

\bibitem{ref41}
J. Xu, Y. Tian, P. Ma, et al., ``Prediction-guided multi-objective reinforcement learning for continuous robot control,'' in \textit{International Conference on Machine Learning}, PMLR, 2020, pp. 10607--10616.

\bibitem{ref42}
F. Song et al., ``Evolutionary multi-objective reinforcement learning based trajectory control and task offloading in UAV-assisted mobile edge computing,'' \textit{IEEE Transactions on Mobile Computing}, vol. 22, no. 12, pp. 7387--7405, Dec. 2023, doi: 10.1109/TMC.2022.3208457.

\bibitem{ref43}
W. Ma, L. Zheng, and H. Zhou, ``A novel edge computing optimization method based on multi-objective optimization theory,'' in \textit{2022 Global Conference on Robotics, Artificial Intelligence and Information Technology (GCRAIT)}, Chicago, IL, USA, 2022, pp. 128--130, doi: 10.1109/GCRAIT55928.2022.00035.

\bibitem{ref44}
H. Huang, J. Liang, and G. Min, ``Joint DNN model deployment, selection, and configuration for heterogeneous inference services toward edge intelligence,'' \textit{IEEE Transactions on Mobile Computing}, vol. 24, no. 11, pp. 12726--12741, Nov. 2025, doi: 10.1109/TMC.2025.3586793.

\bibitem{ref45}
H. Hao, C. Xu, W. Zhang, S. Yang, and G.-M. Muntean, ``Joint task offloading, resource allocation, and trajectory design for multi-UAV cooperative edge computing with task priority,'' \textit{IEEE Transactions on Mobile Computing}, vol. 23, no. 9, pp. 8649--8663, Sept. 2024, doi: 10.1109/TMC.2024.3350078.

\bibitem{ref46}
N. Mu, Y. Luan, and Q.-S. Jia, ``Preference-based multi-objective reinforcement learning,'' \textit{IEEE Transactions on Automation Science and Engineering}, vol. 22, pp. 18737--18749, 2025, doi: 10.1109/TASE.2025.3589271.

\bibitem{ref47}
A. L. C., B. X. W., A. C. X., et al., ``Failure-resilient DAG task scheduling in edge computing,'' \textit{Computer Networks}, 2021, doi: 10.1016/j.comnet.2021.108361.

\bibitem{ref48}
M. Al-Osta, A. Bali, and A. Gherbi, ``Event-driven and semantic-based approach for data processing on IoT gateway devices,'' \textit{Journal of Ambient Intelligence and Humanized Computing}, vol. 10, no. 12, pp. 4663--4678, 2019.

\bibitem{ref49}
J. Li, W. Liang, M. Huang, and X. Jia, ``Reliability-aware network service provisioning in mobile edge-cloud networks,'' \textit{IEEE Transactions on Parallel and Distributed Systems}, vol. 31, no. 7, pp. 1545--1558, Jul. 2020.

\bibitem{ref50}
Q. Zhang, F. Liu, and C. Zeng, ``Adaptive interference-aware VNF placement for service-customized 5G network slices,'' in \textit{IEEE INFOCOM 2019 - IEEE Conference on Computer Communications}, IEEE, 2019, pp. 2449--2457.

\end{thebibliography}
```
