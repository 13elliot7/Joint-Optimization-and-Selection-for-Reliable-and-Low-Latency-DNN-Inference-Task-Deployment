```latex
\section{Conclusion}
\noindent
This paper investigates the deployment problem of DNN inference tasks with multiple optimization objectives in edge computing. We first define the problem model, and identify the multi-objective optimization goals of service reliability, accuracy reliability, and inference delay along with their corresponding constraints in a cloud-edge-end network. Subsequently, we propose an algorithm based on NSGA-II to explore the Pareto solution set of the problem. Additionally, we propose an expectation-based heuristic approach to find a suitable deployment strategy from the obtained Pareto solution set. Experimental results demonstrate that compared to baseline methods, our algorithm exhibits a superior Pareto frontier in generating the Pareto solution set for DNN inference task deployment strategies. Our algorithm also demonstrates advantages in terms of reliability and inference delay in DNN inference task deployment, with minimal failed deployments and feasible running times.
Numerous opportunities lie ahead for future research endeavors. With the rising complexity of edge computing network scenarios, the spotlight on multi-objective optimization will continue to intensify. The domain of reinforcement learning has received considerable attention in the context of multi-objective optimization. Our forthcoming efforts will involve leveraging reinforcement learning to enhance our algorithms, enabling them to more adeptly adjust to evolving network environments. Furthermore, we aim to delve into the integration of additional optimization objectives, such as cost, to expand the scope of application scenarios for our algorithms.




\begin{thebibliography}{1}
\bibliographystyle{IEEEtran}



\bibitem{ref1}  
L. Jiao, D. Wang, Y. Bai, et al., ``Deep learning in visual tracking: A review,'' \textit{IEEE Transactions on Neural Networks and Learning Systems}, vol. 34, no. 9, pp. 5497–5516, 2021.  
\bibitem{ref2}  
W. Samek, G. Montavon, S. Lapuschkin, et al., ``Explaining deep neural networks and beyond: A review of methods and applications,'' \textit{Proceedings of the IEEE}, vol. 109, no. 3, 2021. DOI: 10.1109/JPROC.2021.3060483.  
\bibitem{ref3}  
Y. Tang, C. Zhao, J. Wang, et al., ``Perception and navigation in autonomous systems in the era of learning: A survey,'' \textit{IEEE Transactions on Neural Networks and Learning Systems}, 2022.  

\bibitem{ref4}  
N. Pathak, S. Misra, A. Mukherjee, et al., ``UAV virtualization for enabling heterogeneous and persistent UAV-as-a-service,'' \textit{IEEE Transactions on Vehicular Technology}, vol. 69, no. 6, pp. 6731–6738, 2020.  

\bibitem{ref5}  
V. Sze, Y. H. Chen, T. J. Yang, et al., ``Efficient processing of deep neural networks: A tutorial and survey,'' \textit{Proceedings of the IEEE}, vol. 105, no. 12, pp. 2296–2319, 2017. DOI: 10.1109/JPROC.2017.2761740.  

\bibitem{ref6}  
Gill S S, Golec M, Hu J, et al. ``Edge AI: A taxonomy, systematic review and future directions,'' in \textit{Cluster Computing}, 2025, 28(1): 18.

\bibitem{ref7}  
M. Patel, B. Naughton, C. Chan, N. Sprecher, S. Abeta, and A. Neal, ``Mobile-edge computing introductory technical white paper,'' \textit{White Paper, Mobile-edge Computing (MEC)
industry initiative, 2014}.  

\bibitem{ref8}  
X. Wang, Y. Han, V. C. M. Leung, et al., ``Convergence of edge computing and deep learning: A comprehensive survey,'' \textit{IEEE Communications Surveys \& Tutorials}, vol. 22, no. 2, pp. 869–904, 2020.

\bibitem{ref9}  
H. Ma, R. Li, X. Zhang, et al., ``Reliability-aware online scheduling for DNN inference tasks in mobile edge computing,'' \textit{IEEE Internet of Things Journal}, 2023.  

\bibitem{ref10}  
S. Teerapittayanon, B. McDanel, and H. T. Kung, ``Distributed deep neural networks over the cloud, the edge and end devices,'' 2017. DOI: 10.48550/arXiv.1709.01921.  

\bibitem{ref11}  
X. Deng, J. Yin, P. Guan, et al., ``Intelligent delay-aware partial computing task offloading for multiuser industrial Internet of Things through edge computing,'' \textit{IEEE Internet of Things Journal}, 2023.

\bibitem{ref12}  
M. Min, D. Xu, L. Xiao, et al., ``Learning-based computation offloading for IoT devices with energy harvesting,'' 2017. DOI: 10.48550/arXiv.1712.08768. 
  

\bibitem{ref13}  
Y. Huang, F. Wang, F. Wang, et al., ``DeePar: A hybrid device-edge-cloud execution framework for mobile deep learning applications,'' in \textit{IEEE INFOCOM 2019 - IEEE Conference on Computer Communications Workshops (INFOCOM WKSHPS)}, IEEE, 2019. DOI: 10.1109/INFCOMW.2019.8845240.  

\bibitem{ref14}  
T. Mohammed, C. Joe-Wong, R. Babbar, and M. D. Francesco, ``Distributed inference acceleration with adaptive DNN partitioning and offloading,'' in \textit{Proc. INFOCOM}, 2020, pp. 854–863.  


\bibitem{ref15}  
J. Li, W. Liang, Y. Li, et al., ``Throughput maximization of delay-aware DNN inference in edge computing by exploring DNN model partitioning and inference parallelism,'' \textit{IEEE Transactions on Mobile Computing}, vol. 22, no. 5, pp. 3017–3030, 2021.  

\bibitem{ref16}
X. Xu, K. Yan, S. Han, B. Wang, X. Tao and P. Zhang, ``Learning-Based Edge-Device Collaborative DNN Inference in IoVT Networks,'' in \textit{IEEE Internet of Things Journal}, vol. 11, no. 5, pp. 7989-8004, 1 March1, 2024, doi: 10.1109/JIOT.2023.3317785.

\bibitem{ref17}  
J. Xu, H. Yao, R. Zhang, T. Mai and M. Guizani, ``Low Latency and Accuracy-Guaranteed DNN Inference for UAV-Assisted IoT Networks,'' in \textit{IEEE Transactions on Cognitive Communications and Networking}, doi: 10.1109/TCCN.2025.3542443.

\bibitem{ref18}  
Z. Zhuang et al., ``DECC: Delay-Aware Edge-Cloud Collaboration for Accelerating DNN Inference,'' in \textit{IEEE Transactions on Emerging Topics in Computing}, vol. 13, no. 2, pp. 438-450, April-June 2025, doi: 10.1109/TETC.2024.3404551.

\bibitem{ref19}  
B. Lin, Y. Huang, J. Zhang, et al., ``Cost-driven off-loading for DNN-based applications over cloud, edge, and end devices,'' \textit{IEEE Transactions on Industrial Informatics}, vol. 16, no. 8, pp. 5456–5466, 2020. DOI: 10.1109/TII.2019.2961237.  

\bibitem{ref20}  
Z. Xu, et al., ``Energy-aware inference offloading for DNN-driven applications in mobile edge clouds,'' \textit{IEEE Transactions on Parallel and Distributed Systems}, vol. 32, no. 4, pp. 799–814, Apr. 2021.  

\bibitem{ref21}
W. Jiang, H. Han, Q. Wang, L. Qian, F. Wei and G. Feng, ``Energy-Efficient Resource Allocation for Accuracy-Aware Cooperative DNN Inference in IoT,'' \textit{IEEE INFOCOM 2025 - IEEE Conference on Computer Communications Workshops (INFOCOM WKSHPS)}, London, United Kingdom, 2025, pp. 1-6, doi: 10.1109/INFOCOMWKSHPS65812.2025.11152853.

\bibitem{ref22}  
H. Liu, L. Cao, T. Pei, Q. Deng, and J. Zhu, ``A fast algorithm for energy-saving offloading with reliability and latency requirements in multi-access edge computing,'' \textit{IEEE Access}, vol. 8, pp. 151–161, 2020.  

\bibitem{ref23}  
J. Zilic, A. Aral, and I. Brandi, ``EFPO: Energy efficient and failure predictive edge offloading,'' in \textit{Proc. 12th IEEE/ACM International Conference on Ubiquitous Computing and Communications (UCC)}, 2019, pp. 165–175.  
\bibitem{ref24}  
K. Liu, C. Liu, G. Yan, V. C. S. Lee, and J. Cao, ``Accelerating DNN inference with reliability guarantee in vehicular edge computing,'' \textit{IEEE/ACM Transactions on Networking}, vol. 31, no. 6, pp. 3238–3253, Dec. 2023. DOI: 10.1109/TNET.2023.3279512.  

\bibitem{ref25}  
M. Taheri, et al., ``Exploration of activation fault reliability in quantized systolic array-based DNN accelerators,'' in \textit{2024 25th International Symposium on Quality Electronic Design (ISQED)}, San Francisco, CA, USA, 2024, pp. 1–8. DOI: 10.1109/ISQED60706.2024.10528372.

\bibitem{ref26}
G. Yan, C. Liu and K. Liu, ``ASPM: Reliability-Oriented DNN Inference Partition and Offloading in Vehicular Edge Computing,'' \textit{2023 IEEE 26th International Conference on Intelligent Transportation Systems (ITSC)}, Bilbao, Spain, 2023, pp. 3298-3303, doi: 10.1109/ITSC57777.2023.10422172.

\bibitem{ref27}
M. Deutel, G. Kontes, at al., ``Multi-Objective Bayesian Optimization with Reinforcement Learning for Edge Deployment of DNNs on Microcontrollers,'' in \textit{Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO '25 Companion)}. Association for Computing Machinery, New York, NY, USA, 19–20.

\bibitem{ref28}  
K. Deb and H. Jain, ``An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach, part I: Solving problems with box constraints,'' \textit{IEEE Transactions on Evolutionary Computation}, vol. 18, no. 4, pp. 577–601, 2014. DOI: 10.1109/TEVC.2013.2281535.  

\bibitem{ref29}  
C. A. C. Coello and M. S. Lechuga, ``MOPSO: A proposal for multiple objective particle swarm optimization,'' in \textit{Proc. 2002 Congress on Evolutionary Computation (CEC'02)}, 2002, pp. 1051–1056.

\bibitem{ref30}  
J. Xu, Y. Tian, P. Ma, et al., ``Prediction-guided multi-objective reinforcement learning for continuous robot control,'' in \textit{International Conference on Machine Learning}, PMLR, 2020, pp. 10607–10616.  

\bibitem{ref31}
F. Song et al., ``Evolutionary Multi-Objective Reinforcement Learning Based Trajectory Control and Task Offloading in UAV-Assisted Mobile Edge Computing,'' in \textit{IEEE Transactions on Mobile Computing}, vol. 22, no. 12, pp. 7387-7405, Dec. 2023, doi: 10.1109/TMC.2022.3208457. 

\bibitem{ref32}  
W. Ma, L. Zheng, and H. Zhou, ``A novel edge computing optimization method based on multi-objective optimization theory,'' in \textit{2022 Global Conference on Robotics, Artificial Intelligence and Information Technology (GCRAIT)}, Chicago, IL, USA, 2022, pp. 128–130. DOI: 10.1109/GCRAIT55928.2022.00035.  

\bibitem{ref33}
M. Mounesan, X. Zhang and S. Debroy, ``Infer-EDGE: Dynamic DNN Inference Optimization in Just-in-Time Edge-AI Implementations,'' in \textit{NOMS 2025-2025 IEEE Network Operations and Management Symposium}, Honolulu, HI, USA, 2025, pp. 1-9, DOI: 10.1109/NOMS57970.2025.11073623.

\bibitem{ref34}
H. Huang, J. Liang and G. Min, ``Joint DNN Model Deployment, Selection, and Configuration for Heterogeneous Inference Services Toward Edge Intelligence,'' in \textit{IEEE Transactions on Mobile Computing}, vol. 24, no. 11, pp. 12726-12741, Nov. 2025, doi: 10.1109/TMC.2025.3586793.

\bibitem{ref35}
H. Hao, C. Xu, W. Zhang, S. Yang and G. -M. Muntean, ``Joint Task Offloading, Resource Allocation, and Trajectory Design for Multi-UAV Cooperative Edge Computing With Task Priority,'' in \textit{IEEE Transactions on Mobile Computing}, vol. 23, no. 9, pp. 8649-8663, Sept. 2024, doi: 10.1109/TMC.2024.3350078.

\bibitem{ref36}
N. Mu, Y. Luan and Q. -S. Jia, ``Preference-Based Multi-Objective Reinforcement Learning,'' in \textit{IEEE Transactions on Automation Science and Engineering}, vol. 22, pp. 18737-18749, 2025, DOI: 10.1109/TASE.2025.3589271.

\bibitem{ref37}  
A. L. C., B. X. W., A. C. X., et al., ``Failure-resilient DAG task scheduling in edge computing,'' \textit{Computer Networks}, 2021. DOI: 10.1016/j.comnet.2021.108361.  

\bibitem{ref38}  
M. Al-Osta, A. Bali, and A. Gherbi, ``Event-driven and semantic-based approach for data processing on IoT gateway devices,'' \textit{Journal of Ambient Intelligence and Humanized Computing}, vol. 10, no. 12, pp. 4663–4678, 2019.  
\bibitem{ref39}  
J. Li, W. Liang, M. Huang, and X. Jia, ``Reliability-aware network service provisioning in mobile edge-cloud networks,'' \textit{IEEE Transactions on Parallel and Distributed Systems}, vol. 31, no. 7, pp. 1545–1558, Jul. 2020.

\bibitem{ref40}  
Q. Zhang, F. Liu, and C. Zeng, ``Adaptive interference-aware VNF placement for service-customized 5G network slices,'' in \textit{IEEE INFOCOM 2019 - IEEE Conference on Computer Communications}, IEEE, 2019, pp. 2449–2457.  


\end{thebibliography}
\end{document}
```

