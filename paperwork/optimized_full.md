```latex
\begin{abstract}
Deep neural network (DNN) inference services are increasingly deployed across cloud-edge-end collaborative environments to support real-time intelligent applications. However, online DNN inference deployment remains challenging because DNN tasks usually contain directed acyclic graph (DAG) dependencies, while edge resources, network bandwidth, node reliability, and link reliability vary over time under continuous workloads. In addition, different applications may have heterogeneous preferences for service reliability, inference accuracy, inference delay, and total energy consumption. Existing deployment methods often optimize a single objective, rely on fixed weighted aggregation, or ignore the dynamic degradation caused by accumulated load and heat, making them insufficient for long-running online inference systems. To address these challenges, this paper formulates a dynamic multi-objective DNN inference deployment problem in a cloud-edge-end environment. The proposed model jointly characterizes DAG-structured inference, online resource-state evolution, reliability degradation, and energy consumption. Based on this model, we propose DAREED, a DAG-Aware Reliable and Energy-Efficient Evolutionary Deployment algorithm. DAREED combines DAG-aware three-objective Pareto candidate generation over inference accuracy, service reliability and inference delay with energy-aware search guidance and preference-based deployment selection. These mechanisms enable DAREED to generate high-quality candidate deployments while exploiting DNN topology and dynamic resource states. Comprehensive simulations evaluate DAREED in terms of Pareto front quality, deployment performance, robustness, scalability, and preference adaptability. The results demonstrate that DAREED achieves a favorable balance among reliability, accuracy, delay, and energy efficiency under online DNN inference workloads.
\end{abstract}

\section{Introduction}
\IEEEPARstart{I}{n} recent years, artificial intelligence technologies, particularly deep learning, have catalyzed the proliferation of numerous intelligent mobile applications \cite{ref1}, owing to their strong feature learning capabilities and high inference accuracy \cite{ref2}. Intelligent services leveraging deep neural networks (DNNs) for real-time inference and decision-making have garnered significant attention and adoption, spanning domains such as intelligent security and intelligent inspection \cite{ref3}-\cite{ref4}. DNN inference involves deriving computational results through tightly integrated computational tasks with the DNN model \cite{ref5}. This inference process typically encompasses model training and online inference. During model training, the DNN model undergoes iterative training to adjust its weights and is then stored as an accessible service model on a cloud server \cite{ref6}. Subsequently, online inference entails deploying the trained model on end devices, edge servers, or cloud servers to execute DNN inference tasks and deliver intelligent services that support swift decision-making and real-time perception. Mobile edge computing (MEC), leveraging computing and storage resources at the network edge, emerges as a promising technology paradigm \cite{ref7}. Consequently, the deployment of DNN inference tasks, particularly in online collaborative inference involving cloud, edge and end, has become a research focus \cite{ref8}-\cite{ref12}.

In cloud-edge-end collaborative inference, a DNN task is usually composed of multiple dependent layers or subtasks. These subtasks form a directed acyclic graph (DAG), where each node represents computation and each edge represents intermediate data transmission. By partitioning and deploying DNN subtasks on different computing nodes, the system can exploit heterogeneous computing resources and reduce end-to-end delay. However, the deployment decision is highly challenging. First, end devices and edge servers have limited and time-varying resources. A deployment scheme that is feasible at one moment may become unsuitable later due to resource occupation by previously accepted DNN requests. Second, network links also become dynamic under online workloads. Congested links may suffer from bandwidth degradation, increasing transmission delay and reducing communication reliability. Third, edge servers and end devices are more vulnerable to failures and performance fluctuation than cloud servers. In long-running online inference systems, node load and accumulated heat may reduce both service reliability and inference accuracy. Therefore, DNN inference deployment must consider not only inference delay, but also service reliability, inference accuracy and the dynamic reliability of transmission links.

Energy consumption is another important factor in DNN inference deployment. Edge servers and end devices are often deployed in resource-constrained environments, and frequent DNN inference may introduce considerable computation and communication energy consumption. Deploying more subtasks on powerful cloud or edge nodes may improve reliability or processing speed, but it may also increase transmission energy and network load. Conversely, local execution may reduce communication cost but can suffer from limited computing capacity and lower reliability. Therefore, reliable and energy-efficient inference deployment requires a careful balance among service reliability, inference accuracy, inference delay and total energy consumption.

Existing studies on DNN inference deployment have investigated model partitioning, service placement, task offloading and resource scheduling in edge computing. Many of them focus on minimizing inference delay or energy consumption, while reliability is often modeled as a static node attribute or ignored. Some multi-objective methods combine different optimization goals into a single weighted objective. However, fixed weights are difficult to adapt to heterogeneous DNN requests and changing system states. In addition, many deployment strategies treat DNN inference as a linear chain or a coarse-grained task, which cannot fully exploit the DAG structure of DNN computation. The lack of DAG-aware evolutionary operators may lead to inefficient search, broken dependency locality and poor deployment quality. Furthermore, when multiple DNN requests arrive online, the accumulated load and heat of nodes and links make reliability and bandwidth time-varying, which calls for a dynamic deployment mechanism rather than a one-shot static placement policy.

Motivated by these observations, this paper studies dynamic reliable and energy-efficient DNN inference deployment in a cloud-edge-end collaborative environment. We model DNN inference requests as DAGs and describe the substrate network through dynamic resource, reliability and energy states. A DNN request is regarded as a running block after being accepted, occupying selected nodes and links for several time slots and continuously affecting subsequent deployment decisions. Based on this model, the deployment problem is formulated as a multi-objective optimization problem that jointly considers service reliability, inference accuracy, inference delay and total energy consumption.

To solve the problem, we propose a DAG-Aware Reliable and Energy-Efficient Evolutionary Deployment algorithm, named DAREED. DAREED adopts a customized evolutionary framework that combines DAG-aware Pareto search, resource-aware variation, constraint handling and energy-aware preference-based deployment selection. This design allows the algorithm to preserve the exploration capability of genetic evolution while explicitly exploiting DNN topology and dynamic edge network states.

The main contributions of this paper are summarized as follows.

    • We formulate a dynamic cloud-edge-end DNN inference deployment model for online collaborative inference. The model jointly characterizes DAG-structured inference dependencies, time-varying resource states, reliability degradation and energy consumption. Based on this model, the deployment problem is formulated as a multi-objective optimization problem that jointly considers service reliability, inference accuracy, inference delay and total energy consumption.\par

    • We propose DAREED, a DAG-aware reliable and energy-efficient evolutionary deployment algorithm. DAREED integrates three-objective Pareto candidate generation, DAG-aware evolutionary operators, resource-aware variation, constraint-aware selection and energy-aware preference-based deployment selection. These mechanisms enable the algorithm to exploit DNN dependency structures and dynamic cloud-edge-end states while preserving the exploration capability of genetic evolution.\par

    • We conduct comprehensive simulations to evaluate DAREED in terms of Pareto front quality, deployment performance, robustness, scalability and application preference adaptability. The experimental design provides a systematic basis for validating the effectiveness of DAREED under online DNN inference workloads.\par

The remainder of this paper is organized as follows. Section 2 reviews related work on DNN inference deployment, reliable edge computing and multi-objective optimization. Section 3 presents the motivation of this work. Section 4 formulates the dynamic DNN inference deployment problem. Section 5 presents the proposed DAREED algorithm. Section 6 evaluates the performance of DAREED through simulations. Finally, Section 7 concludes this paper.

\section{Related Work}
\noindent
This section reviews the studies related to DNN inference deployment, reliability-aware edge computing, and multi-objective optimization. The discussion focuses on the limitations of existing deployment models under dynamic cloud-edge-end inference environments and motivates the design of a DAG-aware evolutionary deployment algorithm.

\subsection{DNN Inference Task Deployment}
\noindent
Existing studies on DNN inference task deployment can be broadly classified into delay-oriented, resource efficiency-oriented, and reliability-oriented approaches.

\subsubsection{Delay-oriented Deployment}
\noindent
In terms of delay-oriented optimization, several approaches have been proposed to accelerate DNN inference across end devices, edge servers, and cloud servers. Y. Huang et al. \cite{ref13} proposed the DeePar framework, which improves DNN inference performance by orchestrating the allocation of computing resources across end devices, edge servers, and cloud servers. T. Mohammed et al. \cite{ref14} proposed a distributed algorithm based on exchange matching to deploy DNN inference requests from multiple users onto edge servers, aiming to minimize the maximum end-to-end delay experienced by users. Teerapittayanon S. et al. \cite{ref10} proposed a collaborative DNN inference architecture that operates across the cloud and edge sides. By deploying the shallow part of a neural network to edge nodes and end devices, the architecture effectively reduces inference delay. Deng X. et al. \cite{ref11} proposed a deep reinforcement learning based task deployment method for delay-sensitive tasks. Li J. et al. \cite{ref15} investigated throughput maximization for DNN inference tasks under delay constraints and devised offline and online algorithms to ensure that each request satisfies its inference delay requirement while maximizing the number of processed requests.

Some researchers further studied collaborative DNN partitioning to accelerate inference in resource-constrained networks. X. Xu et al. \cite{ref16} proposed a deep reinforcement learning based edge-device collaborative DNN inference framework for IoVT networks, jointly optimizing model partitioning, early-exit selection, and resource allocation to balance inference delay and accuracy. J. Xu et al. \cite{ref17} introduced a constrained deep reinforcement learning algorithm for UAV-assisted IoT networks, jointly optimizing task assignment, DNN model selection by knowledge distillation, and bandwidth allocation to minimize average inference delay while satisfying strict accuracy requirements. Z. Zhuang et al. \cite{ref18} proposed a delay-aware edge-cloud collaboration algorithm, which accelerates DNN inference by pipelining the execution of a DNN model partitioned into multiple independent branches and distributed between the edge and the cloud.

\subsubsection{Resource Efficiency-oriented Deployment}
\noindent
Resource efficiency is another important concern in DNN inference deployment. Lin B. et al. \cite{ref19} studied the distributed deployment of DNN inference tasks across cloud servers, edge nodes, and devices, and proposed an adaptive discrete particle swarm optimization algorithm to enhance resource utilization while satisfying inference delay constraints. Z. Xu et al. \cite{ref20} investigated DNN inference offloading in MEC networks and presented both a random algorithm and an online algorithm for real-time applications to minimize total energy consumption. Min M. et al. \cite{ref12} introduced a Q-learning based task deployment scheme that selects edge nodes by considering device energy and wireless bandwidth between edge nodes. W. Jiang et al. \cite{ref21} jointly optimized DNN branch selection, partition layer selection, and computing and communication resource allocation to minimize the total energy consumption of IoT devices. Recent edge inference systems further emphasize the practical importance of energy-aware runtime decisions. M. Mounesan et al. \cite{ref22} designed Infer-EDGE to balance latency, inference accuracy, and device energy consumption in just-in-time edge-AI implementations, while Z. Zhang et al. \cite{ref23} proposed E4, which integrates early-exit inference and DVFS to reduce energy consumption for edge video analytics.

\subsubsection{Reliability-oriented Deployment}
\noindent
Although many studies focus on delay or resource efficiency, reliability is becoming increasingly important for DNN inference services. H. Liu et al. \cite{ref24} incorporated reliability requirements as constraints to achieve energy-efficient offloading strategies. J. Zilic et al. \cite{ref25} integrated reliability objectives and system performance optimization into the offloading decision-making process through fault prediction based on historical data. Some studies regard reliability as an optimization objective. Ma H. et al. \cite{ref9} considered reliability as the optimization objective in DNN inference task scheduling. Liu et al. \cite{ref26} studied centralized and distributed resource allocation schemes to maximize inference reliability in vehicular edge computing environments. Reliability has also been studied in DNN accelerator design. Taheri et al. \cite{ref27} proposed an automated framework for exploring the tradeoff among quantization, activation fault reliability, model accuracy, and hardware efficiency in DNN accelerators. G. Yan et al. \cite{ref28} used overlapped DNN partitioning and mobility-aware offloading to jointly address acceleration and reliability in vehicular edge computing.

In summary, most existing studies focus on minimizing inference delay or improving resource efficiency, while reliability is often treated as a constraint, a static node attribute, or a single optimization objective. However, DNN inference deployment in cloud-edge-end environments involves multiple tightly coupled objectives, including service reliability, inference accuracy, inference delay, and total energy consumption. The challenge of finding an appropriate deployment also appears across the compute continuum. M. Deutel et al. \cite{ref29} employed Multi-Objective Bayesian Optimization combined with reinforcement learning to balance conflicting objectives such as accuracy, memory, and computational complexity on resource-constrained microcontroller units. These studies indicate that single-objective optimization and simple weighted aggregation are insufficient for DNN deployment under heterogeneous and dynamic environments.

Recent studies also show the importance of preserving the graph structure of DNN inference during scheduling and deployment. Y. Du et al. \cite{ref30} studied multi-UAV collaborative edge inference with a multi-branch DNN and developed GA-based schedulers, including GA-DAG, to reduce end-to-end latency by exploiting DNN execution dependencies. J. Cao et al. \cite{ref31} proposed EdgeServing, a deadline-aware multi-DNN serving system that jointly selects model, exit point, and batch size under edge latency constraints. These studies highlight the need to consider DNN execution structure and runtime scheduling context, but they do not jointly address DAG-aware deployment, dynamic service reliability, inference accuracy, inference delay, and total energy consumption in online cloud-edge-end environments.

\subsection{Reliability-aware Edge Computing}
\noindent
Reliability-aware edge computing aims to maintain service quality by improving the robustness of task offloading, scheduling, and resource allocation decisions. Existing studies usually model reliability as a static node property, a constant failure probability, or a physical constraint known before deployment. J. Zhou et al. \cite{ref32} studied joint offloading and scheduling in multi-user MEC systems, where the execution success probability of processors is modeled by a static exponential distribution. A. M. Rasouli et al. \cite{ref33} proposed RASOUL, an online reinforcement learning strategy that selects edge servers for IoT devices according to server availability, but the availability is derived from long-term mean time between failures and does not capture the cumulative effect of continuous workloads. In UAV-assisted edge computing, H. Hao et al. \cite{ref34} modeled reliability-aware task offloading and improved task success rate, while device failure rates are still treated as static prior distributions. Z. Zhou et al. \cite{ref35} proposed a reliability-aware real-time task mapping method for multi-server edge systems with adaptive DVFS against transient faults, but the hard-failure risk of physical devices is assumed to be known offline.

In long-running edge systems, reliability can also be affected by dynamic physical degradation. When edge nodes process computation-intensive DNN inference tasks, high load may lead to heat accumulation, and the accumulated thermal stress can reduce service stability. The physics of integrated circuits shows that degradation mechanisms such as negative-bias temperature instability and electromigration are highly sensitive to operating temperature. J. Zhou et al. \cite{ref36} proposed a scheduling framework that jointly optimizes lifetime, energy, and makespan in multiprocessor systems by tracking transient temperature through an RC thermal network and updating cumulative device degradation. For distributed IoT systems, K. Ergun et al. \cite{ref37} proposed a dynamic reliability management scheme that adjusts offloading ratios according to the physical degradation states accumulated from historical gateway activities. For machine learning inference, O. Shafi et al. \cite{ref38} developed IceEdge, a thermal-aware inference serving framework that controls GPU temperature and improves service stability for compact edge servers running deep learning models.

These studies show that ignoring load- and heat-induced degradation may lead to deployment decisions that become unreliable during long-term operation. However, existing reliability-aware edge computing studies are rarely designed for DAG-structured DNN inference tasks. They usually do not jointly capture layer dependency, dynamic node reliability, link reliability, inference accuracy, inference delay, and total energy consumption in an online deployment model. This paper addresses this gap by modeling DNN inference requests as DAGs and evaluating service reliability and inference accuracy under time-varying node and link states.

\subsection{Multi-Objective Optimization}
\noindent
Many real-world problems are characterized by the coexistence of multiple objectives. Balancing these objectives under various constraints requires the consideration of competing or complementary goals. Multi-objective optimization usually involves exploring a set of tradeoff solutions instead of deriving only one deterministic optimum, which makes it suitable for deployment problems with heterogeneous service requirements.

Multi-objective optimization has attracted extensive attention in many fields. Numerous algorithms have been developed, including genetic algorithms, NSGA-II \cite{ref39}, multi-objective particle swarm optimization \cite{ref40}, and multi-objective reinforcement learning. Xu J. et al. \cite{ref41} proposed a prediction-guided multi-objective reinforcement learning algorithm to discover the Pareto front for continuous robot control. F. Song et al. \cite{ref42} used an improved evolutionary multi-objective reinforcement learning algorithm to determine UAV trajectory control and task offloading policies. Ma W. et al. \cite{ref43} devised an edge computing optimization algorithm based on multi-objective optimization principles. M. Mounesan et al. \cite{ref22} proposed an advantage actor-critic reinforcement learning framework to balance end-to-end delay, inference accuracy, and device energy consumption. H. Huang et al. \cite{ref44} jointly optimized deployment location, DNN model selection, and application configuration through a heterogeneous-agent reinforcement learning algorithm to simultaneously optimize delay, accuracy, and resource cost. H. Hao et al. \cite{ref45} proposed a deep reinforcement learning algorithm based on latent space to maximize the long-term average system gain composed of energy consumption, task delay, and priority in a multi-UAV cooperative edge computing system.

Although general multi-objective algorithms provide powerful global search capability, directly applying them to DNN inference deployment still faces significant challenges. Generic algorithms usually treat the decision space as a black box. Without exploiting the layer dependencies of DNN computation graphs and heterogeneous cloud-edge-end resource constraints, the search space grows rapidly with the number of DNN layers and physical nodes. Blind search may generate many infeasible solutions that violate dependency, hierarchy, CPU, link, or delay constraints, resulting in unstable convergence and low search efficiency.

Existing studies on DNN inference deployment \cite{ref9}, \cite{ref11}, \cite{ref15} often transform multiple objectives into a single objective through weighted aggregation. However, fixed weights are difficult to determine in dynamic network scenarios, especially when different DNN requests have different preferences for service reliability, inference accuracy, inference delay, and total energy consumption. N. Mu et al. \cite{ref46} showed that preference information can guide multi-objective reinforcement learning toward Pareto-optimal policies without relying on manually designed scalar reward functions. Nevertheless, how to efficiently generate high-quality Pareto candidates for DAG-structured online DNN inference and how to select a final deployment according to request preferences remain challenging.

To address these issues, this paper designs DAREED, a DAG-aware reliable and energy-efficient evolutionary deployment algorithm. DAREED incorporates DNN topology information into initialization, crossover, mutation, repair, constraint-dominance sorting, and local search. By combining multi-objective evolutionary search with DAG-aware operators and dynamic reliability-energy evaluation, DAREED reduces invalid exploration and supports preference-based deployment selection under online cloud-edge-end inference workloads.

\section{Motivation}
\noindent
An illustrative example of cloud-edge-end collaborative DNN inference deployment is shown in \autoref{fig_motivation}. The physical infrastructure contains three types of computing nodes, including end devices, edge servers, and a cloud server. A DNN inference request is initiated by an end device and consists of multiple dependent inference layers. These layers form a DAG, where each vertex represents a computation subtask and each directed edge represents intermediate data transmission. A deployment scheme needs to determine not only where each inference layer is executed, but also how the intermediate data are transmitted among selected nodes.

This deployment decision is challenging for three reasons. First, different DNN inference requests may have different service preferences. For example, an augmented reality request may be highly delay-sensitive, an industrial inspection request may require high inference accuracy, and a safety-critical monitoring request may place more emphasis on service reliability. Meanwhile, battery-powered or resource-constrained scenarios may prefer lower total energy consumption. A single-objective strategy can optimize only one aspect of the deployment decision and may sacrifice other important metrics. For instance, deploying most layers near the initiating end device may reduce inference delay, but the limited computing capability and lower reliability of end devices may reduce inference accuracy and service reliability. Deploying all layers on the cloud may improve reliability and computation capability, but it may introduce long transmission paths and higher communication energy consumption.

Second, the deployment quality is strongly affected by the DAG structure of DNN inference. In a chain-like or coarse-grained view, moving one layer only changes the execution location of that layer. In a DAG-structured inference task, however, moving a layer may affect multiple predecessor and successor transmissions. Therefore, a deployment that appears attractive from the perspective of node computing capability may still cause large intermediate transmission delay, low path reliability, or excessive link energy consumption. This observation indicates that the search process should explicitly exploit the topological order, dependency relations, and critical path of DNN inference tasks.

Third, online DNN inference deployment is inherently dynamic. Once a DNN request is accepted, it occupies selected nodes and links for several time slots. As multiple requests arrive over time, node load, link load, heat state, effective bandwidth, service reliability, and inference accuracy may change continuously. A node that is reliable and fast at the current moment may become overloaded or overheated after serving several requests. Similarly, a link with sufficient bandwidth may become congested, leading to larger inference delay and lower communication reliability. Therefore, deployment decisions made without considering dynamic state evolution may become unsuitable during long-running online inference services.

These observations motivate a multi-objective deployment method that jointly considers service reliability, inference accuracy, inference delay, and total energy consumption. Instead of aggregating these objectives using fixed weights during candidate generation, it is more flexible to first generate a set of non-dominated candidate deployment schemes and then select a final scheme according to the preference of the current request. As illustrated in \autoref{fig_motivation}, a delay-sensitive request may select a candidate with shorter edge-side execution paths, a reliability-sensitive request may prefer nodes and links with higher dynamic reliability, an energy-sensitive request may avoid high-power computing nodes or long transmission paths, and a balanced request may choose a deployment that provides stable performance across all objectives.

The above motivation leads to the design of DAREED. The algorithm searches for high-quality candidate deployments through evolutionary optimization while embedding DAG-aware, resource-aware and constraint-aware mechanisms into the search process. These mechanisms help preserve dependency locality, guide the search according to dynamic resource states, and reduce invalid deployments. Finally, preference-based deployment selection chooses one concrete deployment scheme from the candidate set according to the application requirements of the current DNN inference request.

\begin{figure}[!t]
\centering
\includegraphics[width=2.5in]{fig_motivation.pdf}
\caption{Motivating example of dynamic DAG-aware DNN inference deployment in a cloud-edge-end environment.}
\label{fig_motivation}
\end{figure}

\section{Problem Formulation}
\noindent
In this section, the DNN inference task deployment problem is formulated under a dynamic cloud-edge-end environment. The model considers the time-varying load of computing nodes and transmission links, the reliability degradation caused by accumulated heat, the effective bandwidth variation of links, and the energy consumption of both computation and data transmission. All the notations used in this paper and their meanings are listed in \autoref{tab:table1}.

\begin{table}[!t]
\caption{Parameter symbols\label{tab:table1}}
\centering
\begin{tabular}{>{\centering\arraybackslash}m{3cm}p{5cm}}
\hline
Notation & Meaning\\
\hline
$G=(N,L)$ & Network topology\\

$N=N_{end}\cup N_{edge}\cup N_{cloud}$ & $N_{end}$, $N_{edge}$ and $N_{cloud}$ are end device node set, edge computing node set and cloud computing node set, respectively\\

$L$ & Directed link set, $\forall l=(m,n)\in L,\ m,n\in N,\ m\neq n$\\

$C^n_{cpu}$, $\bar C^n_{cpu}$ & Available CPU resource and maximum CPU capacity of node $n$\\

$\eta_n$ & Computing power of unit computing resource on node $n$\\

$P^{comp}_n$ & Computing power consumption coefficient of node $n$\\

$R^0_n$, $A^0_n$ & Initial service reliability and initial inference accuracy of node $n$\\

$R_n(t)$, $A_n(t)$ & Dynamic service reliability and dynamic inference accuracy of node $n$ at time slot $t$\\

$\rho_n(t)$, $h_n(t)$ & Load ratio and heat state of node $n$ at time slot $t$\\

$B_l$, $\hat B_l(t)$ & Basic bandwidth and effective bandwidth of link $l$ at time slot $t$\\

$Q^0_l$, $Q_l(t)$ & Initial reliability and dynamic reliability of link $l$\\

$\mu_l(t)$, $g_l(t)$ & Load ratio and heat state of link $l$ at time slot $t$\\

$\varepsilon_l$ & Unit transmission energy consumption of link $l$\\

$Task$ & DNN inference task request set\\

$task_i$ & A DNN inference task request\\

$G_i=(V_i,E_i)$ & DAG of DNN inference $task_i$\\

$V_i$ & The set of inference subtasks or layers of $task_i$\\

$E_i$ & The dependency edge set between inference subtasks of $task_i$\\

$C'^{i,j}_{cpu}$ & CPU resources required by inference layer $v^i_j$\\

$F_{i,j}$ & Computational workload required by inference layer $v^i_j$\\

$D^i_{p,q}$ & Data amount transmitted on dependency edge $(v^i_p,v^i_q)$\\

$D^i_{in}$, $D^i_{out}$ & Input data size and output result size of $task_i$\\

$T^i_{limit}$ & Delay requirement of $task_i$\\

$T^i_{all}(t)$ & Estimated total execution delay of $task_i$ under the system state of time slot $t$\\

$\Delta$ & Length of a discrete time slot\\

$S_i$ & Number of time slots occupied by accepted $task_i$\\

$x^n_{i,j}$ & If inference layer $v^i_j$ of $task_i$ is deployed on node $n$, it is 1; otherwise, it is 0\\

$y^l_{i,p,q}$ & If dependency edge $(v^i_p,v^i_q)$ of $task_i$ is mapped onto link $l$, it is 1; otherwise, it is 0\\

$y^l_{i,in}$, $y^l_{i,out}$ & If the input uploading path or output returning path of $task_i$ uses link $l$, it is 1; otherwise, it is 0\\

$n^i_{initiate}$ & The initiating node of $task_i$\\

$R'_i(t)$ & Dynamic service reliability of $task_i$\\

$A'_i(t)$ & Inference accuracy of $task_i$\\

$E^i_{comp}$, $E^i_{tran}$, $E^i_{all}$ & Computation energy, transmission energy and total energy consumption of $task_i$\\
\hline
\end{tabular}
\end{table}

\subsection{Resource Model}
\noindent
\subsubsection{Substrate Network}
\noindent
This paper focuses on a cloud-edge-end collaborative inference scenario. The physical network is denoted as $G=(N,L)$, where $N$ is the computing node set and $L$ is the communication link set. The node set consists of end devices, edge servers and cloud servers, namely $N=N_{end}\cup N_{edge}\cup N_{cloud}$. Each node $n\in N$ is characterized by its maximum CPU capacity $\bar C^n_{cpu}$, current available CPU resource $C^n_{cpu}$, floating-point computing rate $\eta_n$, computing power consumption coefficient $P^{comp}_n$, initial service reliability $R^0_n$, and initial inference accuracy $A^0_n$.

The network state is time-varying in the considered online inference scenario. The system is divided into discrete time slots $t=0,1,2,\cdots$. During each time slot, accepted DNN tasks occupy CPU and link resources continuously. Therefore, the current deployment decision is affected not only by the basic capacity of nodes and links, but also by the running DNN tasks that have not yet been completed. For node $n$, its load ratio at time slot $t$ is defined as
\begin{equation}
\label{eq:node_load}
\rho_n(t)=\frac{\bar C^n_{cpu}-C^n_{cpu}(t)}{\bar C^n_{cpu}}.
\end{equation}
The heat state of node $n$ is updated by an exponential moving average:
\begin{equation}
\label{eq:node_heat}
h_n(t+1)=\lambda_h h_n(t)+(1-\lambda_h)\rho_n(t),
\end{equation}
where $\lambda_h\in[0,1)$ controls the influence of historical load.

For each directed link $l\in L$, $B_l$ denotes its basic bandwidth and $\varepsilon_l$ denotes its unit transmission energy consumption. Let $\mu_l(t)$ and $g_l(t)$ denote the current load ratio and heat state of link $l$, respectively. The link heat is updated as
\begin{equation}
\label{eq:link_heat}
g_l(t+1)=\lambda_g g_l(t)+(1-\lambda_g)\mu_l(t),
\end{equation}
where $\lambda_g\in[0,1)$ is the heat recovery coefficient of links. To describe bandwidth degradation under congestion, the effective bandwidth of link $l$ at time slot $t$ is modeled as
\begin{equation}
\label{eq:eff_bw}
\hat B_l(t)=\max\left\{\gamma_{min}B_l,\frac{B_l}{1+\gamma_h g_l(t)+\gamma_\mu \mu_l(t)}\right\},
\end{equation}
where $\gamma_h$ and $\gamma_\mu$ represent the influence of link heat and link load, respectively, and $\gamma_{min}$ is the minimum bandwidth ratio.

\subsubsection{DNN Inference Task Model}
\noindent
Let $Task$ denote the set of DNN inference requests. A DNN inference task $task_i$ is initiated by an end device $n^i_{initiate}$ and has a delay requirement $T^i_{limit}$. Each DNN inference task is modeled as a directed acyclic graph $G_i=(V_i,E_i)$, where $V_i$ is the set of inference layers and $E_i$ is the dependency set among layers. For each inference layer $v^i_j\in V_i$, $C'^{i,j}_{cpu}$ denotes the CPU requirement and $F_{i,j}$ denotes the computational workload. For each dependency edge $(v^i_p,v^i_q)\in E_i$, $D^i_{p,q}$ denotes the intermediate data amount that needs to be transmitted when the two dependent layers are deployed on different nodes or connected through a network path.

In the dynamic execution model, a DNN task is not regarded as completed immediately after deployment. Once accepted, $task_i$ becomes a running block and occupies its selected nodes and links for several time slots. The running time is estimated by the delay model and then discretized as
\begin{equation}
\label{eq:slot_num}
S_i=\left\lceil\frac{T^i_{all}(t)}{\Delta}\right\rceil,
\end{equation}
where $\Delta$ is the length of a time slot. During these $S_i$ slots, the accepted task contributes to node load, link load, node heat and link heat, thereby influencing the deployment of subsequent DNN requests.

\subsection{Delay Model}
\noindent
The inference delay of a DNN inference task consists of input uploading delay, processing delay on the DAG critical path, inter-layer transmission delay, and output returning delay. Queuing delay and propagation delay are not the focus of this paper and are omitted for simplicity. The path between two computing nodes is selected according to the shortest path whose edge weight is the reciprocal of link bandwidth. Under the dynamic model, the transmission delay is calculated by the effective bandwidth of links.

If inference layer $v^i_j$ is deployed on node $n$, its processing delay is
\begin{equation}
\label{eq:proc_delay}
d^n_{i,j}=\frac{F_{i,j}}{\eta_n}.
\end{equation}
For a dependency edge $(v^i_p,v^i_q)$, let $P(n_p,n_q)$ be the link set on the selected path between the node hosting $v^i_p$ and the node hosting $v^i_q$. Its transmission delay at time slot $t$ is
\begin{equation}
\label{eq:tran_delay}
d^i_{p,q}(t)=\sum_{l\in P(n_p,n_q)}\frac{D^i_{p,q}}{\hat B_l(t)}.
\end{equation}
Similarly, the input uploading delay and output returning delay are calculated by
\begin{equation}
\label{eq:input_delay}
d^i_{in}(t)=\sum_{l\in P(n^i_{initiate},n_1)}\frac{D^i_{in}}{\hat B_l(t)},
\end{equation}
\begin{equation}
\label{eq:output_delay}
d^i_{out}(t)=\sum_{l\in P(n_{|V_i|},n^i_{initiate})}\frac{D^i_{out}}{\hat B_l(t)},
\end{equation}
where $n_1$ and $n_{|V_i|}$ are the deployment nodes of the first and last inference layers, respectively.

Let $\mathcal{P}_i$ denote the set of all directed paths in the DAG of $task_i$. The delay of a DAG path $p\in\mathcal{P}_i$ is the sum of processing delays and dependency transmission delays along this path. Therefore, the critical path delay is
\begin{equation}
\label{eq:critical_delay}
d^i_{cri}(t)=\max_{p\in\mathcal{P}_i}\left(\sum_{v^i_j\in p}\sum_{n\in N}x^n_{i,j}d^n_{i,j}
+\sum_{(v^i_p,v^i_q)\in p}d^i_{p,q}(t)\right).
\end{equation}
Consequently, the estimated inference delay of $task_i$ is
\begin{equation}
\label{eq:total_delay}
T^i_{all}(t)=d^i_{in}(t)+d^i_{cri}(t)+d^i_{out}(t).
\end{equation}

\subsection{Dynamic Reliability Model}
\noindent
The reliability of DNN inference tasks refers to the probability that the task can be continuously served and produce reliable inference results \cite{ref9}. Edge failures may occur more frequently than cloud failures \cite{ref47}, and resource-constrained edge devices may also suffer from preemption or unstable data processing during service execution \cite{ref48}, \cite{ref49}. In the proposed model, reliability is affected by both basic node quality and dynamic system state. High node load and accumulated heat may increase the probability of service degradation or failure. Similarly, congested and overheated links may reduce the reliability of intermediate data transmission.

For computing node $n$, the dynamic service reliability and inference accuracy at time slot $t$ are modeled as
\begin{equation}
\label{eq:dynamic_r}
R_n(t)=clip\left(R^0_n e^{-\alpha_r\rho_n(t)-\beta_r h_n(t)},R_{min},R^0_n\right),
\end{equation}
\begin{equation}
\label{eq:dynamic_a}
A_n(t)=clip\left(A^0_n e^{-\alpha_a\rho_n(t)-\beta_a h_n(t)},A_{min},A^0_n\right),
\end{equation}
where $\alpha_r$ and $\beta_r$ control the sensitivity of service reliability to node load and heat, $\alpha_a$ and $\beta_a$ control the sensitivity of inference accuracy, and $clip(\cdot)$ limits the value within the specified lower and upper bounds.

For link $l$, the dynamic reliability is modeled as
\begin{equation}
\label{eq:dynamic_link_r}
Q_l(t)=clip\left(Q^0_l e^{-\alpha_l\mu_l(t)-\beta_l g_l(t)},Q_{min},Q^0_l\right),
\end{equation}
where $\alpha_l$ and $\beta_l$ describe the reliability degradation caused by link load and heat.

For a deployment strategy of $task_i$, the dynamic service reliability considers both the unique computing nodes used by the task and the unique links traversed by its input, dependency and output transmissions. Let $N_i^x$ and $L_i^x$ denote these two sets. The service reliability of $task_i$ is
\begin{equation}
\label{eq:task_service_r}
R'_i(t)=\prod_{n\in N_i^x}R_n(t)\prod_{l\in L_i^x}Q_l(t).
\end{equation}
The inference accuracy is related to the amount of computation executed on each node. Let $F^i_n=\sum_{v^i_j\in V_i}x^n_{i,j}F_{i,j}$ be the workload assigned to node $n$, and $F^i_{all}=\sum_{v^i_j\in V_i}F_{i,j}$. The inference accuracy of $task_i$ is defined as
\begin{equation}
\label{eq:task_accuracy_r}
A'_i(t)=\prod_{n\in N_i^x}A_n(t)^{F^i_n/F^i_{all}}.
\end{equation}
This weighted form avoids repeatedly multiplying the same node accuracy for multiple small layers and better reflects the contribution of each node to the whole inference computation.

\subsection{Energy Model}
\noindent
To reduce resource cost while maintaining reliability and delay performance, energy consumption is incorporated as an additional optimization direction. The total energy consumption of $task_i$ consists of computation energy and transmission energy:
\begin{equation}
\label{eq:total_energy}
E^i_{all}=E^i_{comp}+E^i_{tran}.
\end{equation}
If inference layer $v^i_j$ is deployed on node $n$, its computation energy is the product of computation power and execution time. Therefore, the computation energy of $task_i$ is
\begin{equation}
\label{eq:comp_energy}
E^i_{comp}=\sum_{v^i_j\in V_i}\sum_{n\in N}x^n_{i,j}P^{comp}_n\frac{F_{i,j}}{\eta_n}.
\end{equation}
The transmission energy includes input uploading and intermediate data transmission among DNN layers. Since the output result is usually much smaller than the input and intermediate data, output returning energy is not included in this model. Thus, the transmission energy is
\begin{equation}
\label{eq:tran_energy}
E^i_{tran}=
\sum_{l\in P(n^i_{initiate},n_1)}\varepsilon_lD^i_{in}
+\sum_{(v^i_p,v^i_q)\in E_i}\sum_{l\in P(n_p,n_q)}\varepsilon_lD^i_{p,q}.
\end{equation}
For the evolutionary search procedure, total energy consumption is also transformed into a maximization-oriented energy utility:
\begin{equation}
\label{eq:energy_utility}
U^i_E=\frac{1}{E^i_{all}}.
\end{equation}

\subsection{Dynamic Mixed-Integer Formulation}
\noindent
Herein, the mathematical formulation of the DNN inference task deployment problem is proposed. The decision variable $x^n_{i,j}$ represents the deployment of inference layers on computing nodes, while $y^l_{i,p,q}$ represents the mapping of dependency transmissions onto physical links. Because the formulation includes nonlinear reliability products, exponential degradation, reciprocal delay and energy utilities, and dynamic state transitions, it is a nonlinear mixed-integer model rather than a linear integer program.

\subsubsection{Constraint}
\noindent
\textbf{Single deployment constraint}: Each inference layer must be deployed on exactly one computing node:
\begin{equation}
\label{eq:single_deploy}
\sum_{n\in N}x^n_{i,j}=1,\quad \forall task_i\in Task,\ v^i_j\in V_i.
\end{equation}
In addition, an inference layer can only be deployed on the initiating end device, an edge node, or a cloud node. Other end devices are not allowed to host this task:
\begin{equation}
\label{eq:end_constraint}
x^n_{i,j}=0,\quad \forall n\in N_{end}\setminus\{n^i_{initiate}\}.
\end{equation}

\noindent
\textbf{Computing resource constraint}: At each time slot, the CPU resources occupied by all running DNN tasks and the newly accepted task cannot exceed the maximum CPU capacity of each node:
\begin{equation}
\label{eq:resource_constraint}
\sum_{task_i\in \Omega(t)}\sum_{v^i_j\in V_i}x^n_{i,j}C'^{i,j}_{cpu}\leq \bar C^n_{cpu},\quad \forall n\in N,
\end{equation}
where $\Omega(t)$ denotes the set of running tasks at time slot $t$ together with the candidate task to be deployed.

\noindent
\textbf{Link load constraint}: The bandwidth occupied by all running DNN tasks on each link should not exceed the current effective bandwidth. Since the exact completion delay and the occupied bandwidth are mutually dependent, the link load is approximated by evenly spreading the transmitted data over an estimated running time $\widehat T^i_{all}(t)$ obtained from the current effective-bandwidth state before admission:
\begin{equation}
\label{eq:link_constraint}
\sum_{task_i\in \Omega(t)}
\left(
y^l_{i,in}\frac{D^i_{in}}{\widehat T^i_{all}(t)}
+\sum_{(v^i_p,v^i_q)\in E_i}y^l_{i,p,q}\frac{D^i_{p,q}}{\widehat T^i_{all}(t)}
+y^l_{i,out}\frac{D^i_{out}}{\widehat T^i_{all}(t)}
\right)
\leq \hat B_l(t),\quad \forall l\in L.
\end{equation}
This approximation is used for admission control and load-state prediction; the reported inference delay is still calculated by the delay model in \eqref{eq:total_delay}.

\noindent
\textbf{Delay constraint}: The total estimated delay of the newly accepted DNN inference task should satisfy its delay requirement:
\begin{equation}
\label{eq:delay_constraint}
T^i_{all}(t)\leq T^i_{limit}.
\end{equation}

\noindent
\textbf{Dynamic state transition constraint}: Once a DNN task is accepted, it occupies resources for $S_i$ time slots. During this period, the node load, link load, heat state, effective bandwidth and dynamic reliability are updated according to \eqref{eq:node_load}--\eqref{eq:dynamic_link_r}. After the task is completed, its occupied CPU and link load contribution are released.

\subsubsection{Objective Function}
\noindent
The deployment problem jointly considers four objectives: service reliability, inference accuracy, inference delay and total energy consumption. The objective of service reliability maximization is
\begin{equation}
\label{eq:obj_service}
\max_{x,y} R'_i(t)=\max_{x,y}\left(\prod_{n\in N_i^x}R_n(t)\prod_{l\in L_i^x}Q_l(t)\right).
\end{equation}
The objective of inference accuracy maximization is
\begin{equation}
\label{eq:obj_accuracy}
\max_{x} A'_i(t)=\max_{x}\left(\prod_{n\in N_i^x}A_n(t)^{F^i_n/F^i_{all}}\right).
\end{equation}
The objective of inference delay minimization is
\begin{equation}
\label{eq:obj_delay_min}
\min_{x,y}T^i_{all}(t).
\end{equation}
The objective of total energy consumption minimization is
\begin{equation}
\label{eq:obj_energy_min}
\min_{x,y}E^i_{all}.
\end{equation}

Since the evolutionary search evaluates candidate deployments in a maximization-oriented form, inference delay is further transformed into a delay utility so that feasible solutions with smaller delay have larger objective values, while delay-violating solutions are penalized:
\begin{equation}
\label{eq:delay_utility}
U^i_T=
\begin{cases}
\frac{1}{T^i_{all}(t)}, & T^i_{all}(t)\leq T^i_{limit},\\
T^i_{limit}-T^i_{all}(t), & T^i_{all}(t)>T^i_{limit}.
\end{cases}
\end{equation}
Similarly, total energy consumption is transformed into an energy utility:
\begin{equation}
\label{eq:energy_utility_obj}
U^i_E=\frac{1}{E^i_{all}}.
\end{equation}

In summary, the DNN inference deployment problem can be expressed as the following multi-objective optimization problem:
\begin{equation}
\label{eq:multi_obj}
\left\{\max_{x,y}R'_i(t),\max_{x}A'_i(t),\min_{x,y}T^i_{all}(t),\min_{x,y}E^i_{all}\right\},
\end{equation}
subject to \eqref{eq:single_deploy}--\eqref{eq:delay_constraint} and the dynamic state transition equations. In the algorithmic implementation, $R'_i(t)$, $A'_i(t)$ and $U^i_T$ are used for Pareto dominance, while $U^i_E$ is used for candidate scoring, resource-aware variation, local refinement and final preference-based selection. This formulation enables the deployment algorithm to select nodes and paths according to the current cloud-edge-end system state, avoid overloaded and overheated resources, and balance service reliability, inference accuracy, inference delay and total energy consumption for online DNN inference requests.

\section{Algorithm Design}
\noindent
The DNN inference deployment problem formulated in the previous section can be regarded as a variant of the virtual network embedding (VNE) problem, because each DNN inference request needs to map its DAG-structured virtual layers and dependency transmissions onto a physical cloud-edge-end substrate network. Since the VNE problem has been proven to be NP-hard \cite{ref50}, the considered problem is also computationally intractable for exhaustive search. In addition, the deployment decision involves time-varying node reliability, link reliability, effective bandwidth, dynamic resource occupation and energy consumption. These factors make the search space large and make the evaluation of a deployment strategy dependent on the current system state. Therefore, it is impractical to obtain the optimal solution by traversing all possible deployment schemes.

To solve this problem, we design a DAG-Aware Reliable and Energy-Efficient Evolutionary Deployment algorithm, named DAREED. The algorithm follows a two-stage framework composed of evolutionary candidate search and preference-based deployment selection. In the first stage, DAREED searches for a high-quality Pareto candidate set by considering service reliability, inference accuracy and inference delay. In the second stage, DAREED selects the final deployment scheme from the candidate set by incorporating task preferences and total energy consumption. To fit the DAG structure of DNN inference and the dynamic characteristics of cloud-edge-end networks, DAREED incorporates DAG-aware evolutionary operators, resource-aware search guidance, constraint handling and elite refinement. After a DNN request is accepted, it is registered as a running block and continuously occupies resources for several time slots. The system state is then updated before the next request is processed.

\subsection{DAREED Framework}
\noindent
The overall algorithm is driven by online DNN request arrivals. At each time slot, the system first observes the current node load, link load, heat state, dynamic reliability and effective bandwidth. Then, for the newly arrived DNN request, the algorithm executes a customized evolutionary search to obtain candidate deployment schemes. If a feasible deployment scheme satisfying the delay requirement is found, the task is accepted and added to the running task set. Otherwise, the task is rejected. After each request decision, the system advances one time slot, updates resource occupation and refreshes the dynamic reliability of nodes and links.

For a DNN inference request $task_i$, a deployment individual is encoded as a one-dimensional chromosome:
\begin{equation}
\label{eq:chromosome}
X^m_i=\{x^m_{i,1},x^m_{i,2},\cdots,x^m_{i,|V_i|}\},
\end{equation}
where $x^m_{i,j}$ denotes the computing node selected for inference layer $v^i_j$ in individual $m$. This encoding is compact and directly corresponds to the layer-to-node deployment decision. The link mapping is obtained implicitly by the shortest path between the selected nodes of two dependent layers.

The fitness values of individual $m$ are evaluated by the dynamic model:
\begin{equation}
\label{eq:fitness_values}
F(X^m_i)=\left(A^m_i(t),R^m_i(t),U^m_T,U^m_E\right),
\end{equation}
where $A^m_i(t)$ is the inference accuracy, $R^m_i(t)$ is the service reliability, $U^m_T$ is the maximization-oriented utility transformed from inference delay, and $U^m_E$ is the maximization-oriented utility transformed from total energy consumption. In DAREED, the first three fitness components are mainly used for Pareto ranking, while $U^m_E$ is further used in candidate scoring, mutation guidance, local search and final deployment selection.

\subsection{DAG Context Construction}
\noindent
Before the evolutionary search starts, the algorithm extracts the topological information of the current DNN request. For $G_i=(V_i,E_i)$, the following auxiliary information is constructed: topological order, predecessor set, successor set, dependency data size, critical path task set and candidate node set. The topological order is used to initialize and repair deployment schemes according to dependency relations. The critical path is estimated by accumulating computational workload along the DAG and is used to guide crossover, mutation and local search.

For each layer $v^i_j$, the candidate node set is denoted by $\mathcal{C}_{i,j}$ and defined as
\begin{equation}
\label{eq:candidate_nodes}
\mathcal{C}_{i,j}=\{n\in N\mid C^n_{cpu}(t)\geq C'^{i,j}_{cpu},\ n\in N_{cloud}\cup N_{edge}\cup\{n^i_{initiate}\}\}.
\end{equation}
If no node satisfies the current CPU condition, the algorithm keeps the node-layer legality condition and relies on the repair and constraint violation mechanism to guide the search back to feasible regions. This design avoids premature search failure in heavily loaded dynamic scenarios.

\subsection{Evolutionary Candidate Search}
\noindent
The evolutionary candidate search stage of DAREED is responsible for generating a Pareto candidate set for a single DNN request under the current dynamic system state. This stage is built around six mechanisms tailored to DAG-structured inference tasks and dynamic cloud-edge-end resources.

\subsubsection{DAG-aware Population Initialization}
\noindent
The initial population is generated by combining several heuristic deployment modes instead of relying only on random sampling. Specifically, the population contains cloud-preferred individuals, edge-local individuals, resource-preferred individuals, reliability-preferred individuals, topology-aware individuals and random feasible individuals. The cloud-preferred mode deploys all layers to the cloud node to provide a high-reliability reference solution. The edge-local mode prioritizes nearby edge nodes to reduce wide-area transmission cost. The resource-preferred mode selects nodes with more available CPU resources. The reliability-preferred mode selects nodes with higher service reliability and inference accuracy under the current system state. The topology-aware mode selects nodes according to predecessor and successor communication cost. The random mode preserves population diversity.

For a candidate node $n$ of layer $v^i_j$, the topology-aware score is calculated as
\begin{equation}
\label{eq:candidate_score}
S_{i,j,n}=w_cS^c_n+w_tS^t_{i,j,n}+w_lS^l_{i,j,n}+w_rS^r_n+w_qS^q_{i,j,n}+w_hS^h_n+w_eS^e_{i,j,n}+w_fS^f_{i,j,n},
\end{equation}
where $S^c_n$ is the CPU score, $S^t_{i,j,n}$ is the execution time score, $S^l_{i,j,n}$ is the link delay score, $S^r_n$ is the node reliability score, $S^q_{i,j,n}$ is the path reliability score, $S^h_n$ is the heat score, $S^e_{i,j,n}$ is the energy score, and $S^f_{i,j,n}$ is the feasibility score. All scores are normalized to follow the rule that a larger value indicates a better candidate. Different role-based subpopulations use different weight vectors to emphasize reliability, latency, energy or feasibility.

\subsubsection{Role-based Subpopulation Evolution}
\noindent
To improve diversity and avoid concentrating the whole population on a single optimization direction, the population is divided into several role-based subpopulations, including reliability-oriented, latency-oriented, energy-oriented and feasibility-oriented subpopulations. In each generation, parent individuals are mainly selected from the same role subpopulation, while periodic cross-role parent selection is allowed to exchange useful genetic information among different roles. The reliability-oriented subpopulation tends to select nodes and paths with higher dynamic reliability, the latency-oriented subpopulation emphasizes computing speed and transmission delay, the energy-oriented subpopulation prefers low computation and transmission energy, and the feasibility-oriented subpopulation emphasizes sufficient residual resources and low overload risk.

\subsubsection{DAG Block Crossover}
\noindent
For a DAG-structured DNN, arbitrary chromosome segment exchange may break the dependency locality among strongly related layers. Therefore, DAREED adopts a DAG block crossover. The crossover boundary is selected from the critical path or a topological boundary. Given two parent individuals $X^a_i$ and $X^b_i$, the tasks after the selected boundary in topological order are exchanged:
\begin{equation}
\label{eq:dag_crossover}
\begin{cases}
\hat x^a_{i,j}=x^b_{i,j},\quad \hat x^b_{i,j}=x^a_{i,j}, & v^i_j\in \mathcal{B},\\
\hat x^a_{i,j}=x^a_{i,j},\quad \hat x^b_{i,j}=x^b_{i,j}, & v^i_j\notin \mathcal{B},
\end{cases}
\end{equation}
where $\mathcal{B}$ denotes the selected DAG block. After crossover, the offspring individuals are repaired to satisfy node legality and resource constraints as much as possible.

\subsubsection{Resource-aware Skew Mutation}
\noindent
Random mutation may move a layer to an overloaded, overheated or low-reliability node. To better exploit the current edge network state, DAREED uses a resource-aware skew mutation. The mutation operator first selects a layer, with higher probability assigned to critical path layers. Then, it samples a new node from the legal candidate set according to the candidate score in \eqref{eq:candidate_score}. Therefore, nodes with more residual CPU, lower execution time, lower communication cost, higher dynamic reliability, lower heat, lower energy cost and lower overload risk are more likely to be selected. This mutation keeps the exploration ability of genetic algorithms while making the search direction consistent with the dynamic deployment objective.

\subsubsection{Constraint-aware Repair and Constraint Violation}
\noindent
After crossover and mutation, an offspring may violate node legality, CPU resource, hierarchy or link overload constraints. The repair mechanism scans tasks in topological order. If a task is placed on an illegal end device or violates the hierarchy relation with its predecessors, the task is moved to the best legal candidate node. If CPU resources are still overloaded, the algorithm first migrates non-critical-path tasks, so that the delay-sensitive critical path is disturbed as little as possible.

For individuals that cannot be fully repaired, DAREED does not discard them immediately. Instead, a normalized constraint violation value is calculated:
\begin{equation}
\label{eq:constraint_violation}
CV(X^m_i)=CV_T(X^m_i)+CV_C(X^m_i)+CV_H(X^m_i)+CV_L(X^m_i),
\end{equation}
where $CV_T$ is the delay violation ratio, $CV_C$ is the CPU overload ratio, $CV_H$ is the hierarchy violation ratio, and $CV_L$ is the predicted link overload ratio. The delay violation ratio is defined as
\begin{equation}
\label{eq:delay_cv}
CV_T(X^m_i)=\frac{\max(0,T^m_i-T^i_{limit})}{T^i_{limit}}.
\end{equation}
The constraint tolerance $\epsilon_g$ decreases with the generation index $g$:
\begin{equation}
\label{eq:epsilon}
\epsilon_g=\epsilon_0\left(1-\frac{g}{G-1}\right),
\end{equation}
where $\epsilon_0$ is the initial tolerance and $G$ is the maximum number of generations. In early generations, slightly infeasible individuals are allowed to preserve diversity. In later generations, the algorithm gradually focuses on strictly feasible solutions.

\subsubsection{Constraint-dominance Sorting and Crowding Distance}
\noindent
The algorithm extends the non-dominated sorting rule with constraint violation. For two individuals $a$ and $b$, $a$ dominates $b$ under the current tolerance $\epsilon_g$ if one of the following conditions holds: i) $CV_a\leq\epsilon_g$ and $CV_b>\epsilon_g$; ii) both are infeasible and $CV_a<CV_b$; iii) both are feasible and $a$ is no worse in inference accuracy, service reliability and delay utility, and better in at least one of them. The feasible dominance relation can be expressed as
\begin{equation}
\label{eq:dominance}
\begin{cases}
A_a\geq A_b,\\
R_a\geq R_b,\\
U^a_T\geq U^b_T,\\
A_a>A_b\ \cup\ R_a>R_b\ \cup\ U^a_T>U^b_T.
\end{cases}
\end{equation}
After constraint-dominance sorting, the crowding distance is calculated within each Pareto front:
\begin{equation}
\label{eq:crowding}
Dis_m=\sum^{Z}_{z=1}\frac{f_z(m+1)-f_z(m-1)}{f^{max}_z-f^{min}_z},
\end{equation}
where $Z=3$ in the Pareto sorting stage and $f_z$ represents inference accuracy, service reliability or delay utility. The next-generation population is selected according to Pareto rank, constraint violation and crowding distance, thereby maintaining both feasibility and diversity.

\subsubsection{Elite Local Search}
\noindent
To further improve convergence quality, DAREED performs local search on a small number of elite individuals after each generation. The elite set is selected according to constraint violation and weighted score. For each elite individual, the algorithm first attempts to move critical path layers to their top-ranked candidate nodes. A neighboring solution is accepted if it does not increase constraint violation, is not worse in the three Pareto objectives, and achieves a higher weighted score. This local search enhances fine-grained adjustment around promising regions without significantly increasing the overall search cost.

\subsection{Preference-based Deployment Selection}
\noindent
The evolutionary candidate search stage outputs a set of non-dominated candidate deployment schemes. Since a specific DNN inference request needs one exact deployment scheme, DAREED further applies a preference-based selection strategy by considering task preferences and total energy consumption. The weights of inference accuracy, service reliability and inference delay are derived from the expected inference accuracy, expected service reliability and delay requirement of the task. Let $w_a$, $w_r$, $w_t$ and $w_e$ denote the final weights of inference accuracy, service reliability, inference delay and total energy consumption, respectively. In the adaptive mode, $w_e$ is set as a fixed energy preference and the remaining weight is distributed according to the normalized task expectations:
\begin{equation}
\label{eq:adaptive_weight}
w_a+w_r+w_t+w_e=1.
\end{equation}
For individual $m$, the final selection score is
\begin{equation}
\label{eq:final_score}
Score(X^m_i)=w_a\bar A^m_i+w_r\bar R^m_i+w_t\bar U^m_T+w_e\bar U^m_E,
\end{equation}
where $\bar A^m_i$, $\bar R^m_i$, $\bar U^m_T$ and $\bar U^m_E$ are the normalized inference accuracy, service reliability, inference-delay fitness and energy-consumption fitness, respectively. The individual with the largest score and positive delay utility is selected as the final deployment scheme. If no individual satisfies the delay requirement, the DNN request is rejected.

\subsection{Online Execution Procedure}
\noindent
The pseudocode of DAREED is shown in \autoref{alg:alg1}. To make the online decision process explicit, let $\Omega(t)$ denote the set of running DNN requests at time slot $t$, $\mathcal{A}$ denote the accepted deployment set, and $\mathcal{M}$ denote the performance metric set. The algorithm takes as input the dynamic physical network, the DNN request sequence, the population size $M$, and the maximum generation number $G_{\max}$. For each request, DAREED first generates a Pareto candidate set through constraint-aware evolutionary search, and then selects one feasible deployment by preference-based scoring.

\begin{algorithm}[htbp]
\caption{DAG-Aware Reliable and Energy-Efficient Evolutionary Deployment (DAREED)}\label{alg:alg1}
\begin{algorithmic}[1]
\STATE \textbf{Input}: substrate network $G=(N,L)$, request sequence $\mathcal{T}$, population size $M$, maximum generation $G_{\max}$
\STATE \textbf{Output}: accepted deployment set $\mathcal{A}$ and metric set $\mathcal{M}$
\STATE Initialize $\Omega(0)\gets\emptyset$, $\mathcal{A}\gets\emptyset$, and dynamic states $\Theta(0)$
\STATE $i\gets 1$, $t\gets 0$
\WHILE{$i\leq |\mathcal{T}|$ or $\Omega(t)\neq\emptyset$}
    \STATE $\Theta(t)\gets UpdateState(G,\Omega(t))$
    \IF{$i>|\mathcal{T}|$}
        \STATE $\Omega(t+1)\gets AdvanceRunningSet(\Omega(t))$
        \STATE $t\gets t+1$
        \STATE \textbf{continue}
    \ENDIF
    \STATE Let $task_i=(G_i,T^i_{limit},n^i_{initiate})$ and construct context $\mathcal{K}_i$
    \STATE $P_0\gets InitializePopulation(task_i,\mathcal{K}_i,\Theta(t),M)$
    \FOR{$g=0$ to $G_{\max}-1$}
        \STATE $\epsilon_g\gets \epsilon_0\left(1-g/(G_{\max}-1)\right)$
        \STATE $Q_g\gets Variation(P_g,\mathcal{K}_i,\Theta(t))$
        \STATE $\tilde Q_g\gets Repair(Q_g,\mathcal{K}_i,\Theta(t))$
        \STATE Evaluate $F(X)=\left(A_i(X,t),R_i(X,t),U_T(X,t),U_E(X,t)\right)$ and $CV(X)$ for all $X\in P_g\cup\tilde Q_g$
        \STATE $\mathcal{F}_g\gets ConstraintSort(P_g\cup\tilde Q_g,\epsilon_g)$
        \STATE $P_{g+1}\gets EnvironmentalSelect(\mathcal{F}_g,M)$
        \STATE $P_{g+1}\gets LocalRefine(P_{g+1},\mathcal{K}_i,\Theta(t))$
    \ENDFOR
    \STATE $\mathcal{P}_i\gets ND(P_{G_{\max}})$
    \STATE $\mathcal{P}^f_i\gets\{X\in\mathcal{P}_i\mid CV(X)=0,\ T_i(X,t)\leq T^i_{limit}\}$
    \IF{$\mathcal{P}^f_i\neq\emptyset$}
        \STATE $X_i^\ast\gets \arg\max_{X\in\mathcal{P}^f_i} Score(X)$
        \STATE $S_i\gets \lceil T_i(X_i^\ast,t)/\Delta\rceil$
        \STATE $\Omega(t)\gets \Omega(t)\cup\{(task_i,X_i^\ast,S_i)\}$
        \STATE $\mathcal{A}\gets \mathcal{A}\cup\{(task_i,X_i^\ast)\}$
    \ELSE
        \STATE Reject $task_i$ and record one failed deployment
    \ENDIF
    \STATE $\Omega(t+1)\gets AdvanceRunningSet(\Omega(t))$
    \STATE $i\gets i+1$, $t\gets t+1$
\ENDWHILE
\STATE $\mathcal{M}\gets AggregateMetrics(\mathcal{A})$
\STATE \textbf{return} $\mathcal{A}$ and $\mathcal{M}$
\end{algorithmic}
\end{algorithm}

In \autoref{alg:alg1}, $\Theta(t)$ contains the current CPU availability, node load, link load, heat state, effective bandwidth, and dynamic reliability values. $Variation(\cdot)$ includes role-based parent selection, DAG block crossover, and resource-aware skew mutation. $EnvironmentalSelect(\cdot)$ denotes constraint-dominance sorting followed by crowding-distance selection, and $ND(\cdot)$ returns the non-dominated candidate set. After a request is accepted, its estimated delay is converted into the number of occupied time slots. During these slots, the accepted request contributes to the node and link loads until its remaining running time becomes zero.

\subsection{Complexity}
\noindent
Assume that the population size is $M$, the maximum number of generations is $G_{\max}$, the number of layers in a DNN request is $v$, the number of physical nodes is $s$, and the number of physical links is $l$. DAG context construction needs to traverse DNN nodes and dependency edges, with complexity $O(v+|E_i|)$. DAG-aware initialization evaluates candidate nodes for each layer and each individual, with complexity $O(Mvs)$.

In each generation, role-based parent selection has complexity $O(M)$. DAG block crossover and skew mutation mainly operate on layer-level chromosomes and candidate nodes, with complexity $O(Mvs)$ in the worst case. Fitness evaluation requires calculating inference accuracy, service reliability, inference delay and total energy consumption. Considering shortest-path link lookup and dependency transmission evaluation, the complexity can be written as $O(M(v+|E_i|+l))$. Constraint-dominance sorting has complexity $O(M^2)$, and crowding distance calculation has complexity $O(M\log M)$ within each front. Elite local search is performed only on a small ratio $\theta$ of the population and tries at most $k$ candidate nodes for critical layers, with complexity $O(\theta M k v s)$ in the worst case.

Therefore, the overall time complexity of DAREED for one DNN request can be approximated as
\begin{equation}
\label{eq:complexity}
O\left(G_{\max}\cdot\left(M^2+Mvs+M(v+|E_i|+l)+\theta Mkvs\right)\right).
\end{equation}
Since $M^2$ and $Mvs$ are usually the dominant terms, the complexity can be simplified as $O(G_{\max}(M^2+Mvs))$ when the network scale is moderate and the elite local search ratio is small. The preference-based selection stage only scans the final population and calculates weighted scores, with complexity $O(M)$. The DAG-aware operators and local search reduce invalid exploration and improve the quality of the final deployment under dynamic cloud-edge-end conditions.

\section{Performance Analysis}
\noindent
In this section, we conduct extensive simulations to evaluate the effectiveness and efficiency of the proposed DAG-Aware Reliable and Energy-Efficient Evolutionary Deployment algorithm (DAREED). The experiments are designed to answer the following questions: whether DAREED can produce high-quality Pareto candidate sets, whether it improves dynamic reliability and energy efficiency under online DNN arrivals, whether it remains robust under different dynamic degradation levels, and how each customized evolutionary component contributes to the final performance.

\subsection{Simulation Environment Setup}
\noindent
All simulations are conducted in a cloud-edge-end collaborative inference environment. Following common MEC-based DNN inference simulation settings \cite{ref9}, the default topology contains one cloud node, ten edge nodes, and multiple end devices connected to edge nodes. The number of end devices is set according to the topology configuration, and the default setting uses approximately three end devices per edge node. The edge nodes are connected by an edge network, while each edge node is also connected to the cloud. DNN inference requests are generated online and are initiated by end devices.

The transmission rates between edge nodes are randomly selected within $[100,150]$ Mbps. The transmission rates between edge nodes and the cloud are randomly selected within $[1,10]$ Mbps, and the transmission rates between end devices and edge nodes are within $[20,50]$ Mbps. The CPU capacity of end devices is randomly selected within $[8,15]$, and their computing rates are randomly selected within $[1,4]$ GFLOPS. Edge nodes have CPU capacities within $[16,31]$ and computing rates within $[16,23]$ GFLOPS. The cloud node is configured with abundant computing resources and a computing rate of 28 GFLOPS.

For dynamic service quality, the initial service reliability and inference accuracy of end devices are randomly selected within $[0.90,0.95]$, while those of edge nodes are randomly selected within $[0.94,0.99]$. The cloud node has service reliability and inference accuracy of 0.99. During online execution, node and link reliability dynamically change according to load and heat states. The default slot length is 100 ms. The DNN requests contain 8 to 19 subtasks. Each subtask requires 1 to 2 CPU units, and its computational workload is randomly generated. The transmission data amount on each DNN dependency edge is randomly generated within the configured range. The delay requirement of each DNN request is randomly selected within $[500,1999]$ ms.

In the energy model, the computing power coefficients of cloud, edge and end nodes are set to 18, 10 and 4, respectively. The unit transmission energy coefficients of cloud-edge, edge-edge and end-edge links are set according to link types. Each experiment is repeated multiple times under different random seeds, and the mean and standard deviation are reported.

\begin{table}[!t]
\caption{Experimental Parameter Setting\label{tab:exp_setting}}
\centering
\begin{tabular}{p{5.2cm}p{2.5cm}}
\hline
Parameter & Setting\\
\hline
Number of cloud nodes & 1\\
Number of edge nodes & 10\\
Number of end devices & 30 by default\\
Edge-edge bandwidth & $[100,150]$ Mbps\\
Edge-cloud bandwidth & $[1,10]$ Mbps\\
End-edge bandwidth & $[20,50]$ Mbps\\
CPU capacity of edge nodes & $[16,31]$\\
Computing rate of edge nodes & $[16,23]$ GFLOPS\\
CPU capacity of end devices & $[8,15]$\\
Computing rate of end devices & $[1,4]$ GFLOPS\\
Cloud computing rate & 28 GFLOPS\\
Service reliability/inference accuracy of edge nodes & $[0.94,0.99]$\\
Service reliability/inference accuracy of end devices & $[0.90,0.95]$\\
Service reliability/inference accuracy of cloud & 0.99\\
Number of subtasks per DNN & $[8,19]$\\
DNN delay requirement & $[500,1999]$ ms\\
Slot length & 100 ms\\
Default population size & 60\\
Default iteration number & 200\\
Default mutation probability & 0.25\\
\hline
\end{tabular}
\end{table}

\subsection{Metrics and Comparison Methods}
\noindent
The following metrics are used to evaluate different algorithms.

\textbf{Average service reliability}: The average service reliability of successfully accepted DNN inference requests, considering both the dynamic reliability of computing nodes and the reliability of traversed links.

\textbf{Average inference accuracy}: The average inference accuracy of successfully accepted DNN inference requests, calculated according to the workload distribution over dynamically changing computing nodes.

\textbf{Average estimated inference delay}: The average estimated completion delay of successfully accepted DNN inference requests, including input uploading, critical-path processing, inter-layer transmission and result returning delay.

\textbf{Average total energy consumption}: The average total energy consumption of successfully accepted DNN inference requests, including computation energy and transmission energy.

\textbf{Rejected or failed deployments}: The number of DNN requests that cannot be accepted due to resource, delay or search failure.

\textbf{Average running time}: The wall-clock time consumed by an algorithm in each simulation run.

\textbf{Hypervolume}: Hypervolume is used to evaluate Pareto front quality. For each DNN request, the feasible non-dominated candidate set is normalized in the three-dimensional objective space composed of service reliability, inference accuracy and delay satisfaction. The reference point is set to $(0,0,0)$, and the final hypervolume value is averaged over DNN requests. A larger hypervolume indicates a Pareto front with better convergence and diversity.

The comparison algorithms are listed in \autoref{tab:comparison_algorithms}. For Pareto front quality evaluation, DAREED is compared with a generic NSGA-II based evolutionary deployment algorithm (NSGA-ED), random deployment algorithm (RDA), and simulated annealing algorithm (SAA). For final online deployment performance, DAREED is compared with NSGA-ED, RDA, local-first deployment (LPD), maximum-resource deployment (MRD), and RTBL \cite{ref9}.

\begin{table}[!t]
\caption{Comparison Algorithms\label{tab:comparison_algorithms}}
\centering
\begin{tabular}{p{1.6cm}p{1.6cm}p{4.4cm}}
\hline
Stage & Algorithm & Strategy\\
\hline
Pareto front
& DAREED & The proposed DAG-aware reliable and energy-efficient evolutionary deployment algorithm.\\
& NSGA-ED & A generic NSGA-II based evolutionary deployment algorithm without the DAG-aware customized operators.\\
& RDA & Randomly generates feasible deployment schemes and extracts the non-dominated set.\\
& SAA & Uses simulated annealing under multiple weight profiles to generate candidate deployment schemes.\\
\hline
Online deployment
& DAREED & Selects the final deployment by preference-based scoring from the evolutionary candidate set.\\
& NSGA-ED & Selects the final deployment from a generic evolutionary candidate set.\\
& RTBL & Reliability-aware task scheduling based on Lyapunov optimization and bandit learning.\\
& RDA & Randomly selects deployment nodes while satisfying basic constraints.\\
& MRD & Prioritizes computing nodes with more remaining CPU resources.\\
& LPD & Prioritizes local or nearby nodes to reduce transmission delay.\\
\hline
\end{tabular}
\end{table}

\subsection{Simulation Result Analysis}
\noindent

\subsubsection{Parameter Sensitivity}
\noindent
We first evaluate the influence of genetic search parameters on DAREED. Three groups of sensitivity experiments are conducted by varying population size, iteration number and mutation probability, respectively. The default DNN request number is 30. When one parameter is varied, the remaining parameters are fixed to their default values.

\autoref{fig_param_population} shows the performance of DAREED under different population sizes. The population size is selected from 20, 40, 60, 80 and 100. The expected output of this experiment is a line plot with average service reliability, average inference accuracy, average inference delay, average total energy consumption, rejected request number and running time. The concrete data will be filled after the final simulation results are obtained. According to the intended analysis, a larger population is expected to improve search diversity and Pareto front coverage, but it also increases running time.

\begin{figure}[!t]
\centering
\includegraphics[width=2.6in]{fig_param_population.pdf}
\caption{Sensitivity analysis under different population sizes.}
\label{fig_param_population}
\end{figure}

\autoref{fig_param_iteration} shows the influence of iteration number. The iteration number is selected from 80, 120, 160, 200, 240 and 280. This figure should report the tradeoff between search quality and computation cost. The expected conclusion is that the improvement becomes marginal after a certain number of iterations, while the running time continues to increase.

\begin{figure}[!t]
\centering
\includegraphics[width=2.6in]{fig_param_iteration.pdf}
\caption{Sensitivity analysis under different iteration numbers.}
\label{fig_param_iteration}
\end{figure}

\autoref{fig_param_mutation} shows the influence of mutation probability. The mutation probability is selected from 0.10, 0.15, 0.20, 0.25, 0.30 and 0.35. This experiment is used to choose an appropriate exploration intensity. A too small mutation probability may limit exploration, while a too large mutation probability may weaken convergence stability.

\begin{figure}[!t]
\centering
\includegraphics[width=2.6in]{fig_param_mutation.pdf}
\caption{Sensitivity analysis under different mutation probabilities.}
\label{fig_param_mutation}
\end{figure}

Based on these sensitivity experiments, the default population size, iteration number and mutation probability are set to 60, 200 and 0.25, respectively. The final selected parameters and the corresponding representative results are summarized in \autoref{tab:parameter_selection}.

\begin{table}[!t]
\caption{Parameter Selection Results\label{tab:parameter_selection}}
\centering
\begin{tabular}{p{2.6cm}p{1.5cm}p{3.4cm}}
\hline
Parameter & Selected value & Reason\\
\hline
Population size & 60 & TODO: fill selected tradeoff among service reliability, inference accuracy, inference delay, total energy consumption and runtime.\\
Iteration number & 200 & TODO: fill convergence observation.\\
Mutation probability & 0.25 & TODO: fill mutation sensitivity conclusion.\\
\hline
\end{tabular}
\end{table}

\subsubsection{Pareto Front Quality}
\noindent
The Pareto front quality is evaluated using DAREED, NSGA-ED, RDA and SAA. Pareto dominance is performed over inference accuracy, service reliability, delay utility and total energy consumption. Hypervolume is calculated in the corresponding normalized four-dimensional space of inference accuracy, service reliability, delay satisfaction and energy satisfaction.

\autoref{fig_pareto_3d} compares the nondominated solution sets produced by all algorithms for one matched representative repeat and DNN request. Inference accuracy, service reliability, and delay satisfaction form the three spatial axes, while point color encodes energy satisfaction. Black outlines identify members of the joint four-objective reference front constructed from the union of all algorithms' solution sets. The algorithm panels prevent heterogeneous requests and repeated runs from being mistaken for one Pareto front.

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{fig_pareto_3d.pdf}
\caption{Energy-encoded three-dimensional projections of four-objective nondominated solution sets for a median-size matched case. Black outlines denote the joint reference front.}
\label{fig_pareto_3d}
\end{figure}

To provide a quantitative comparison, \autoref{tab:pareto_hv} reports the average hypervolume, the average number of Pareto points per DNN request, and the standard deviation over repeated runs. The concrete values are left as placeholders before the final simulations are completed.

\begin{table}[!t]
\caption{Pareto Front Quality Comparison\label{tab:pareto_hv}}
\centering
\begin{tabular}{p{1.7cm}p{1.8cm}p{1.8cm}p{1.5cm}}
\hline
Algorithm & Avg. HV & Avg. Pareto points & Std. HV\\
\hline
DAREED & TODO & TODO & TODO\\
NSGA-ED & TODO & TODO & TODO\\
SAA & TODO & TODO & TODO\\
RDA & TODO & TODO & TODO\\
\hline
\end{tabular}
\end{table}

\subsubsection{Overall Performance under Different Numbers of DNN Requests}
\noindent
We then evaluate the online deployment performance under different numbers of DNN requests. The number of DNN requests is set to 5, 10, 20, 30 and 40. This experiment evaluates how different algorithms behave as the online inference load increases.

\autoref{fig_overall_reliability} shows the average service reliability and average inference accuracy. The expected output is a two-subfigure line plot. DAREED is expected to maintain better service quality because it avoids overloaded and overheated nodes and links by using dynamic reliability-aware candidate scoring and constraint-aware evolution.

\begin{figure}[!t]
\centering
\includegraphics[width=2.6in]{fig_overall_reliability.pdf}
\caption{Average service reliability and inference accuracy under different numbers of DNN requests.}
\label{fig_overall_reliability}
\end{figure}

\autoref{fig_overall_delay_energy} shows the average inference delay and average total energy consumption. This figure is intended to demonstrate the tradeoff among inference delay, service reliability, inference accuracy and total energy consumption. LPD may achieve competitive delay in some cases due to local deployment preference, while DAREED is expected to achieve a better overall balance between inference delay and total energy consumption.

\begin{figure}[!t]
\centering
\includegraphics[width=2.6in]{fig_overall_delay_energy.pdf}
\caption{Average inference delay and total energy consumption under different numbers of DNN requests.}
\label{fig_overall_delay_energy}
\end{figure}

\autoref{fig_overall_failure_runtime} shows the rejected or failed deployment number and the running time. The expected output is a two-subfigure line plot. The final analysis should report whether DAREED reduces failed deployments under heavier loads and whether its running time remains acceptable for online deployment.

\begin{figure}[!t]
\centering
\includegraphics[width=2.6in]{fig_overall_failure_runtime.pdf}
\caption{Failed deployments and running time under different numbers of DNN requests.}
\label{fig_overall_failure_runtime}
\end{figure}

\subsubsection{Cloud Constraint Stress Test}
\noindent
To examine whether the performance of DAREED depends on an idealized cloud setting, we further design a stress test with three cloud-side configurations. The default configuration follows the basic setup in \autoref{tab:exp_setting}. The constrained-compute configuration reduces the available CPU capacity of the cloud node. The degraded-reliability configuration lowers the initial service reliability and inference accuracy of the cloud node. The narrow-backhaul configuration reduces the edge-cloud bandwidth range. These settings are used to evaluate whether DAREED can adaptively shift deployment decisions toward edge and end resources when cloud-side conditions become less favorable.

\autoref{fig_cloud_stress} shows the expected comparison under the three cloud-side stress settings. The final results should report the changes in service reliability, inference accuracy, inference delay, total energy consumption and failed deployments. This experiment is intended to demonstrate that DAREED does not rely solely on the cloud node for reliability improvement, but selects deployment schemes according to the joint state of cloud, edge, end nodes and communication links.

\begin{figure}[!t]
\centering
\includegraphics[width=2.6in]{fig_cloud_stress.pdf}
\caption{Performance under cloud-side stress settings.}
\label{fig_cloud_stress}
\end{figure}

\begin{table}[!t]
\caption{Cloud Constraint Stress Test Results\label{tab:cloud_stress}}
\centering
\begin{tabular}{p{2.0cm}p{1.3cm}p{1.3cm}p{1.3cm}p{1.2cm}}
\hline
Setting & Service rel. & Infer. acc. & Delay & Energy\\
\hline
Default & TODO & TODO & TODO & TODO\\
Constrained compute & TODO & TODO & TODO & TODO\\
Degraded reliability & TODO & TODO & TODO & TODO\\
Narrow backhaul & TODO & TODO & TODO & TODO\\
\hline
\end{tabular}
\end{table}

\subsubsection{Dynamic Reliability Degradation}
\noindent
To evaluate robustness under dynamic system degradation, we construct three scenarios: weak, medium and strong. In the weak scenario, the reliability degradation coefficients are small and heat decays quickly. In the medium scenario, the default dynamic parameters are used. In the strong scenario, node and link reliability are more sensitive to load and heat, and heat decays more slowly.

\autoref{fig_dynamic_degradation} shows the performance of different algorithms under the three degradation levels. The expected output is a grouped bar chart covering service reliability, inference accuracy, average total energy consumption and failed deployment number. DAREED is expected to be more stable under the strong degradation scenario, because its mutation, repair and local search are guided by load, heat and dynamic reliability.

\begin{figure}[!t]
\centering
\includegraphics[width=2.6in]{fig_dynamic_degradation.pdf}
\caption{Performance under different dynamic reliability degradation levels.}
\label{fig_dynamic_degradation}
\end{figure}

\begin{table}[!t]
\caption{Dynamic Degradation Results\label{tab:dynamic_results}}
\centering
\begin{tabular}{p{1.5cm}p{1.5cm}p{1.3cm}p{1.3cm}p{1.3cm}p{1.2cm}}
\hline
Scenario & Algorithm & Service rel. & Infer. acc. & Energy & Failed\\
\hline
Weak & DAREED & TODO & TODO & TODO & TODO\\
Weak & Best baseline & TODO & TODO & TODO & TODO\\
Medium & DAREED & TODO & TODO & TODO & TODO\\
Medium & Best baseline & TODO & TODO & TODO & TODO\\
Strong & DAREED & TODO & TODO & TODO & TODO\\
Strong & Best baseline & TODO & TODO & TODO & TODO\\
\hline
\end{tabular}
\end{table}

\subsubsection{Ablation Study}
\noindent
To verify the contribution of each customized component in DAREED, we conduct an ablation study. The complete DAREED algorithm is compared with five variants: w/o DAG-aware initialization, w/o DAG block crossover, w/o skew mutation, w/o elite local search, and a plain evolutionary variant that disables all four customized components.

\autoref{fig_ablation} shows the ablation results. The expected output is a grouped bar chart. If the DAG-aware initialization is removed, the initial population may contain more low-quality candidates. If DAG block crossover is removed, offspring may lose dependency locality. If skew mutation is removed, the mutation operator cannot exploit the current dynamic resource state. If elite local search is removed, the final convergence quality may decline.

\begin{figure}[!t]
\centering
\includegraphics[width=2.6in]{fig_ablation.pdf}
\caption{Ablation study of DAREED components.}
\label{fig_ablation}
\end{figure}

\begin{table}[!t]
\caption{Ablation Study Results\label{tab:ablation}}
\centering
\begin{tabular}{p{2.7cm}p{1.3cm}p{1.3cm}p{1.3cm}p{1.2cm}}
\hline
Variant & Service rel. & Infer. acc. & Energy & Failed\\
\hline
Full DAREED & TODO & TODO & TODO & TODO\\
w/o DAG init. & TODO & TODO & TODO & TODO\\
w/o block crossover & TODO & TODO & TODO & TODO\\
w/o skew mutation & TODO & TODO & TODO & TODO\\
w/o elite local search & TODO & TODO & TODO & TODO\\
Plain evolutionary & TODO & TODO & TODO & TODO\\
\hline
\end{tabular}
\end{table}

\subsubsection{Large-scale Network Evaluation}
\noindent
We further evaluate scalability under three network scales. The small scenario contains 1 cloud node, 10 edge nodes and 30 end devices. The medium scenario contains 1 cloud node, 30 edge nodes and 120 end devices. The large scenario contains 1 cloud node, 60 edge nodes and 300 end devices. The number of DNN requests is scaled accordingly.

\autoref{fig_scale} shows the performance under different network scales. This experiment is used to verify whether DAREED can exploit richer edge resources while keeping the running time acceptable. The final discussion should report the growth trend of running time and whether service reliability, inference accuracy, inference delay and total energy consumption remain stable as the topology grows.

\begin{figure}[!t]
\centering
\includegraphics[width=2.6in]{fig_scale.pdf}
\caption{Scalability evaluation under different network scales.}
\label{fig_scale}
\end{figure}

\begin{table}[!t]
\caption{Large-scale Network Results\label{tab:scale}}
\centering
\begin{tabular}{p{1.4cm}p{1.7cm}p{1.4cm}p{1.4cm}p{1.2cm}p{1.2cm}}
\hline
Scale & Algorithm & Delay & Energy & Failed & Runtime\\
\hline
Small & DAREED & TODO & TODO & TODO & TODO\\
Medium & DAREED & TODO & TODO & TODO & TODO\\
Large & DAREED & TODO & TODO & TODO & TODO\\
\hline
\end{tabular}
\end{table}

\subsubsection{Application Preference Scenarios}
\noindent
Finally, we evaluate whether DAREED can adapt to different application preferences. Four preferences are considered: delay-sensitive, reliability-sensitive, energy-sensitive and balanced. These preferences affect the final deployment selection stage by changing the weights assigned to inference delay, service reliability, inference accuracy and total energy consumption.

\autoref{fig_preference} shows the resulting tradeoffs. In the delay-sensitive case, DAREED is expected to select schemes with lower inference delay. In the reliability-sensitive case, it is expected to select schemes with higher service reliability and inference accuracy. In the energy-sensitive case, it is expected to reduce total energy consumption. In the balanced case, the selected deployments are expected to provide stable overall performance.

\begin{figure}[!t]
\centering
\includegraphics[width=2.6in]{fig_preference.pdf}
\caption{Performance under different application preferences.}
\label{fig_preference}
\end{figure}

\begin{table}[!t]
\caption{Application Preference Results\label{tab:preference}}
\centering
\begin{tabular}{p{2.0cm}p{1.4cm}p{1.4cm}p{1.4cm}p{1.2cm}}
\hline
Preference & Delay & Service rel. & Infer. acc. & Energy\\
\hline
Delay-sensitive & TODO & TODO & TODO & TODO\\
Reliability-sensitive & TODO & TODO & TODO & TODO\\
Energy-sensitive & TODO & TODO & TODO & TODO\\
Balanced & TODO & TODO & TODO & TODO\\
\hline
\end{tabular}
\end{table}

\subsection{Summary}
\noindent
The complete experimental evaluation is expected to demonstrate that DAREED can generate high-quality Pareto candidate sets and achieve a favorable balance among service reliability, inference accuracy, inference delay and total energy consumption. The final manuscript should fill the placeholders in the figures, tables and textual conclusions after the complete experimental results are obtained.

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
J. Xu, H. Yao, R. Zhang, T. Mai, and M. Guizani, ``Low latency and accuracy-guaranteed DNN inference for UAV-assisted IoT networks,'' \textit{IEEE Transactions on Cognitive Communications and Networking}, early access, 2025, doi: 10.1109/TCCN.2025.3542443.

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
Z. Zhang, Y. Zhao, M.-C. Chang, C. Lin, and J. Liu, ``E4: Energy-efficient DNN inference for edge video analytics via early-exit and DVFS,'' arXiv preprint arXiv:2503.04865, 2025.

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
M. Deutel, G. Kontes, et al., ``Multi-objective Bayesian optimization with reinforcement learning for edge deployment of DNNs on microcontrollers,'' in \textit{Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO '25 Companion)}, ACM, New York, NY, USA, 2025, pp. 19--20.

\bibitem{ref30}
Y. Du, S. Xu, and Y. Yu, ``Joint scheduling of sensing data offloading and edge inference for multi-UAV networks,'' arXiv preprint arXiv:2605.03898, 2026.

\bibitem{ref31}
J. Cao, X. Li, Q. Liu, T. Han, N. Zhang, and W. Shi, ``EdgeServing: Deadline-aware multi-DNN serving at the edge,'' arXiv preprint arXiv:2605.05527, 2026.

\bibitem{ref32}
J. Zhou, X. Hou, Y. Zeng, et al., ``Quality of experience and reliability-aware task offloading and scheduling for multi-user mobile-edge computing systems,'' \textit{IEEE Transactions on Services Computing}, early access, 2025.

\bibitem{ref33}
A. M. Rasouli, M. Esnaashari, and M. Ansari, ``RASOUL: A reliability-aware task allocation strategy to improve success rate and energy saving in mobile-edge computing,'' \textit{IEEE Internet of Things Journal}, early access, 2025.

\bibitem{ref34}
H. Hao, C. Xu, W. Zhang, et al., ``Reliability-aware optimization of task offloading for UAV-assisted edge computing,'' \textit{IEEE Transactions on Computers}, early access, 2025.

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
