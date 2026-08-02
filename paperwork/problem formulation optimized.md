```latex
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

$E^i_{\min}$, $E^i_{\max}$, $\bar E^i$ & Request-specific energy reference bounds and normalized energy satisfaction\\
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
In the evolutionary search, the original total energy consumption $E^i_{all}$ is retained as a minimization objective rather than being compressed by a reciprocal transformation. For preference scoring and Pareto-front normalization only, it is mapped to an energy-satisfaction value:
\begin{equation}
\label{eq:energy_satisfaction}
\bar E^i=
\operatorname{clip}\left(
\frac{E^i_{\max}-E^i_{all}}
{E^i_{\max}-E^i_{\min}},0,1
\right),
\end{equation}
where $E^i_{\min}$ and $E^i_{\max}$ are deterministic request-specific reference bounds derived from the minimum and maximum computation energy over legal nodes and an upper bound on path transmission energy. A larger $\bar E^i$ indicates lower energy consumption, while Pareto dominance is still evaluated directly using $E^i_{all}$.

\subsection{MILP Formulation}
\noindent
Herein, the mathematical formulation of the DNN inference task deployment problem is proposed. The decision variable $x^n_{i,j}$ represents the deployment of inference layers on computing nodes, while $y^l_{i,p,q}$ represents the mapping of dependency transmissions onto physical links.

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
\textbf{Link load constraint}: The bandwidth occupied by all running DNN tasks on each link should not exceed the current effective bandwidth:
\begin{equation}
\label{eq:link_constraint}
\sum_{task_i\in \Omega(t)}
\left(
y^l_{i,in}\frac{D^i_{in}}{T^i_{all}(t)}
+\sum_{(v^i_p,v^i_q)\in E_i}y^l_{i,p,q}\frac{D^i_{p,q}}{T^i_{all}(t)}
+y^l_{i,out}\frac{D^i_{out}}{T^i_{all}(t)}
\right)
\leq \hat B_l(t),\quad \forall l\in L.
\end{equation}

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

Inference delay is transformed into a maximization-oriented delay utility so that feasible solutions with smaller delay have larger objective values, while delay-violating solutions are penalized:
\begin{equation}
\label{eq:delay_utility}
U^i_T=
\begin{cases}
\frac{1}{T^i_{all}(t)}, & T^i_{all}(t)\leq T^i_{limit},\\
T^i_{limit}-T^i_{all}(t), & T^i_{all}(t)>T^i_{limit}.
\end{cases}
\end{equation}

In contrast, total energy consumption is retained in its original form because it is directly minimized. Accordingly, the objective vector used by the evolutionary search is
\begin{equation}
\label{eq:algorithm_objective_vector}
F_i(X,t)=
\left(\hat A'_i(X,t),\hat R'_i(X,t),U^i_T(X,t),E^i_{all}(X)\right),
\end{equation}
where the first three components are maximized and the fourth component is minimized. The hatted reliability quantities denote the predicted post-admission values defined in the algorithm section.

In summary, the DNN inference deployment problem can be expressed as the following multi-objective optimization problem:
\begin{equation}
\label{eq:multi_obj}
\left\{\max_{x,y}R'_i(t),\max_{x}A'_i(t),\min_{x,y}T^i_{all}(t),\min_{x,y}E^i_{all}\right\},
\end{equation}
subject to \eqref{eq:single_deploy}--\eqref{eq:delay_constraint} and the dynamic state transition equations. In the implementation, $U^i_T$ represents feasible delay quality, whereas $E^i_{all}$ directly participates in four-objective dominance, crowding distance, local refinement and preference-based selection. This formulation enables the deployment algorithm to select nodes and paths according to the current cloud-edge-end system state, avoid overloaded and overheated resources, and balance service reliability, inference accuracy, inference delay and total energy consumption for online DNN inference requests.
```
