```latex
\section{Problem Formulation}
\noindent
In this section, the reliability and delay-oriented DNN inference task deployment problem is described and modeled. All the notations used in this paper and their meanings are listed in \autoref{tab:table1}.
\begin{table}[!t]  
\caption{Parameter symbols\label{tab:table1}}  
\centering  
\begin{tabular}{>{\centering\arraybackslash}m{3cm}p{5cm}}  
\hline  
Notation & Meaning\\
\hline  
$G=(N,L)$ & Network topology\\
  
$N=N_{end}\cup N_{edge}\cup N_{cloud}$ & $N_{end}, N_{edge}, N_{cloud}$ are end device node-set, edge computing node set and cloud computing node-set, respectively\\
  
$L$ & Link set, $\forall l = (m,n)\in L\  m,n\in N\   and\  m\neq n$\\
  
$C^{n}_{cpu}$ & CPU resource on node $n$\\
  
$A_{n}$ & Inference accuracy reliability of computing node $n$\\
  
$R_{n}$ & Service reliability of computing node $n$\\
  
$\eta_{n}$ & The computing power of unit computing resources $n$, in the number of floating-point operations per second\\
  
$B_{l}$ & Transmission rate of link $l$, $\forall l=(m,n)\in L$ \\
  
$Task$ & DNN inference task set\\
  
$task_i$ & A DNN inference $task_i$\\
  
$G_i=(V_i,E_i)$ & DAG of DNN inference $task_i$ \\
  
$V_i$ & The set of inference layers of $task_i$\\
  
$E_i$ & The set of dependency layers of $task_i$, $\forall e^i_j=(v^i_p,v^i_q)\in E_i,v^i_p,v^i_q\in V_i \ and \ v^i_p \neq v^i_q$\\
  
$C'^{i,j}_{cpu}$ & The CPU resources required for the inference layer $v^i_j$  of $task_i$\\
  
$R'_i$ & Service reliability of $task_i$\\
  
$A'_i$ & Accuracy reliability of $task_i$\\
  
$F_{i,j}$ & The computational workload required for inference layer $v^i_j$ of $task_i$, calculated based on the total floating-point operations\\
  
$T^i_{all}$ & The total execution delay of $task_i$\\
  
$t^i_{tran}$ & Transmission delay of $task_i$\\
  
$t^i_{proc}$ & Processing delay of $task_i$\\
  
$T^i_{limit}$ & Delay requirements for $task_i$\\
  
$F'^i_{e^i_j}$ & The amount of data transmission required for dependency layer $e^i_j$ of $task_i$, $e^i_j\in E_i$\\
  
$x^n_{i,j}$ & If inference layer $v^i_j$ of $task_i$ is deployed on computing node $n$, it is 1; otherwise, it is 0\\
  
$y^l_{i,e^i_j}$ & If  dependency layer $e^i_j$ of $task_i$ is mapped on the link $l$, it is 1; otherwise, it is 0 \\
  
$n^i_{initiate}$ & The initiating node of $task_i$ \\
  
$d^n_{i,j}$ & The processing delay of inference layer $v^i_j$ of $task_i$ on computing node $n$\\
  
$d'^i_{e^i_j}$ & The transmission delay of dependency layer $e^i_j$ of $task_i$\\
  
$R^{esp}_{i}$ & Expected service reliability for  $task_i$ \\
  
$A^{esp}_{i}$ & Expected accuracy reliability for  $task_i$ \\
\hline  
\end{tabular}  
\end{table}  
\subsection{Resource Model}
\noindent
\subsubsection{Substrate Network}
\noindent
This paper focuses on network scenarios encompassing central cloud, edge computing nodes, and end devices. We denote the network structure as $G=(N, L)$. Where $N$ represents a collection of computing nodes, including end devices, edge computing nodes, and central cloud, such that $N=N_{end} \cup N_{edge} \cup N_{cloud}$. $L$ denotes the set of interconnected links between computing nodes, where each link $l \in L$ corresponds to a node pair $(m, n)$, with $m, n$ being nodes in $N$. For each link $l$, the symbol $B_l$ represents the transmission rate. Each computing node is equipped with specific computing resources $C^n_{cpu}$, quantified in CPU cores. $\eta_{n}$ denotes the computing power of unit computing resources $n$.
\subsubsection{DNN Inference Task Model}
\noindent
A DNN inference task request is initiated by an end device. Let $Task$ denote the set of DNN inference tasks. An individual DNN inference task $task_i$ within this set has a specific inference delay requirement denoted by $T^i_{limit}$. Each DNN inference task can be modeled as a directed acyclic graph $G_i=(V_i, E_i)$. 
This graph has topological characteristics, that is, the execution of each subtask depends on the implementation of the previous subtask, and the correct execution of the entire DNN inference task depends on the completion of all subtasks. 
$V_i$ represents the inference layer set of $task_i$, while $E_i$ represents the dependency set between layers. For each $v^i_j \in V_i,\ j \in [1,|V_i|] $, its deployment on computing nodes requires the utilization of CPU resources, denoted as $C^{'i,j}_{cpu}$, $F_{i,j}$ represents the computational workload required by the inference layer $v^i_j$ of DNN inference $task_i$.
Considering the need for data transmission between DNN inference layers, we use $F^{'i}_{e^i_j}$ to represent the amount of data transmission on dependency layer $e^i_j$.
\subsection{Delay model}
\noindent
The delay of DNN inference tasks generally comprises processing delay, queuing delay, propagation delay, and transmission delay. Among these, propagation delay and queuing delay are peripheral to our focus, typically small and thus negligible. Therefore, this paper assumes that the total delay of DNN inference tasks equals the sum of processing delay and transmission delay. Each DNN inference task $task_i$ has multiple paths, and we refer to the path with the longest delay as the critical path. The set of inference layers on the critical path is denoted as $V^{cri}_i$, and the set of dependency layers is denoted as $E_i^{cri}$. The delay of the critical path is the total inference delay of the DNN inference task.
The processing delay $d^n_{i,j}$ of inference layer $v^i_j$ of $task_i$ on computing node n is illustrated as

\begin{equation}
\label{eq1}
d^n_{i,j} = \frac{F_{i,j}}{\eta_n*C^{'i,j}_{cpu}}.
\end{equation}
Therefore, the total processing delay of the inference $task_i$ is depicted as
\begin{equation}
\label{eq2}
t^i_{proc} = \sum^{|N|}_{n=1}\sum^{|V_i^{cri}|}_{j=1}x^n_{i,j}*d^n_{i,j},
\end{equation}
where $x^n_{i,j}$ represents whether inference layer $v^i_j$ of $task_i$ is deployed on computing node $n$.\par
Using $F'^i_{e^i_j}$ to represent the amount of data that dependency layer $e^i_j$ needs to transmit, the transmission delay of DNN inference dependency layer $e^i_j$ is illustrated as

\begin{equation}
\label{eq3}
d'^i_{e^i_j} = \sum^{|L|}_{l=1}y^l_{i,e^i_j}\frac{F'^i_{e^i_j}}{B_l}.
\end{equation}
Here, $y^l_{i,e^i_j}$ represents whether the dependency layer $e^i_j$ of $task_i$ is mapped on link $l$.\par
Consequently, the total transmission delay of the inference $task_i$ is depicted as
\begin{equation}
\label{eq4}
t^i_{tran} = \sum^{|E_i^{cri}|}_{j=1}d'^i_{e^i_j}.
\end{equation}
According to the above equation, the total execution delay of DNN inference $task_i$ is given as
\begin{equation}
\label{eq5}
T^i_{all}=t^i_{proc}+t^i_{tran}.
\end{equation}
\subsection{Reliability Model}
\noindent
The reliability of DNN inference tasks refers to the probability of successfully and accurately executing these tasks. This paper focuses on the scheduling reliability of DNN inference tasks \cite{ref9}, which can be categorized into service reliability and accuracy reliability. Research \cite{ref37} indicates that edge server failures are more prevalent than cloud server failures. In the MEC context, a computing node failure forces the termination of services it hosts, significantly impacting the Quality of Service (QoS). Additionally, due to the limited computing capacity of mobile devices and edge nodes, executing DNN inference tasks on them typically yields lower accuracy. Moreover, even DNN inference tasks executed in data centers might be preempted by higher-priority tasks \cite{ref38}, leading to diminished accuracy and ultimately unreliable inference results. These aspects collectively contribute to the potential inability or inaccurate of DNN inference tasks, resulting in unreliable task execution \cite{ref39}.\par
In this paper, we assume that the failures of network nodes hosting DNN inference tasks are independent. The service reliability of a computing node refers to the probability of that the computing node hosting a DNN inference task will provide services normally. We use $R_n$ to represent the service reliability of computing node $n$. The service reliability is highest for the central cloud, followed by edge computing nodes, and finally end devices. While service reliability is typically a dynamic variable, for simplification, we assume that the service reliability of computing nodes remains constant during the execution of DNN inference tasks. Therefore, the service reliability $R'_i$ of DNN inference $task_i$ is expressed as
\begin{equation}
\label{eq6}
R'_i=\prod^{|N|}_{n=1}\prod^{|V_i|}_{j=1}x^n_{i,j}*R_n.
\end{equation}
For DNN inference $task_i$, we use $R_i^{des}$ to represent the expected service reliability as a reference value for deployment strategy selection.\par
The accuracy reliability of DNN inference tasks hinges on the accuracy with which the underlying computing nodes compute the results of these tasks. This metric evaluates the consistency and precision of the inference results of DNN models under varying input conditions. Instability or unreliability in the model’s inferences can potentially lead to erroneous decisions or outputs, thus impacting the overall application’s performance and reliability. Therefore, accuracy reliability is also a crucial metric for assessing the reliability of DNN inference tasks. It is not difficult to find that this reliability requires not only the provision of uninterrupted services by the computing nodes but also the delivery of accurate computational results. This article uses $A_n$ to represent the inference accuracy reliability of the computing node. Equation (7) depicts the accuracy reliability of DNN inference tasks as follows
\begin{equation}
\label{eq7}
A'_i=\prod^{|N|}_{n=1}\prod^{|V_i|}_{j=1}x^n_{i,j}*R_n*A_n.
\end{equation}
For DNN inference $task_i$, we use $A_i^{des}$ to represent the expected accuracy reliability as a reference value for deployment strategy selection.
\subsection{MILP Formulation}
\noindent
Herein, the mathematical formulation of the problem of reliability and delay-oriented DNN inference task deployment is proposed, starting with the proposal of the constraints.
\subsubsection{Constraint}
\noindent
\textbf{Single deployment constraint}: Let the initiating node of DNN inference $task_i$ be $n^i_{initiate}\in N$. This constraint restricts that each inference layer $v^i_j$ of DNN inference $task_i$ must be deployed on a computing node, and each inference layer can only be deployed on one computing node. In addition, the computing node can only be either the mobile device initiating the inference task, an edge computing node, or a central cloud computing node, which is given as
\begin{equation}
\begin{split}
\label{eq8}
&\sum^{|N|}_{n=1}x^n_{i,j}=1,i\in \{k \in \mathbb{Z}^+ \mid k \le |Task|\},\\ &\ j\in \{k \in \mathbb{Z}^+ \mid k \le |V_i|\}, n\in N_{cloud}\cup N_{edge}\cup \{n^i_{initiate}\}.
\end{split}
\end{equation}
\noindent
\textbf{Computing resource constraint}: This constraint restricts the total computation resources required by tasks deployed on computing node n must not exceed the maximum computing resources that the computing node can provide, which is given as
\begin{equation}
\label{eq9}
\sum^{|V|}_{i=1}\sum^{|V_i|}_{j=1}x^n_{i,j}*C'^{i,j}_{cpu}\leq C^n_{cpu}.
\end{equation}
\noindent
\textbf{Delay constraint}: This constraint restricts that the total delay of inference $task_i$ is less than the delay requirement of the task, which is given as
\begin{equation}
\label{eq10}
T^i_{all}\leq T^i_{limit}.
\end{equation}


\subsubsection{Objective Function}
\noindent   
The multi-objective optimization direction of this paper focuses on high scheduling reliability and low delay. Here, scheduling reliability encompasses service reliability and accuracy reliability.

\textbf{Service reliability}:
The objective function for maximizing service reliability is depicted as
\begin{equation}
\label{eq11}
\max_{x^n_{i,j}}{\prod^{|N|}_{n=1}\prod^{|V_i|}_{j=1}x^n_{i,j}*R_n}.
\end{equation}
\noindent
\textbf{Accuracy reliability}:
The objective function for maximizing accuracy reliability is depicted as
\begin{equation}
\label{eq12}
\max_{x^n_{i,j}}{\prod^{|N|}_{n=1}\prod^{|V_i|}_{j=1}x^n_{i,j}*R_n*A_n}.
\end{equation}
\noindent
\textbf{Total delay}:
This paper considers delay as both a constraint and an optimization objective. The objective function for minimizing total delay is presented as
\begin{equation}
\label{eq13}
\min_{x^n_{i,j},y^l_{i,e^i_j}}T^i_{all}=t^i_{proc}+t^i_{tran}.
\end{equation}
```

