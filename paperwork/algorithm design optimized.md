```latex
\section{Algorithm Design}
\noindent
The DNN inference deployment problem formulated in the previous section can be regarded as a variant of the virtual network embedding (VNE) problem, because each DNN inference request needs to map its DAG-structured virtual layers and dependency transmissions onto a physical cloud-edge-end substrate network. Since the VNE problem has been proven to be NP-hard \cite{ref50}, the considered problem is also computationally intractable for exhaustive search. In addition, the deployment decision involves time-varying node reliability, link reliability, effective bandwidth, dynamic resource occupation and energy consumption. These factors make the search space large and make the evaluation of a deployment strategy dependent on the current system state. Therefore, it is impractical to obtain the optimal solution by traversing all possible deployment schemes.

To solve this problem, we design a DAG-Aware Reliable and Energy-Efficient Evolutionary Deployment algorithm, named DAREED. The algorithm follows a two-stage framework composed of evolutionary candidate search and preference-based deployment selection. In the first stage, DAREED performs a four-objective evolutionary search that jointly considers predicted service reliability, predicted inference accuracy, inference delay and total energy consumption. Four-objective constraint-dominance and crowding-distance selection guide population evolution. In the second stage, DAREED applies request-specific preference weights to the strictly feasible candidates maintained throughout the search and selects one deployment. Pareto-front export and hypervolume evaluation are enabled only in the dedicated Pareto experiment and are not part of ordinary online deployment. To fit the DAG structure of DNN inference and the dynamic characteristics of cloud-edge-end networks, DAREED incorporates DAG-aware evolutionary operators, resource-aware search guidance, constraint handling and elite refinement. After a DNN request is accepted, it is registered as a running block and continuously occupies resources for several time slots. The system state is then updated before the next request is processed.

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
F(X^m_i)=
\left(
\hat A^m_i(t\mid X^m_i),
\hat R^m_i(t\mid X^m_i),
U^m_T,
E^m_i
\right),
\end{equation}
where $\hat A^m_i(t\mid X^m_i)$ and $\hat R^m_i(t\mid X^m_i)$ are the one-step post-admission inference accuracy and service reliability predicted for candidate $X^m_i$, $U^m_T$ is the maximization-oriented delay utility, and $E^m_i$ is the original total energy consumption. The first three components are maximized and $E^m_i$ is minimized. All four components participate in Pareto ranking, crowding distance, local search and final preference-based scoring after normalization.

\subsection{One-Step Post-Admission State Prediction}
\noindent
Evaluating a candidate only with the pre-deployment state may overestimate the reliability of nodes and links that become heavily loaded after admission. DAREED therefore performs a one-step, non-mutating prediction for each candidate deployment. Let $\Delta C_n(X)$ denote the CPU demand added to node $n$ by candidate $X$. The predicted node load and heat are
\begin{equation}
\label{eq:predicted_node_load}
\hat\rho_n(t\mid X)=
\rho_n(t)+\frac{\Delta C_n(X)}{\bar C^n_{cpu}},
\end{equation}
\begin{equation}
\label{eq:predicted_node_heat}
\hat h_n(t\mid X)=
\lambda_h h_n(t)+(1-\lambda_h)\hat\rho_n(t\mid X).
\end{equation}
For link $l$, let $\Delta D_l(X)$ be the candidate traffic carried by the link and $\hat T_i(X,t)$ be the estimated candidate runtime. Its predicted load and heat are
\begin{equation}
\label{eq:predicted_link_load}
\hat\mu_l(t\mid X)=
\mu_l(t)+
\frac{\Delta D_l(X)}
{\hat T_i(X,t)\hat B_l(t)},
\end{equation}
\begin{equation}
\label{eq:predicted_link_heat}
\hat g_l(t\mid X)=
\lambda_g g_l(t)+(1-\lambda_g)\hat\mu_l(t\mid X).
\end{equation}
The predicted node and link reliability values are then obtained by substituting $\hat\rho_n$, $\hat h_n$, $\hat\mu_l$ and $\hat g_l$ into the dynamic degradation model. These values are used to evaluate $\hat R'_i(t\mid X)$ and $\hat A'_i(t\mid X)$ without modifying the actual environment state. The real state is updated only after a candidate has been accepted.

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
The evolutionary candidate search stage of DAREED is responsible for generating four-objective Pareto ranks for a single DNN request under the current dynamic system state. This stage is built around seven mechanisms tailored to DAG-structured inference tasks and dynamic cloud-edge-end resources.

\subsubsection{DAG-aware Population Initialization}
\noindent
The initial population is generated from cloud-preferred, edge-local, resource-preferred, reliability-preferred and topology-aware templates, followed by template perturbation and random feasible sampling. Candidate chromosomes are repaired and deduplicated during construction. DAREED attempts to make at least 75\% of the initial population unique; if this target cannot be reached within a bounded retry budget, feasible individuals are duplicated only to complete the required population size. This strategy preserves heuristic coverage without allowing a small number of templates to dominate the initial population. The cloud-preferred mode deploys all layers to the cloud node to provide a high-reliability reference solution. The edge-local mode prioritizes nearby edge nodes to reduce wide-area transmission cost. The resource-preferred mode selects nodes with more available CPU resources. The reliability-preferred mode selects nodes with higher service reliability and inference accuracy under the current system state. The topology-aware mode selects nodes according to predecessor and successor communication cost.

For a candidate node $n$ of layer $v^i_j$, the topology-aware score is calculated as
\begin{equation}
\label{eq:candidate_score}
S_{i,j,n}=w_cS^c_n+w_tS^t_{i,j,n}+w_lS^l_{i,j,n}+w_rS^r_n+w_qS^q_{i,j,n}+w_hS^h_n+w_eS^e_{i,j,n}+w_fS^f_{i,j,n},
\end{equation}
where $S^c_n$ is the CPU score, $S^t_{i,j,n}$ is the execution time score, $S^l_{i,j,n}$ is the link delay score, $S^r_n$ is the node reliability score, $S^q_{i,j,n}$ is the path reliability score, $S^h_n$ is the heat score, $S^e_{i,j,n}$ is the energy score, and $S^f_{i,j,n}$ is the feasibility score. All scores are normalized to follow the rule that a larger value indicates a better candidate. Different role-based subpopulations use different weight vectors to emphasize reliability, latency, energy or feasibility.

\subsubsection{Role-based Subpopulation Evolution}
\noindent
To improve diversity and avoid concentrating the whole population on a single optimization direction, each individual is assigned a persistent role label: reliability, latency, energy or feasibility. Role-specific candidate-score weights guide variation, and environmental selection preserves an approximately equal quota for every role. Parents are normally sampled from the same role, while every five generations one parent may be sampled globally to realize cross-role migration. This label-and-quota implementation maintains specialization without allocating four additional full populations. The reliability role tends to select nodes and paths with higher dynamic reliability, the latency role emphasizes computing speed and transmission delay, the energy role prefers low computation and transmission energy, and the feasibility role emphasizes sufficient residual resources and low overload risk.

\subsubsection{DAG Block Crossover}
\noindent
For a DAG-structured DNN, arbitrary chromosome segment exchange may break the dependency locality among strongly related layers. Therefore, DAREED adopts a successor-closed DAG block crossover. For an anchor layer $v_a$, the crossover block is
\begin{equation}
\label{eq:successor_closed_block}
\mathcal{B}(v_a)=\{v_a\}\cup \operatorname{Desc}(v_a),
\end{equation}
such that every successor of a layer in $\mathcal{B}$ also belongs to $\mathcal{B}$. Candidate blocks are size-controlled, with a preferred upper bound of $\lceil0.4|V_i|\rceil$; if no block satisfies the preferred range, the smallest available proper successor closure, possibly a singleton, is used. Given two parent individuals $X^a_i$ and $X^b_i$, the assignments in the selected block are exchanged with probability $p_c$:
\begin{equation}
\label{eq:dag_crossover}
\begin{cases}
\hat x^a_{i,j}=x^b_{i,j},\quad \hat x^b_{i,j}=x^a_{i,j}, & v^i_j\in \mathcal{B},\\
\hat x^a_{i,j}=x^a_{i,j},\quad \hat x^b_{i,j}=x^b_{i,j}, & v^i_j\notin \mathcal{B},
\end{cases}
\end{equation}
where $\mathcal{B}$ denotes the selected successor-closed DAG block. The offspring are accepted only when structural repair succeeds and the number of genes changed by repair does not exceed $\max(1,\lfloor|\mathcal{B}|/2\rfloor)$; otherwise, the corresponding parent is retained.

\subsubsection{Resource-aware Skew Mutation}
\noindent
Random mutation may move a layer to an overloaded, overheated or low-reliability node. To better exploit the current edge network state, DAREED uses a resource-aware skew mutation. With probability $p_m$, the operator selects a layer, assigning half of the selection probability to critical-path layers and half to all layers. The current node is excluded from the candidate set. Candidate scores are min-max normalized within the selected layer and converted into probabilities by a numerically stable softmax:
\begin{equation}
\label{eq:skew_softmax}
P(n\mid v^i_j)=
\frac{\exp((\tilde S_{i,j,n}-\tilde S_{\max})/\tau)}
{\sum_{q\in\mathcal{C}_{i,j}\setminus\{x_{i,j}\}}
\exp((\tilde S_{i,j,q}-\tilde S_{\max})/\tau)}.
\end{equation}
Therefore, nodes with more residual CPU, lower execution time, lower communication cost, higher dynamic reliability, lower heat, lower energy cost and lower overload risk are more likely to be selected. The mutated chromosome is retained only if structural repair succeeds; otherwise, the parent assignment is restored. Crossover is independently activated with probability $p_c$.

\subsubsection{Constraint-aware Repair and Constraint Violation}
\noindent
After crossover and mutation, an offspring may violate node legality, CPU resource, hierarchy or link overload constraints. The repair mechanism scans tasks in topological order. If a task is placed on an illegal end device or violates the hierarchy relation with its predecessors, the task is moved to the best legal candidate node. If CPU resources are still overloaded, the algorithm first migrates non-critical-path tasks, so that the delay-sensitive critical path is disturbed as little as possible.

Repair explicitly returns the repaired chromosome, a success flag, the number of changed genes and a failure reason. If crossover or mutation produces a chromosome that cannot be structurally repaired, that offspring falls back to its parent and is not silently inserted as a repaired solution. Structurally valid individuals may still have nonzero delay, CPU, hierarchy or predicted-link-overload violation. These residual violations are quantified by the following normalized constraint violation value:
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
where $\epsilon_0$ is the initial tolerance and $G$ is the maximum number of generations. In early generations, slightly infeasible individuals are allowed to preserve diversity. In later generations, the algorithm gradually focuses on strictly feasible solutions. Thus, structural repair failure and temporary constraint infeasibility are treated separately.

\subsubsection{Constraint-dominance Sorting and Crowding Distance}
\noindent
The algorithm extends the non-dominated sorting rule with constraint violation. For two individuals $a$ and $b$, $a$ dominates $b$ under the current tolerance $\epsilon_g$ if one of the following conditions holds: i) $CV_a\leq\epsilon_g$ and $CV_b>\epsilon_g$; ii) both are infeasible and $CV_a<CV_b$; iii) both are feasible and $a$ is no worse in inference accuracy, service reliability and delay utility, has no larger total energy consumption, and is strictly better in at least one of the four objectives. The feasible dominance relation can be expressed as
\begin{equation}
\label{eq:dominance}
\begin{cases}
A_a\geq A_b,\\
R_a\geq R_b,\\
U^a_T\geq U^b_T,\\
E_a\leq E_b,\\
A_a>A_b\ \cup\ R_a>R_b\ \cup\ U^a_T>U^b_T\ \cup\ E_a<E_b.
\end{cases}
\end{equation}
After constraint-dominance sorting, the crowding distance is calculated within each Pareto front:
\begin{equation}
\label{eq:crowding}
Dis_m=\sum^{Z}_{z=1}\frac{f_z(m+1)-f_z(m-1)}{f^{max}_z-f^{min}_z},
\end{equation}
where $Z=4$ and the four dimensions are inference accuracy, service reliability, delay utility and total energy consumption. The energy dimension is sorted in its original minimization direction, while the distance magnitude is calculated from neighboring objective values as in the other dimensions. The next-generation population is selected according to Pareto rank, constraint violation, crowding distance and role quota, thereby maintaining feasibility, diversity and role specialization.

\subsubsection{Elite Local Search}
\noindent
To further improve convergence quality, DAREED performs local search on a small number of elite individuals after each generation. The elite set is selected according to constraint violation and weighted score. For each elite individual, the algorithm first attempts to move critical path layers to their top-ranked candidate nodes. A neighboring solution is accepted only if it does not increase constraint violation, is not worse in all four Pareto objectives, including no increase in total energy, and achieves a higher normalized preference score. This local search enhances fine-grained adjustment around promising regions without significantly increasing the overall search cost.

\subsection{Preference-based Deployment Selection}
\noindent
The evolutionary search uses four-objective non-dominated fronts for environmental selection and maintains the preference-best strictly feasible candidate across generations. Since a specific DNN inference request needs one exact deployment scheme, DAREED applies a preference-based strategy to choose one candidate. The weights of inference accuracy, service reliability and inference delay are derived from the expected inference accuracy, expected service reliability and delay requirement of the task. Let $w_a$, $w_r$, $w_t$ and $w_e$ denote the final weights of inference accuracy, service reliability, inference delay and total energy consumption, respectively. In the adaptive mode, $w_e$ is set as a fixed energy preference and the remaining weight is distributed according to the normalized task expectations:
\begin{equation}
\label{eq:adaptive_weight}
w_a+w_r+w_t+w_e=1.
\end{equation}
For individual $m$, the final selection score is
\begin{equation}
\label{eq:final_score}
Score(X^m_i)=w_a\bar A^m_i+w_r\bar R^m_i+w_t\bar U^m_T+w_e\bar E^m_i,
\end{equation}
where $\bar A^m_i$, $\bar R^m_i$ and $\bar U^m_T$ are the normalized inference accuracy, service reliability and inference-delay fitness, respectively, and
\begin{equation}
\label{eq:normalized_energy_score}
\bar E^m_i=
\operatorname{clip}\left(
\frac{E^i_{\max}-E^m_i}{E^i_{\max}-E^i_{\min}},0,1
\right)
\end{equation}
is the normalized energy satisfaction. During evolution, DAREED maintains the highest-scoring strictly feasible candidate found across generations. After the final generation, it also examines the strictly feasible final population. Before admission, the selected chromosome is re-evaluated and accepted only if $CV(X)\leq10^{-12}$ and $U_T(X)>0$. Otherwise, the request is rejected.

\subsection{Online Execution Procedure}
\noindent
The pseudocode of DAREED is shown in \autoref{alg:alg1}. To make the online decision process explicit, let $\Omega(t)$ denote the set of running DNN requests at time slot $t$, $\mathcal{A}$ denote the accepted deployment set, and $\mathcal{M}$ denote the performance metric set. The algorithm takes as input the dynamic physical network, the DNN request sequence, the population size $M$, and the maximum generation number $G_{\max}$. For each request, DAREED performs constraint-aware four-objective evolution, maintains a strictly feasible preference-best candidate across generations, and revalidates that candidate before admission.

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
    \STATE $X_i^{best}\gets\varnothing$, $s_i^{best}\gets-\infty$
    \FOR{$g=0$ to $G_{\max}-1$}
        \STATE $\epsilon_g\gets \epsilon_0\left(1-g/(G_{\max}-1)\right)$
        \STATE $Q_g\gets Variation(P_g,\mathcal{K}_i,\Theta(t))$
        \STATE $\tilde Q_g\gets Repair(Q_g,\mathcal{K}_i,\Theta(t))$
        \STATE Predict post-admission states for all $X\in P_g\cup\tilde Q_g$
        \STATE Evaluate $F(X)=\left(\hat A_i(X,t),\hat R_i(X,t),U_T(X,t),E_i(X)\right)$ and $CV(X)$
        \STATE Update $(X_i^{best},s_i^{best})$ using candidates satisfying $CV(X)=0$ and $U_T(X)>0$
        \STATE $\mathcal{F}_g\gets ConstraintSort_{4D}(P_g\cup\tilde Q_g,\epsilon_g)$
        \STATE $P_{g+1}\gets RoleQuotaSelect(\mathcal{F}_g,M)$
        \STATE $P_{g+1}\gets LocalRefine_{4D}(P_{g+1},\mathcal{K}_i,\Theta(t))$
    \ENDFOR
    \STATE Re-evaluate $P_{G_{\max}}$ and update $X_i^{best}$ with strictly feasible candidates
    \IF{$X_i^{best}\neq\varnothing$ and $RecheckCV(X_i^{best})=0$}
        \STATE $X_i^\ast\gets X_i^{best}$
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

In \autoref{alg:alg1}, $\Theta(t)$ contains the current CPU availability, node load, link load, heat state, effective bandwidth and dynamic reliability values. $Variation(\cdot)$ includes role-based parent selection, successor-closed DAG block crossover and resource-aware skew mutation. $RoleQuotaSelect(\cdot)$ denotes four-objective constraint-dominance sorting followed by crowding-distance and role-quota selection. In ordinary online experiments, the four-objective ranks are used internally, but Pareto points are not exported. A feasible non-dominated archive is collected only when the experiment suite is explicitly configured for Pareto-front evaluation. After a request is accepted, its estimated delay is converted into the number of occupied time slots. During these slots, the accepted request contributes to the node and link loads until its remaining running time becomes zero.

\subsection{Complexity}
\noindent
Assume that the population size is $M$, the maximum number of generations is $G_{\max}$, the number of layers in a DNN request is $v$, the number of physical nodes is $s$, and the number of physical links is $l$. DAG context construction needs to traverse DNN nodes and dependency edges, with complexity $O(v+|E_i|)$. DAG-aware initialization evaluates candidate nodes for each layer and each individual, with complexity $O(Mvs)$.

In each generation, role-based parent selection has complexity $O(M)$. DAG block crossover and skew mutation mainly operate on layer-level chromosomes and candidate nodes, with complexity $O(Mvs)$ in the worst case. Fitness evaluation calculates inference delay and total energy together with one-step post-admission node/link load, heat, service-reliability and accuracy prediction. With cached physical paths, this evaluation remains $O(M(v+|E_i|+l))$. Adding the fourth objective changes only the constant factor of dominance and crowding-distance operations; their asymptotic complexities remain $O(M^2)$ and $O(M\log M)$, respectively. Elite local search is performed only on a small ratio $\theta$ of the population and tries at most $k$ candidate nodes for critical layers, with complexity $O(\theta M k v s)$ in the worst case.

Therefore, the overall time complexity of DAREED for one DNN request can be approximated as
\begin{equation}
\label{eq:complexity}
O\left(G_{\max}\cdot\left(M^2+Mvs+M(v+|E_i|+l)+\theta Mkvs\right)\right).
\end{equation}
Since $M^2$ and $Mvs$ are usually the dominant terms, the complexity can be simplified as $O(G_{\max}(M^2+Mvs))$ when the network scale is moderate and the elite local search ratio is small. The preference-based selection stage only scans the evaluated candidates and calculates weighted scores, with complexity $O(M)$ per update. Pareto-point export and four-dimensional hypervolume are disabled in ordinary experiments. They are executed only as postprocessing in the Pareto experiment and are excluded from the reported algorithm runtime. The DAG-aware operators and local search reduce invalid exploration and improve the quality of the final deployment under dynamic cloud-edge-end conditions.
```
