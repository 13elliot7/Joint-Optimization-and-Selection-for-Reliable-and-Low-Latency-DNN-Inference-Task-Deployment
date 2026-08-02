```latex
\section{Algorithm Design}
\noindent
Viewing the deployment of DNN inference tasks as embedding the virtual networks of DNN inference tasks into a physical network, the deployment problem of DNN inference tasks we introduce can be seen as a variant of the virtual network embedding(VNE) problem. The VNE problem has been proven to be an NP-hard problem \cite{ref40}. Compared with the VNE problem, the constraints of the proposed DNN inference task deployment problem proposed are relatively relaxed, with only constraints on computing node (equation \eqref{eq9}) in terms of resources and constraints on the deployment of the DNN inference task (equation \eqref{eq8},\eqref{eq10}). This can be seen as a simplified form of VNE. However, due to the fact that our problem requires more optimization objectives (equation \eqref{eq11},\eqref{eq12},\eqref{eq13}) than the VNE problem with additional constraints, we infer that the multi-objective optimization deployment problem of the DNN inference task we are considering is also NP-hard. Therefore, this problem is also an NP-hard problem, making it impractical to straightforwardly search for solutions through traversal. \par
Consequently, we propose a multi-objective optimization algorithm based on NSGA-II, employing genetic evolution to find the Pareto solution set of deployment strategies and choose appropriate strategies from this set. NSGA-II operates on the principle of Pareto dominance, where a solution is considered a Pareto solution if it is better in at least one objective and not worse in the others. 
Our goal is to derive a Pareto solution set for DNN inference task deployment schemes concerning reliability and delay objectives. The Pareto solution set typically consists of multiple solutions. Therefore, it is necessary to seek a specific deployment plan for specific DNN inference tasks. We first seek the Pareto solution set of DNN inference task deployment schemes across multiple optimization objectives. Subsequently, considering the delay and reliability requirements of DNN inference tasks, a heuristic method was adopted to determine the most suitable deployment plan in the Pareto solution set. The overall framework of the algorithm is illustrated in \autoref{fig_2}.
\begin{figure}[htbp]  
\centering  
\includegraphics[width=2.5 in]{fig 2.jpg}  
\caption{THE Overall Framework of the Algorithm}  
\label{fig_2}  
\end{figure} 
\subsection{Reliability and delay-oriented DNN inference task deployment algorithm based on NSGA-II }
\noindent
The NSGA-II algorithm achieves its objectives by maintaining a set of non-dominated solutions. The algorithm generates new solutions for each generation through a combination of genetic operations. It sorts the population through non-dominated properties, selecting the best solution set for the next iteration. The RDODA algorithm, customized for edge computing, enhances NSGA-II performance by designing a specialized population initialization and tailored genetic operations for deployment across cloud, edge, and end nodes. Crucially, it employs a unique fitness function refinement to retain delay-violating solutions, preventing premature convergence and loss of diversity. This is combined with diversity-aware offspring selection to effectively maintain a high-quality, well-distributed Pareto front for multi-objective optimization. The specific steps for finding the Pareto solution set for DNN inference task deployment strategies are as follows:
\subsubsection{Population Initialization} 
\noindent
The algorithm generates an initial parent population for the DNN inference task deployment problem, considering single deployment and resource constraints outlined in equations \eqref{eq8} and \eqref{eq9}. Additionally, deployment schemes that allocate all DNN inference tasks to either the central cloud or end devices are also introduced to enhance population diversity.
\subsubsection{Genetic Operations}
\noindent
Genetic operations include selection, crossover, mutation, and merging. In the selection stage, individuals from the parent population are randomly chosen to promote diversity. The selected individuals then undergo crossover. To minimize effect of genetic manipulation, we use a two-point crossover method. After crossover, random mutation is applied to the offspring. Finally, the offspring population is merged with the parent population. These genetic operations adhere to resource and deployment constraints. 
\subsubsection{Fitness calculation and non-dominated sorting}
\noindent
Initially, the algorithm calculates the fitness of the parent and offspring populations in terms of service reliability, accuracy reliability, and DNN inference task execution delay. The calculation method is based on equation \eqref{eq5}, \eqref{eq6}, and \eqref{eq7}. Since individuals violating the delay constraint may exist, to ensure population diversity, the algorithm does not directly remove individuals that exceed the delay constraint. Instead, the fitness values of individuals exceeding the delay constraint in the delay direction are set to negative to ensure they are inferior in that direction to solutions complying with the delay constraint.  \par
Subsequently, the algorithm performs fast non-dominated sorting on each individual based on the computed fitness values. The steps are as follows: i) Each individual in the population obtained in the previous step is compared with the other solutions to determine their dominance status. Individual b is said to dominate individual a if the conditions in \eqref{eq14} are satisfied, which are given as
\begin{equation}
    \label{eq14}
    \begin{cases}
    R_a\leq R_b, \\
    A_a\leq A_b, \\
    T_a\leq T_b, \\
    R_a<R_b\cup A_a<A_b \cup T_a<T_b.
    \end{cases}
    \end{equation}
ii) Place individuals not dominated by other solutions in a two-dimensional sequence, where the first dimension represents the non-dominated level, and the second dimension corresponds to the respective non-dominated individual; iii) Repeat step 2 until all individuals are placed in the sequence.
    
    
\subsubsection{Crowding distance calculation} 
\noindent
The algorithm initially sorts each individual based on each non-dominated level. Then, the algorithm calculates the crowding degree of each solution based on the fitness value. At each non-dominated level, the algorithm first sets the population crowding degree at the edge to $\infty$, then calculates the crowding distance for each individual, which represents the average distance between two points on both sides of the individual. The calculation method is shown as
    \begin{equation}
    \label{eq15}
    Dis_i=\sum^Z_{z=1}\frac{f_z(i+1)-f_z(i-1)}{f^{max}_z-f^{min}_{z}}.
    \end{equation}
\noindent
Among them, $Z$ represents the number of functions and $f_z(i)$ represents the fitness value of function $z$ of individual $i$. $f^{max}_z $ and $f^{min}_z$ respectively represent the maximum and minimum values under the function z of the current non-dominated level. It can be seen that the sparser the surroundings of an individual, the greater the crowding distance.
    
\subsubsection{Selecting the next-generation population}
\noindent
The algorithm determines the next-generation population through non-dominated sorting and crowding distance screening. To enhance solution diversity, the selection process proceeds as follows: individuals are selected sequentially based on their non-dominated sorting levels. If the current population level is smaller than the expected size of the next generation, all individuals of the current level will be added to the next generation. Otherwise, individuals with higher crowding distances are prioritized and selected to preserve diversity if the current population exceeds the desired size.
\subsubsection{Iterative population} 
\noindent
Repeat steps (1) - (5) until the number of iterations is reached.\par
The pseudocode of reliability and delay-oriented DNN inference task deployment algorithm based on NSGA-II(RDODA) is shown in \autoref{alg:alg1}.
\begin{algorithm}[htbp]  
\caption{Reliability and Delay-Oriented DNN Inference Task Deployment Algorithm Based on NSGA-II (RDODA)}\label{alg:alg1}  
\begin{algorithmic}[1] 
\STATE \textbf{Input}: Physical network topology $G=(N, L)$, DNN inference task $G_i=(V_i, E_i)$  
\STATE \textbf{Output}: A Pareto solution set $X$ for DNN inference tasks, and the fitness value set $VALUES$  
\STATE \textbf{Initial value}: Population size: popSize; Number of iterations: gen  
\STATE $i \gets 0$ \\

\WHILE{$i < popSize$}  
    \STATE Randomly generate population $M$, and the corresponding solution for individual $m$ in this population $x^m$  
    \IF{\textbf{checkResource(m)} $\&\&$ \textbf{checkPlacement(m)}}  
        \STATE $i \gets i + 1$  
    \ENDIF  
\ENDWHILE  
\STATE $i \gets 0$  
\WHILE{$i < gen$}  
    \STATE \textbf{q $\gets$ crossAndMutate()} \% Cross mutation operation to obtain subpopulations  
    \STATE \textbf{res $\gets$ q + M} \% Merge father and son populations  
    \STATE \textbf{values $\gets$ countValues(res)} \% Calculate the fitness values of the parent-child population in various directions  
    \STATE \textbf{dominatedSort(res)} \% Calculate the non-dominated sorting of the population  
    \STATE \textbf{crowdingDistance(res)} \% Calculate the crowding distance of the population  
    \STATE \textbf{M $\gets$ getNextRes(res)} \% Obtain the next generation population 
\ENDWHILE  

\FOR{m in M}  
    \IF{$checkDelay(m)$}  
        \STATE add $x^m$ to the $X$  
        \STATE add $countValues(m)$ to the $VALUES$  
    \ENDIF  
\ENDFOR  

\STATE \textbf{return} $X, VALUES$  
\end{algorithmic}  
\end{algorithm}  
\noindent
The algorithm takes as input the physical network and the DNN inference task to be deployed. It will output the Pareto solution set for the DNN inference task along with the fitness values corresponding to each optimization objective for each solution. The initial conditions include the population size(popSize) and the number of iterations(gen). Initially, the algorithm generates the initial population (Lines 4-10). Solutions are generated while adhering to deployment constraints (equation \eqref{eq3}) and resource constraints (equation \eqref{eq4}). Once a solution that satisfies these constraints is found, it is added to the population. Next, genetic algorithm operations including selection, crossover, mutation, and merging are executed on the initial population iteratively (Lines 11-19). Upon reaching the maximum number of iterations, select the Pareto solution set that satisfies the delay constraint from M to represent the deployment of DNN inference tasks(20-25). Return the Pareto solution set X and the VALUES set that encompasses the fitness values of each individual in the solution set(Line 26).
\subsection{Deployment scheme selection algorithm based on expectation}
\noindent
Algorithm 1 yields a Pareto solution set, offering a range of solutions without specifying a particular deployment plan. Therefore, to derive a deployment plan tailored for a given DNN inference task, we need to select the most suitable deployment from the Pareto solution set. To accomplish this, we introduce three expectation conditions: the expected service reliability $R^{des}_i$, the expected accuracy reliability $A^{des}_i$, and the expected delay of DNN inference $task_i$. In order to select an optimal solution suitable for a specific DNN inference task among various solutions, we quantify the weights of the task across three objective functions. We utilize the normalization formula of equation \eqref{eq16} to calculate the weights of service reliability and accuracy reliability, denoted as $R_w$ and $A_w$, respectively. Here, $I_w$ represents $R_w$ or $A_w$. When $I_w$ represents $R_w$, $I=R^{des}_i$, $I_{max}$ represents the maximum expected service reliability, and $I_{min}$ represents the minimum service reliability. When $I_w$ represents $A_w$, $I=A^{des}_{i}$, $I_{max}$ represents the maximum accuracy reliability, and $I_{min}$ represents the minimum accuracy reliability. Equation \eqref{eq17} is used to calculate the weight of delay, where expected delay T as a reference value is set as the delay constraint of the task, $T_{max}$ represents the maximum delay requirement among all the tasks, while $T_{min}$ represents the minimum delay requirement among all the tasks. In practical network scenarios, $T$, $R^{des}_i$, and $A^{des}_{i}$ can be obtained through the expected values of DNN inference tasks corresponding to their features, rather than through subjective settings, which helps to improve the flexibility of the method in different scenarios.
\begin{equation}
\label{eq16}
I_w=\frac{I-I_{min}}{I_{max}-I_{min}}
\end{equation}
\begin{equation}
\label{eq17}
T_w=\frac{T_{max}-T}{T_{max}-T_{min}}
\end{equation}
We let $I_i$ take $R_w,\ A_w$ and $T_w$ respectively, and use equation \eqref{eq18} to calculate the proportion $I'_i$ between the service reliability weight, accuracy reliability weight, and delay weight, where $I'_i$ corresponds to $I'_1,\ I'_2,\ I'_3$ respectively, that is, the proportion between the service reliability weight, accuracy reliability weight, and delay weight.
\begin{equation}
\label{eq18}
I'_i=\frac{I_i}{I_1+I_2+I_3},i\in {1,2,3}
\end{equation}
We assume that $P^m_1,\ P^m_2,\ P^m_3$ are the normalized results of the service reliability, accuracy reliability, and delay function values corresponding to individual $m$. Using equation \eqref{eq19}, an exact solution can be selected as the precise deployment scheme for a specific DNN inference task.
\begin{equation}
\label{eq19}
\max_{x^{m}}P_1I'_1+P_2I'_2+P_3I'_3
\end{equation}
The pseudo-code of the deployment scheme selection algorithm based on expectation(DSSA) is presented in \autoref{alg:alg2}. This algorithm takes as input the Pareto solution set and fitness value set generated by Algorithm 1. First, it uses equation \eqref{eq16}, \eqref{eq17} to calculate the relative weights of DNN inference tasks within the domains of service reliability, accuracy reliability, and delay(line 3). Next, it uses equation \eqref{eq18} to normalize these weight values and calculate the corresponding proportions(Line 4). Then, it traverses each solution in solution set X, calculates the objective function values corresponding to each solution set, and records the solution set with the maximum value as the optimal deployment strategy(Line6-Line13). The algorithm will finally return the deployment scheme $x^m$ as well as the corresponding accuracy reliability $A^m_i$, service reliability $R^m_i$, and delay $T^m_i$.
\begin{algorithm}[H]  
\caption{Deployment Scheme Selection Algorithm Based on Expectation (DSSA)}\label{alg:alg2}  
\begin{algorithmic}[1] % The argument [1] enables line numbering  
\STATE \textbf{Input}: The Pareto solution set $X$ and fitness value set obtained by Algorithm 1.  
\STATE \textbf{Output}: The deployment method $x^m$ of DNN inference $task_i$, Accurate reliability $A^m$, Service reliability $R^m$, Total delay $T^m$   

\STATE $I_1, \ I_2,\ I_3 \gets \textbf{calculateWeight}(task_i)$ \% Use equations (16-17) to calculate the relative weights within their respective domains  
\STATE $I_1', \ I_2',\ I_3' \gets \textbf{calculateProportion}(I_1, I_2, I_3)$ \% Use equation (18) to calculate the proportion   
\STATE $v^m \gets 0$  
\FOR{$x^n \in X$}  
    \STATE $A^n, R^n, T^n \gets \textbf{getValues}(x^n)$ \% Obtain the corresponding accurate reliability, service reliability, and delay from VALUES  
    \STATE $v^n \gets \textbf{calculateDeploymentValue}(A^n, R^n, T^n, I_1', I_2', I_3')$ \% Use equation (19) to calculate  
    \IF{$v^n > v^m$}  
        \STATE $v^m \gets v^n$  
        \STATE $x^m, A^m_i, R^m_i, T^m_i \gets x^n, A^n, R^n, T^n$  
    \ENDIF  
\ENDFOR  
\STATE \textbf{return} $x^m, A^m_i, R^m_i, T^m$  
\end{algorithmic}  
\end{algorithm}  
\subsection{Complexity}
\noindent
The RDODA algorithm consists of two steps: generating an initial population and performing evolutionary selection. Assume the population has a size of m, and there are n generations. Let the number of tasks be v, and the number of nodes be s. The complexity of generating the initial population is at the level of $O(m\cdot s\cdot v)$, where each population finds a heuristic strategy, simplifying the initial generation process. During each generation's evolutionary selection, genetic manipulation randomly generates sub-populations from the existing population, with the complexity of $O(m^2\cdot v^2\cdot s)$. In selecting the next generation population operation, the non-dominate ranking of the populations has a complexity of $O(m^2)$, and the complexity of calculating crowding distance is at the level of $O(m)$. Therefore, the complexity of selecting the next-generation populations is at the level of $O(m^2)$. So, the total complexity of genetic iteration is at the level of $O(n\cdot m^2\cdot v^2 \cdot s)$. The complexity of choosing the satisfy delay constraint solution from the population is at the level of $O(m)$. In summary, the overall time complexity of algorithm RDODA is at the level of $O(n\cdot m^2\cdot v^2 \cdot s)$.\par
The DSSA algorithm selects the best deployment solution by calculating the values of each solution in the solution set obtained by RDODA, with a time complexity of $O(m\cdot v)$.
```

