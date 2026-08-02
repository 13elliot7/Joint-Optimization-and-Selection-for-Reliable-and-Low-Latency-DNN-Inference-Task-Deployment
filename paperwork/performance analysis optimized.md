```latex
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
Default crossover probability & 0.50\\
\hline
\end{tabular}
\end{table}

\subsection{Metrics and Comparison Methods}
\noindent
The following metrics are used to evaluate different algorithms.

\textbf{Average predicted service reliability}: The average one-step post-admission service reliability of successfully accepted DNN requests, including the predicted reliability of the unique computing nodes and traversed input, dependency and output links.

\textbf{Average predicted inference accuracy}: The average one-step post-admission inference accuracy of successfully accepted DNN requests, weighted by the computational workload assigned to each used node.

\textbf{Average estimated inference delay}: The average estimated completion delay of successfully accepted DNN inference requests, including input uploading, critical-path processing, inter-layer transmission and result returning delay.

\textbf{Average total energy consumption}: The average computation energy plus input-upload and inter-layer transmission energy of successfully accepted DNN requests. Final result-return energy is excluded consistently with the energy model.

\textbf{Rejected or failed deployments}: The number of DNN requests that cannot be accepted due to resource, delay or search failure.

\textbf{Algorithm running time}: The wall-clock time of deployment search and online execution in one simulation run. Pareto-front serialization and hypervolume postprocessing are excluded.

\textbf{Four-dimensional hypervolume}: Hypervolume is calculated only in the dedicated Pareto-front experiment. For each DNN request, strictly feasible non-dominated candidates are represented by normalized inference accuracy, normalized service reliability, delay satisfaction and energy satisfaction. Delay satisfaction is $(T^i_{limit}-T_i)/T^i_{limit}$ clipped to $[0,1]$, and energy satisfaction is defined in \eqref{eq:energy_satisfaction}. The common reference point is $(0,0,0,0)$. Per-request HV values are averaged across DNN requests, and their sum and standard deviation are also recorded. The implementation marks these results as \texttt{4d\_v1}. No Pareto points or HV fields are generated for the overall, dynamic, sensitivity, ablation, scale or preference experiments. A larger hypervolume indicates a Pareto front with better convergence and diversity.

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
The Pareto front quality is evaluated using DAREED, NSGA-ED, RDA and SAA. Pareto dominance is performed over predicted inference accuracy, predicted service reliability, delay utility and raw total energy consumption. For cross-request HV calculation, the corresponding normalized dimensions are inference accuracy, service reliability, delay satisfaction and energy satisfaction. Thus, energy is part of both candidate search and four-dimensional HV evaluation.

\autoref{fig_pareto_3d} compares the nondominated solution sets produced by all algorithms for one matched representative repeat and DNN request. The three spatial axes show inference accuracy, service reliability, and delay satisfaction, while point color encodes the fourth objective, energy satisfaction. Black outlines identify solutions retained in the joint four-objective reference front formed from the union of all algorithms' solution sets. Separating algorithms into panels avoids mixing heterogeneous requests and repeated runs into a single apparent front.

\begin{figure}[!t]
\centering
\includegraphics[width=\columnwidth]{fig_pareto_3d.pdf}
\caption{Energy-encoded three-dimensional projections of four-objective nondominated solution sets for a median-size matched case. Black outlines denote the joint reference front.}
\label{fig_pareto_3d}
\end{figure}

To provide a quantitative comparison, \autoref{tab:pareto_hv} reports the average hypervolume, the average number of Pareto points per DNN request, and the standard deviation over repeated runs. The concrete values are left as placeholders before the final simulations are completed.

\begin{table}[!t]
\caption{Four-Objective Pareto Front Quality Comparison\label{tab:pareto_hv}}
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

\autoref{fig_dynamic_degradation} shows the performance of different algorithms under the three degradation levels. The expected output is a grouped bar chart covering service reliability, inference accuracy, average total energy consumption and failed deployment number. DAREED is expected to be more stable under strong degradation because candidate evaluation predicts the load and heat introduced by the candidate itself before computing service reliability and inference accuracy. Role-aware mutation and local refinement then use these predicted values together with delay, energy and feasibility information. The final conclusion should be stated only after paired-seed results are available.

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
To verify the contribution of each customized component in DAREED, we conduct an ablation study. The current executable ablation suite compares full DAREED with five variants: w/o DAG-aware initialization, w/o DAG block crossover, w/o skew mutation, w/o elite local search and a plain evolutionary variant that disables all four customized components. Because the updated full algorithm also contains persistent role specialization and one-step post-admission reliability prediction, two additional paired ablations should be reported before attributing gains to these mechanisms: w/o role specialization/migration and w/o post-admission prediction. Each ablation should use the same random seeds as full DAREED and report accepted-request reliability, accuracy, delay, total energy, failed deployments and runtime. These two additional variants require corresponding experiment switches before their results can be reported.

\autoref{fig_ablation} shows the ablation results. The expected output is a grouped bar chart. If the DAG-aware initialization is removed, the initial population may contain more low-quality candidates. If DAG block crossover is removed, offspring may lose dependency locality. If skew mutation is removed, the mutation operator cannot exploit the current dynamic resource state. If elite local search is removed, the final convergence quality may decline. Removing role specialization tests whether persistent search directions and migration improve diversity, while disabling post-admission prediction tests whether anticipating the candidate-induced load and heat improves reliability under dynamic workloads.

\begin{figure}[!t]
\centering
\includegraphics[width=2.6in]{fig_ablation.pdf}
\caption{Ablation study of DAREED components.}
\label{fig_ablation}
\end{figure}

\begin{table}[!t]
\caption{Ablation Study Results\label{tab:ablation}}
\centering
\resizebox{\columnwidth}{!}{%
\begin{tabular}{p{2.4cm}p{1.0cm}p{1.0cm}p{0.9cm}p{0.9cm}p{0.9cm}p{1.0cm}}
\hline
Variant & Service rel. & Infer. acc. & Delay & Energy & Failed & Runtime\\
\hline
Full DAREED & TODO & TODO & TODO & TODO & TODO & TODO\\
w/o DAG init. & TODO & TODO & TODO & TODO & TODO & TODO\\
w/o block crossover & TODO & TODO & TODO & TODO & TODO & TODO\\
w/o skew mutation & TODO & TODO & TODO & TODO & TODO & TODO\\
w/o elite local search & TODO & TODO & TODO & TODO & TODO & TODO\\
w/o role specialization & TODO & TODO & TODO & TODO & TODO & TODO\\
w/o post-admission prediction & TODO & TODO & TODO & TODO & TODO & TODO\\
Plain evolutionary & TODO & TODO & TODO & TODO & TODO & TODO\\
\hline
\end{tabular}
}
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
```
