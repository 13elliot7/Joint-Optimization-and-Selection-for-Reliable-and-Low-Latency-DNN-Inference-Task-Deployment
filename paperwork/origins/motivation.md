```latex
\section{Motivation}
\noindent
An instance of DNN inference task deployment investigated in this paper is depicted in \autoref{fig_1}, featuring three types of computing nodes: end devices, edge computing nodes, and a central cloud. An end device connects to an edge computing node, and certain edge computing nodes establish connections with the cloud. The end device initiates the DNN inference task. In this instance, DNN inference task 1 prioritizes delay sensitivity, task 2 places a premium on reliability, and task 3 demands a balance between delay and reliability. With a single-objective optimization approach, optimizing the deployment scheme for such DNN inference tasks proves challenging. While simple weighting methods can adjust optimization across different objectives, determining the most suitable weights becomes complex. As the number of objectives increases, weight determination grows more intricate, and variations in metrics further complicate it. In contrast, the multi-objective optimization method evaluates the Pareto solution set within the current context, enabling the selection of the most appropriate deployment strategy according to the characteristics of each DNN inference task. 
Taking inference task 3 as an illustration, as depicted in Figure 1-(b), focusing on optimizing for minimum delay might lead to deployment on either the end device or the edge node closest to the end device. In contrast, as shown in Figure 1-(c), prioritizing reliability optimization could result in complete deployment on the cloud due to its high-reliability nature. Subsequently, as demonstrated in figure 1-(d), considering multiple objectives may yield diverse deployment schemes based on different algorithms. 
Therefore, this paper endeavors to address these challenges through multi-objective optimization, aiming to identify the Pareto solution set for each DNN inference task. Following this, the paper seeks to choose the most appropriate deployment configuration tailored to the specific requirements of the DNN inference task.

\begin{figure}[!t]
\centering
\includegraphics[width=2.5 in]{fig 1.jpg}
\caption{Example of DNN inference task deployment}
\label{fig_1}
\end{figure}
```

