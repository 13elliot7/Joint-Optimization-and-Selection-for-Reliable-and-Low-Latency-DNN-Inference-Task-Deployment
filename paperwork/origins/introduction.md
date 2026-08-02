```latex
\documentclass[lettersize,journal]{IEEEtran}
\usepackage{amsmath,amsfonts}
\usepackage{algorithmic}
\usepackage{algorithm}
\usepackage{array}
\usepackage[caption=false,font=normalsize,labelfont=sf,textfont=sf]{subfig}
\usepackage{textcomp}
\usepackage{stfloats}
\usepackage{url}
\usepackage{verbatim}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{titlesec} % 导入 titlesec 包  
\usepackage{hyperref}
\usepackage{balance}
% \def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
%     T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}
% \usepackage[numbers]{natbib}
\hypersetup{hypertex=true,
            colorlinks=true,
            linkcolor=blue,
            anchorcolor=blue,
            citecolor=blue}
\providecommand{\algorithmautorefname}{Algorithm}
% 自定义 subsection 样式  
% \titleformat{\subsection}  
%   {\bfseries} % 设置为加粗  
%   {\thesubsection} % 编号格式  
%   {1em} % 编号和标题之间的距离  
%   {} % 标题样式  
% \usepackage{cite}
% \titleformat{\section}  
%   {\bfseries\Large} % 设置加粗和字体大小  
%   {\thesection}     % 编号的格式  
%   {1em}             % 编号和标题之间的距离  
%   {}                % 标题样式
% \titleformat{\subsubsection}  
%   {\normalfont\itshape} % 设置为斜体  
%   {\thesubsubsection} % 编号格式  
%   {1em} % 编号和标题之间的距离  
%   {} % 标题样式  
%   \titlespacing*{\subsubsection}{0pt}{0pt}{0pt}
\hyphenation{op-tical net-works semi-conduc-tor IEEE-Xplore}
% updated with editorial comments 8/9/2021
\usepackage{lineno}  
\begin{document}

\title{A Joint Optimization and Selection for Reliable and Low-Latency DNN Inference Task Deployment in Edge Computing}

\author{Ying Wang,~\IEEEmembership{Member,~IEEE,} Lianze Huang, Junjie Wei, Manjun Zhang, Peng Yu,~\IEEEmembership{Senior Member,~IEEE}, Xuesong Qiu,~\IEEEmembership{Senior Member,~IEEE}, Shaoyong Guo,~\IEEEmembership{Member,~IEEE}
        % <-this % stops a space
\thanks{%Manuscript received 28 February 2024; revised 24 June 2024 and 13 July2024; accepted 28 August 2024. Date of publication 3 September 2024; dateof current version 6 December 2024. 
This work was supported in part by the National Natural Science Foundation of China under Grant 62322103; in part by the Beijing Natural Science Foundation under Grant 4232009; and in part by the Fund of Central University Basic Research Projects under Grant 2023ZCTH11.}% <-this % stops a space
\thanks{The authors are with the State Key Laboratory of Networking and Switching
Technology, Beijing University of Posts and Telecommunications, Beijing
100876, China (e-mail: wangy@bupt.edu.cn).}
}

% The paper headers
\markboth{A Joint Optimization and Selection for Reliable and Low-Latency DNN Inference Task Deployment in Edge Computing}%
{Shell \MakeLowercase{\textit{WANG et al.}}: A Sample Article Using IEEEtran.cls for IEEE Journals}

%\IEEEpubid{0000--0000/00\$00.00~\copyright~2021 IEEE}
% Remember, if you use this you must call \IEEEpubidadjcol in the second
% column for its text to clear the IEEEpubid mark.

\maketitle

\begin{abstract}
Intelligent services based on deep neural networks (DNNs) that offer real-time inference and decision-making capabilities have garnered significant attention. Shifting computing-intensive tasks to mobile edge computing (MEC) networks introduces a novel paradigm for deploying DNN inference tasks.
Nonetheless, breakdowns in edge computing nodes or end devices can cause DNN inference task failures, while execution delays significantly affect service performance. Additionally, traditional single-objective optimization methods for DNN inference deployment fail to meet the diverse needs of varying scenarios. To address the above issues, we propose a multi-objective DNN inference task deployment approach utilizing NSGA-II to optimize both reliability and delay in MEC networks. We first formulate the problem model for deploying DNN inference tasks across cloud, edge, and end, which takes service reliability, accuracy reliability, and delay as multiple optimization objectives. Subsequently, we design a reliability and delay-oriented DNN inference task deployment algorithm based on NSGA-II (RDODA) to optimize DNN inference task deployment by seeking Pareto solution sets. Moreover, we develop a deployment scheme selection algorithm based on expectation (DSSA) to identify appropriate deployment schemes from the Pareto solution set. Experimental results show that the RDODA algorithm outperforms benchmark algorithms RDA and SSA, achieving a better Hypervolume value. Compared to reference algorithms RTBL, LB, RD, and LPD, RDODA+DSSA improves service reliability and accuracy reliability by 3.4\% to 19.6\% and 5.8\% to 27.4\% respectively. In terms of delay, RDODA and DSSA together outperform RD and LB by 5.8\% and 13.7\%.
\end{abstract}

\begin{IEEEkeywords}
DNN Inference Task, Multi-Objective, Reliability, Delay, Deployment
\end{IEEEkeywords}

\section{Introduction}
\IEEEPARstart{I}{n} recent years, artificial intelligence technologies, particularly deep learning, have catalyzed the proliferation of numerous intelligent mobile applications\cite{ref1}, owing to their strong feature learning capabilities and high inference accuracy\cite{ref2}. Notably, intelligent services leveraging deep neural networks (DNNs) for real-time inference and decision-making have garnered significant attention and adoption, spanning domains like intelligent security and intelligent inspection \cite{ref3}-\cite{ref4}. DNN inference involves deriving computational results through tightly integrated computational tasks with the DNN model \cite{ref5}. This inference process typically encompasses two stages, model training and online inference. During model training, the DNN model undergoes iterative training to adjust its weights and is then stored as an accessible service model on a cloud server\cite{ref6}. Subsequently, online inference entails deploying the trained models on end devices, edge servers, or cloud servers to execute DNN inference tasks, delivering users with intelligent services that facilitate swift decision-making and real-time inference. The quality of service provided to users is largely determined by the deployment strategies. Mobile edge computing, leveraging network edge computing and storage resources, emerges as a promising technology \cite{ref7} paradigm, ushering in a fresh approach to DNN inference task deployment. Consequently, the deployment of DNN inference tasks, particularly in the context of online collaborative DNN inference involving cloud, edge, and end, has become a research focus \cite{ref8} - \cite{ref12}. Nevertheless, during the service process, challenges persist due to limited edge resources and computing node failures, leading to reliability and execution delays concerns for DNN inference tasks. A detailed examination is provided below:\par
a. Reliability of DNN inference tasks: Reliability plays a critical role in the execution of DNN inference tasks. Computing node failure or inadequate computing resources leading to diminished accuracy, are likely to have a significant impact on the delivery of inference services, resulting in unreliable inference results or complete service failure. However, whether it is the deployment of DNN inference tasks or the reconstruction of DNN inference tasks, current research centers on performance optimization concerning cost or delay, overlooking crucial aspects of reliability, making it difficult to apply in fields with high reliability and stability requirements.\par
b. Multi-dimensional optimization of DNN inference task deployment: Present research mostly concentrates on optimizing DNN inference task deployment with a singular objective, such as minimizing delay, minimizing power consumption, etc. Some studies integrate multiple optimization objectives through weighted sums to construct a function that is conducive to multi-objective optimization. The evolving diversity and volume of services introduce varying and distinct performance requirements. Techniques reliant on single or multi-objective weighted optimization face challenges in selecting the optimal solution solely based on a single objective or fixed weights. Balancing heterogeneous metrics in practice remains challenging.\par
Based on the above analysis, this paper studies the deployment of DNN inference tasks for multi-objective optimization of reliability and delay in edge computing. Unlike most existing works that only consider a single optimization objective or indirectly achieve multiple objectives through weighting, we treat reliability and execution delay as distinct optimization objectives within the problem domain to obtain the Pareto solution set of the problem, thereby improving the adaptability of DNN inference task deployment decisions. The contributions of this paper are outlined as follows.\par
    • We propose a problem model for deploying DNN inference tasks aimed at optimizing reliability and delay. A resource model for DNN inference task deployment comprising central cloud-edge computing nodes and end devices is introduced. Delay and reliability models for DNN inference tasks are designed. Subsequently, a problem model for deploying DNN inference tasks optimized with regard to both reliability and delay is formulated.\par
    • We design a multi-objective DNN inference task deployment algorithm based on NSGA-II(RDODA). This algorithm focuses on optimizing inference accuracy, service reliability, and delay of DNN inference tasks. By seeking the Pareto solution through multi-objective optimization rather than relying on single-objective optimization or weighting, the algorithm improves optimization efficiency in practical scenarios and better adapts to evolving demands.\par
    • We propose a heuristic deployment scheme selection algorithm based on expectation (DSSA). By assigning weights derived from the delay and reliability requirements of DNN inference tasks, we explore optimal deployment strategies within the Pareto solution set, offering tailored deployment plans for DNN inference tasks.\par
The remainder of this paper is organized as follows: Section 2 reviews the research on DNN inference task deployment and multi-objective optimization. Section 3 introduces our research motivation. Section 4 formulates the problem model. In Section 5, a multi-objective optimization algorithm for DNN inference task deployment, focusing on reliability and delay optimization based on NSGA-II, is proposed. Building upon this algorithm, a heuristic algorithm for selecting DNN inference task deployment strategies based on expectation is proposed. Section 6 evaluates the proposed algorithm. Finally, Section 7 summarizes this paper.
```

