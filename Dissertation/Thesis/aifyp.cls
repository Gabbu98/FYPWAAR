\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{aifyp}[2017/07/14 FYP Report v0.2]

%
% Changelog
%
% v0.2
% * Added front page
% * Added long abstract
% * Added table of contents to example paper
%

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions \relax

\LoadClass[12pt,twoside]{article}
\RequirePackage[onehalfspacing]{setspace}
\RequirePackage[top=1.2in, left=2.7cm, bottom=1.2in, right=2.7cm]{geometry}
\RequirePackage{calc}

\RequirePackage[bottom]{footmisc}

\RequirePackage{caption} \captionsetup[table]{skip=1ex}

\def\@supervisor{\relax}
\newcommand{\supervisor}[1]{\gdef\@supervisor{#1}}

\def\@longabstract{\relax}
\newcommand{\longabstract}[1]{\gdef\@longabstract{#1}}

\def\@acknowledgement{\relax}
\newcommand{\acknowledgement}[1]{\gdef\@acknowledgement{#1}}

\def\@degree{B.Sc. ICT in Artificial Intelligence (Hons.)}
\newcommand{\degree}[1]{\gdef\@degree{#1}}

\renewcommand\maketitle{\par
  \begingroup
    \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
    \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
    \long\def\@makefntext##1{\parindent 1em\noindent
      \hb@xt@1.8em{%
        \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
        \if@twocolumn
        \ifnum \col@number=\@ne
        \@maketitle
        \else
        \twocolumn[\@maketitle]%
        \fi
        \else
        \newpage
        \global\@topnum\z@   % Prevents figures from going at top of page.
        \@maketitle
        \fi
        \thispagestyle{plain}\@thanks
        \endgroup
        \setcounter{footnote}{0}%
        \global\let\thanks\relax
        \global\let\maketitle\relax
        \global\let\@maketitle\relax
        \global\let\@thanks\@empty
        \global\let\@author\@empty
        \global\let\@date\@empty
        \global\let\@title\@empty
        \global\let\title\relax
        \global\let\author\relax
        \global\let\date\relax
        \global\let\and\relax
}
\def\@maketitle{%
  \newpage
    \null
    \vskip 2em%
    \begin{center}%
    \LARGE{\textbf \@title\if\relax\@acknowledgement\else\footnote{\@acknowledgement}\fi}\par
    \large\@author\footnote{Submitted in partial fulfillment of the requirements for the degree of \@degree.}\par
    \normalsize \textbf{Supervised by:} \@supervisor\par
    \@date \par
    \end{center}
}

\newtheorem{theorem}{Theorem}[section]
\newtheorem{conjecture}[theorem]{Conjecture}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{definition}[theorem]{Definition}



\AtBeginDocument{
  %\newsavebox{\CSAItitletempbox} \sbox{\CSAItitletempbox}{\parbox{\textwidth - 1.5in}{
  %  \centering \LARGE \bf \@title}
  %}

  % Do some usage checking
  %% Some measurements to format the title page properly
  \newlength{\textwidththird}
  \newlength{\topheight}
  \newlength{\topleftwidth}
  \newlength{\topmiddlewidth}
  \newlength{\toprightwidth}
  \setlength{\textwidththird}{(\textwidth / 3) - (\fboxsep * 2)}
  \setlength{\topleftwidth}{\the\textwidththird}
  \setlength{\topmiddlewidth}{\the\textwidththird}
  \setlength{\toprightwidth}{\the\textwidththird}
  \addtolength{\topleftwidth}{-3em}
  \addtolength{\topmiddlewidth}{+4em}
  \addtolength{\toprightwidth}{-3em}

  %%% Front Cover
  {
    \thispagestyle{empty}
    \begin{center}
    \textbf{\huge\@title\\[0.25in]}
    \textbf{\large\@author\\[0.25in]}
    {\large \textbf{Supervisor(s):} \@supervisor\\[0.65in]}
    \vfill
    \scalebox{0.8}{\includegraphics{UMLOGOonly_black.png}}
    \vspace{0.25in}\\
    \textbf{\large Faculty of ICT\\[0.05in]}
    \textbf{\large University of Malta\\[0.25in]}
    {\large \@date}
    \vspace{1in}\\
    \vfill
    \textit{Submitted in partial fulfillment of the requirements for the degree of B.Sc. ICT in Artificial Intelligence (Hons.)}
    \end{center}

    \addtolength{\headheight}{0.5in}
    \newpage
    
		\thispagestyle{empty}
    \vspace*{-5cm}\hspace*{-2.7cm}\scalebox{0.9}{\includegraphics{DeclerationOfAuthenticity.pdf}}

    \addtolength{\headheight}{0.5in}


    \thispagestyle{empty}
    \begin{center}\textbf{Abstract:}\end{center}
    \vspace{0.125in}
    \@longabstract

    \clearpage{\pagestyle{empty}}

    \thispagestyle{empty}
    \begin{center}\textbf{Acknowledgements:}\end{center}
    \vspace{0.125in}
    \@acknowledgement

    \clearpage{\pagestyle{empty}}

    \setcounter{page}{1}
  }
} %AtBeginDocument




