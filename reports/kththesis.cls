\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{kththesis}[2016/06/21 KTH Thesis formatting /Örjan Ekeberg]

\newif\ifinswedish
\DeclareOption{english}{}
\DeclareOption{swedish}{\inswedishtrue}

\newif\ifbachelor
\DeclareOption{bachelor}{\bachelortrue}
\DeclareOption{master}{}

%% Send any unknown option to the report class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}

\ProcessOptions\relax

\LoadClass[11pt]{report}

%% Load other packages
\RequirePackage[swedish,english]{babel} % International language support
\RequirePackage[a4paper,twoside,
  top=30mm,bottom=30mm,inner=36mm,outer=18mm,
  headsep=10mm,headheight=5mm]{geometry} % Set paper and contents dimensions
\RequirePackage{fancyhdr} % Take control of headers and footers
\RequirePackage{emptypage} % Supress page numbers on otherwise empty pages
\RequirePackage{titlesec} % Redefine the appearance of headers
\RequirePackage{ragged2e} % Ragged right margin is preferred for readability
\RequirePackage{tocbibind} % Include the bibliography in the table of contents

% Make it possible to conditionally depend on the TeX-engine used
\RequirePackage{ifxetex}
\RequirePackage{ifluatex}
\newif\ifxeorlua
\ifxetex\xeorluatrue\fi
\ifluatex\xeorluatrue\fi

\ifxeorlua
  \RequirePackage{fontspec}
  % this enebles TeX style ligatures such as ---, '', ``, and so on
  \defaultfontfeatures{
    Ligatures={TeX}
  }
  \setmainfont{FreeSerif}
  \setsansfont{FreeSans}
  \setmonofont{FreeMono}
\else
  \RequirePackage[T1]{fontenc}
  \RequirePackage{palatino}
\fi

% Set up the header and footer
\fancyhead{}
\fancyhead[RO]{\sffamily\small\leftmark\qquad\thepage}
\fancyhead[LE]{\sffamily\small\thepage\qquad\leftmark}
\fancyfoot{}
\renewcommand{\headrulewidth}{0pt}
\pagestyle{fancy}

% Add 5% extra linespacing
\linespread{1.05}

% Use ragged right margin and indented first lines of paragraphs
\setlength\RaggedRightParindent{\parindent}
\RaggedRight

% Set the proper format for the headers
\titleformat{\chapter}[display]
  {\normalfont\sffamily\huge\bfseries}
  {\chaptertitlename\ \thechapter}{20pt}{\Huge}
\titleformat{\section}
  {\normalfont\sffamily\Large\bfseries}
  {\thesection}{1em}{}
\titleformat{\subsection}
  {\normalfont\sffamily\large\bfseries}
  {\thesubsection}{1em}{}

\renewenvironment{abstract}{\section*{\abstractname}}{\clearpage}

% Define commands for setting the user definable parts of the title page
\newcommand{\supervisor}[1]{\def\@supervisor{#1}}
\newcommand{\examiner}[1]{\def\@examiner{#1}}
\let\@secondauthor\@empty
\newcommand{\secondauthor}[1]{\def\@secondauthor{#1}}
\let\@subtitle\@empty
\newcommand{\subtitle}[1]{\def\@subtitle{#1}}
\let\@alttitle\@empty
\newcommand{\alttitle}[1]{\def\@alttitle{#1}}
\let\@principal\@empty
\newcommand{\principal}[1]{\def\@principal{#1}}
\let\@email\@empty
\newcommand{\email}[1]{\def\@email{#1}}
\let\@school\@empty
\newcommand{\school}[1]{\def\@school{#1}}
\let\@programme\@empty
\newcommand{\programme}[1]{\def\@programme{#1}}

% Command to print out the standardized flyleaf page
\newcommand{\flyleaf}{
\newgeometry{top=65mm,bottom=30mm,left=62mm,right=18mm}
\thispagestyle{empty}
\ifinswedish\selectlanguage{swedish}\fi
\begin{huge}
  \noindent\sffamily\bfseries \@title \par
\end{huge}
\ifx\@subtitle\@empty\relax
\else
\begin{Large}
  \vspace{1ex}
  \noindent\sffamily\bfseries \@subtitle \par
\end{Large}
\fi

\vspace{10mm}
\begin{large}
  \begin{flushleft}
    \sffamily
    \expandafter\MakeUppercase\expandafter\@author \par
    \ifx\@secondauthor\@empty\relax
    \else\vspace{1ex}\expandafter\MakeUppercase\expandafter\@secondauthor \par
    \fi
  \end{flushleft}
\end{large}

\vfill

\begin{flushleft}
  \sffamily
  \ifinswedish
    \@programme\\
    \ifx\@date\@empty\relax\else Datum: \@date\\\fi
    Handledare: \@supervisor\\
    Examinator: \@examiner\\
    \ifx\@principal\@empty\relax\else Uppdragsgivare: \@principal\\\fi
    \ifx\@alttitle\empty\relax\else Engelsk titel: %
    \begin{otherlanguage}{english}\@alttitle\end{otherlanguage}\\\fi
    \@school\\
  \else
    \@programme\\
    \ifx\@date\@empty\relax\else Date: \@date\\\fi
    Supervisor: \@supervisor\\
    Examiner: \@examiner\\
    \ifx\@principal\@empty\relax\else Principal: \@principal\\\fi
    \ifx\@alttitle\empty\relax\else Swedish title: %
    \begin{otherlanguage}{swedish}\@alttitle\end{otherlanguage}\\\fi
    \@school\\
  \fi
\end{flushleft}
\restoregeometry

\pagenumbering{roman}
\clearpage
}


\newcommand{\frontmatter}{} % NoOP

\newcommand{\mainmatter}{
  \cleardoublepage
  \pagenumbering{arabic}
}

\endinput
