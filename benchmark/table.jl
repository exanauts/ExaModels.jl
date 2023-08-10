using LaTeXStrings, Printf

function varcon(n)
    if n <= 1000
        "$n"
    elseif n <= 1000000'
        @sprintf("%5.1fk", n/1000)
    else
        @sprintf("%5.1fm", n/1000000)
    end
end
fmt(t) = @sprintf("%5.2f", t)
efmt(t) = @sprintf("%1.1e", t)
percent(t) = @sprintf("%5.1f", t * 100) * "\\%"

template = L"""
\documentclass{standalone}
\usepackage{lscape}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{times}

\begin{document}
\centering
\begin{tabular}{|c|c|c|ccc|ccc|ccc|ccc|ccc|}
  \hline
  \multirow{2}{*}{\textbf{Cases}}
  & \multirow{2}{*}{nvars}
  & \multirow{2}{*}{ncons}
  & \multicolumn{3}{c|}{\textbf{ExaModels} (single)}
  & \multicolumn{3}{c|}{\textbf{ExaModels} (multi)}
  & \multicolumn{3}{c|}{\textbf{ExaModels} (GPU)}
  & \multicolumn{3}{c|}{\textbf{JuMP}}
  & \multicolumn{3}{c|}{\textbf{AMPL}}\\
  \cline{4-18}
  & & 
  & grad & jac & hess 
  & grad & jac & hess 
  & grad & jac & hess 
  & grad & jac & hess 
  & grad & jac & hess \\
  \hline
  %% data %%
  \\
  \hline
\end{tabular}
\end{document}
$$
"""

tbl = join((
"""
$(mod(i,3) == 1 ? "\\hline" : "")
$(s.name )
& $(varcon(s.nvar))
& $(varcon(s.ncon))
& $(efmt(s.te.tgrad))
& $(efmt(s.te.tjac))
& $(efmt(s.te.thess))
& $(efmt(s.tec.tgrad))
& $(efmt(s.tec.tjac))
& $(efmt(s.tec.thess))
& $(efmt(s.teg.tgrad))
& $(efmt(s.teg.tjac))
& $(efmt(s.teg.thess))
& $(efmt(s.tj.tgrad))
& $(efmt(s.tj.tjac))
& $(efmt(s.tj.thess))
& $(efmt(s.ta.tgrad))
& $(efmt(s.ta.tjac))
& $(efmt(s.ta.thess))
"""
        for (i,s) in enumerate(save)
), "\\\\\n")


write(
    "result-1.tex",
    replace(
        template,
        "%% data %%" => replace(
            tbl,
            "_" => "\\_"
        )
    )
)

run(`pdflatex result-1.tex`)
