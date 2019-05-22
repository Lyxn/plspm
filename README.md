## Partial Least Squares Path Modeling

### Introduction

This is a python implemention of **Partial Least Squares Path Modeling** for metric data, 
based on [plspm of R version][R-plspm].

[Path analysis][PathAnalysis] is also known as **Structural Equation Modeling(SEM)**. 
It can model causality, on which statistical model would not work.

Further, [Judea Pearl][JudeaPearl]'s causal inference is at the cutting edge of causal analysis, 
which derived from bayesian network and has attracted many people's attention, especially the one who work on social science.
Pearl's book ["The Book of Why"][WHY] has a introduction to the new paradigm.

[R-plspm]: https://github.com/gastonstat/plspm
[PathAnalysis]: https://en.wikipedia.org/wiki/Path_analysis_(statistics)
[JudeaPearl]: http://bayes.cs.ucla.edu/home.htm
[WHY]: http://bayes.cs.ucla.edu/WHY/

### Code

#### plspm.py
Core program

#### test_plspm.py
Test Program

#### utils.py
Utils

### Reference

1. http://www.gastonsanchez.com/PLS_Path_Modeling_with_R.pdf
1. http://causality.cs.ucla.edu/blog/
