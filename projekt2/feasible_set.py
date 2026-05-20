from sage.all import * 
from itertools import combinations
from string import ascii_lowercase
from itertools import product

def random_problem(random_seed = None, rows = 4, cols = 5):
    """Creates random matrix with given dimenstions using given random seed."""
    
    from sage.matrix.constructor import random_echelonizable_matrix
    matrix_space = sage.matrix.matrix_space.MatrixSpace(QQ, rows, cols)
    if random_seed is not None:
        set_random_seed(random_seed)
    A = random_echelonizable_matrix(matrix_space, rank=4, upper_bound=40)
    b = random_vector(QQ, 4)
    c = random_vector(QQ, 5)

    P = InteractiveLPProblemStandardForm(A, b, c)

    
    return P

def basic_solutions(P: InteractiveLPProblemStandardForm):
    """Compute dictionary of basic feasible solutions of P indexed by basic sets, i.e.
       a mapping <basic set> -> <basic solution>."""
    
    A = P.A()
    n = A.ncols() # number of variables
    m = A.nrows() # number of constraints
    A = A.augment(identity_matrix(m))
    S = dict()
    
    b = P.b()
    for BasicSet in combinations(range(n + m), m):
        AB = A.matrix_from_columns(BasicSet)
        if AB.det() != 0:
            # We found a basic set
           
            x = AB.inverse() * b
            if min(x) >= 0:
                # We found a feasible basic solution
                
                S[BasicSet] = vector(QQ, [x[BasicSet.index(i)] if i in BasicSet else 0 for i in range(n+m)])
    return S


def solution_graph(P: InteractiveLPProblemStandardForm):
    r"""Create a graph (V, E) where
    
    V = set of basic feasible sets of P (vertices of the solution graph),
    E = pairs (B1, B2) of basic feasible sets of P such that #(B1 \ B2) = 1 and #(B2 \ B1) = 1.
    
    The edges are oriented in the direction of the gradient of the objective function.
    """
    
    c = vector(QQ, P.c().list() + ([0] * P.A().nrows()))
    
    S = basic_solutions(P)
    V = set(S)
    E = list()
    for B1 in V:
        for B2 in V:
            S1 = set(B1)
            S2 = set(B2)
            if len(S1.difference(S2)) == 1 and len(S2.difference(S1)) == 1:
                if c * S[B1] <= c * S[B2]:
                    E.append((B1, B2, c * S[B2] - c * S[B1]))
   
    g = DiGraph(E)
    return g

def feasible_polyhedron(P):
    """
    Returns feasible polyhedron of problem P (converted to equational form).
    """
    
    A = P.A()
    b = P.b()

    A = A.augment(identity_matrix(A.nrows()))
    
    eqns = []
    for row, coeff in zip(A, b):    
        eqns.append([-coeff] + list(row))

    ieqs = []
    for i in range(A.ncols()):
        v = vector(QQ, A.ncols())
        v[i] = 1
        ieqs.append([0] + list(v))
    
    return Polyhedron(ieqs=ieqs, eqns=eqns)

def labelled_solution_graph(P: InteractiveLPProblemStandardForm):
    c = vector(QQ, P.c().list() + ([0] * P.A().nrows()))
    
    vname = []
    for length in range(1, 3):
        for combo in product(ascii_lowercase, repeat=length):
            vname.append(''.join(combo))
    
    names = {}
    
    S = basic_solutions(P)
    E = list()
    for n1, B1 in enumerate(S):
        names[B1] = vname[n1]
        
        for n2, B2 in enumerate(S):
            S1 = set(B1)
            S2 = set(B2)
            if len(S1.difference(S2)) == 1 and len(S2.difference(S1)) == 1:
                if c * S[B1] <= c * S[B2]:
                    E.append((vname[n1], vname[n2], c * S[B2] - c * S[B1]))
   
    g = DiGraph(E)
    return g, names    

    