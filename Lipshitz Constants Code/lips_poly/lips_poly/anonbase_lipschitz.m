function Lf = anon123_lipschitz(W0, W1, n0, n1, verbose)
    %
    % params:
    %   weights: cell   - weights of neural network
    %   n0: int         - input dimension of neural network
    %   n1: int         - output dimension of neural network
    %   verbose: bool   - controls how much output is printed
    %
    % returns:
    %   Lf: float - Lipschitz constant of neural network

    % V0 matrix definition
    V0 = [W0 zeros(n1, n1); zeros(n1, n0)  eye(n1)];

    % CVX problem formulation
    alpha = 0; beta = 1;

    if verbose == true
        cvx_begin sdp
    else
        cvx_begin sdp quiet
    end
    
    cvx_solver mosek

    variable D(n1,1) nonnegative
    variable L_sq nonnegative

    % Q matrix parameterized by D
    Q = [-2*alpha*beta*diag(D) (alpha+beta)*diag(D);(alpha+beta)*diag(D) -2*diag(D)];

    % M is inner matrix in second term of LMI
    M = [L_sq * eye(n0)   zeros(n0, n1); zeros(n1,n0) -W1'*W1];

    % Goal is to minimize the Lipschitz constant (L)
    minimize L_sq

    % LMI for minimization problem
    subject to
        (V0' * Q * V0) - M <= 0;
    cvx_end

    Lf = sqrt(L_sq);
    
end

