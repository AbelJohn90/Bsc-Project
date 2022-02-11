function L = anonbase_lip_multi(weights, network_dims, verbose, lower, upper,...
    alpha, mode, num_neurons)
    % Anonbase's formulation of finding Lipschitz constant of NN
    % Computes Lipschitz constant according to {mode} parameter
    %
    % params:
    %   weights: cell               - weights of neural network
    %   network_dims: list of ints  - dimensions of neural network
    %   verbose: bool               - controls how much output is printed
    %   lower: float                - naive lower bound for Lip constant
    %   upper: float                - naive upper bound for Lip constant
    %   alpha: float                - sector bounded lower value
    %   mode: str                   - 'repeated'|'non-repeated'|'reduced'
    %
    % returns:
    %   L: float - Lipschitz constant of neural network
    
    warning('off', 'all');
    
    N = sum(network_dims(2:end-1));  % total hidden neurons
    beta = 1;    % for ReLU constraint
    
    if verbose == true
        cvx_begin sdp
    else
        cvx_begin sdp quiet
    end
    
    cvx_solver mosek
    
    % Create Q matrix for ReLU constraint
    
    % Repeated mode - one CVX variable per hidden neuron and (N choose 2)
    % variables to parameterize T matrix that accounts for repeated 
    % nonlinearaities.  This mode of operation is computationally much more
    % expensive, but gives a better lower bound
    if strcmp(mode, 'repeated')
        variable D(N, 1) nonnegative
        variable zeta(N*(N-1)/2, 1) nonnegative
        
        id = eye(N);
        T = diag(D);
        C = nchoosek(1:N, 2);
        E = id(:, C(:, 1)) - id(:, C(:, 2));
        T = T + E * diag(zeta) * E';
        Q = [-2*alpha*beta*T (alpha+beta)*T;(alpha+beta)*T -2*T];
        
%         counter = 1;
%         for i = 1:N
%             for j = i+1:N
%                 T = T + zeta(counter) * ((id(:,i) - id(:,j)) *  (id(i,:) - id(j,:)));
%                 counter = counter + 1;
%             end
%         end
        
    % Repeated-rand mode: uses repeated nonlinearities with a random subset
    % of coupled neurons from the entire set of N choose 2 total neurons
    elseif strcmp(mode, 'repeated-rand')
        
        variable D(N, 1) nonnegative
        variable zeta(num_neurons, 1) nonnegative
        
        id = eye(N);
        T = diag(D);
        C = nchoosek(1:N, 2);
        
        % take a random subset of neurons to couple
        k = randperm(size(C, 1));
        C = C(k(1:num_neurons), :);
        
        E = id(:, C(:, 1)) - id(:, C(:, 2));
        T = T + E * diag(zeta) * E';
        
        Q = [-2*alpha*beta*T (alpha+beta)*T;(alpha+beta)*T -2*T];
    
    % Non-repeated mode - one CVX variable per hidden neuron in the network
    % This is the usual mode of operation
    elseif strcmp(mode, 'non-repeated')
        
        variable D(N, 1) nonnegative
        Q = [-2*alpha*beta*diag(D) (alpha+beta)*diag(D);(alpha+beta)*diag(D) -2*diag(D)];
        
    % Reduced option - one CVX variable per hidden layer
    % Should closely approximate Combettes result
    elseif strcmp(mode, 'reduced')
        
        n_hid = length(network_dims) - 2;
        variable D(n_hid, 1) nonnegative
        
        for i = 1:n_hid
            identities{i} = D(i) * eye(network_dims(i+1));
        end
        
        D_mat = blkdiag(identities{:});
        Q = [-2*alpha*beta*D_mat (alpha+beta)*D_mat;(alpha+beta)*D_mat -2*D_mat];
    
    else
       error('Please set mode == "repeated" | "non-repeated" | "reduced"') 
    end    
    
    variable L_sq nonnegative

    % Create A term in Lipschitz formulation
    first_weights = blkdiag(weights{1:end-1});
    zeros_col = zeros(size(first_weights, 1), size(weights{end}, 2));
    A = horzcat(first_weights, zeros_col);
        
    % Create B term in Lipschitz formulation
    eyes = eye(size(A, 1));
    init_col = zeros(size(eyes, 1), network_dims(1));
    B = horzcat(init_col, eyes);
    
    A_on_B = vertcat(A, B);
        
    % Create M matrix encoding Lipschitz constant
    weight_term = -1 * weights{end}' * weights{end};
    middle_zeros = zeros(sum(network_dims(2:end-2)), sum(network_dims(2:end-2)));    
    lower_right = blkdiag(middle_zeros, weight_term);
    upper_left = L_sq * eye(network_dims(1));
        
%     M = blkdiag(upper_left, lower_right);
    M = cvx(zeros(size(upper_left, 1) + size(lower_right, 1), size(upper_left, 2) + size(lower_right, 2)));
    M(1:size(upper_left, 1), 1:size(upper_left, 1)) = upper_left;
    M(size(upper_left, 1) + 1:end, size(upper_left, 2) + 1:end) = lower_right;
        
    % Solve optimizaiton problem - minimize squared Lipschitz constant
    minimize L_sq
    
    % LMI for minimization problem
    if upper < -1 || lower < -1
        subject to
            (A_on_B' * Q * A_on_B) - M <= 0;
            lower^2 <= L_sq <= upper^2;
        cvx_end
    else
        subject to
            (A_on_B' * Q * A_on_B) - M <= 0;
        cvx_end
    end
    
    L = sqrt(L_sq);
    
end
